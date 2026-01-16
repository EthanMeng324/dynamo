# KV Events 信息流详解

## 核心机制

**KV Events 是推送模式（Push-based），不是拉取模式（Pull-based）！**

- **Workers 主动推送** KV events（当 GPU 中的 KV cache blocks 被创建/删除时）
- **Router 被动接收**这些 events
- **Router 不需要主动查询** workers

## 完整信息流

### 1. vLLM Worker 端

#### vLLM 内部机制

vLLM 使用 **KVEventsConfig** 配置 KV events 发布：

```python
# components/src/dynamo/vllm/args.py:392-396
KVEventsConfig(
    enable_kv_cache_events=True,
    publisher="zmq",  # 使用 ZMQ 发布
    endpoint=f"tcp://*:{port - dp_rank}",
)
```

**vLLM 内部流程：**
1. vLLM 的 KV cache manager（PagedAttention）管理 GPU 内存中的 KV cache blocks
2. 当 block 被创建/删除时，vLLM **内部自动**发送 KV events
3. 通过 ZMQ PUB socket 发布到指定 endpoint

**关键点：**
- vLLM 的 `get_kv_cache_events_async()` 是 vLLM 库内部的方法
- 这个方法从 vLLM 的 KV cache manager 获取 events
- **不是 Dynamo 的代码，是 vLLM 库提供的功能**

#### Dynamo Worker 端订阅

```python
# components/src/dynamo/vllm/main.py:169-235
def setup_kv_event_publisher(...):
    # 创建 ZmqKvEventPublisher
    zmq_config = ZmqKvEventPublisherConfig(
        worker_id=generate_endpoint.connection_id(),
        kv_block_size=vllm_config.cache_config.block_size,
        zmq_endpoint=zmq_endpoint,  # 订阅 vLLM 的 ZMQ endpoint
    )
    kv_publisher = ZmqKvEventPublisher(component=component, config=zmq_config)
```

**工作流程：**
```
vLLM Engine (GPU)
    ↓ (内部自动发送)
ZMQ PUB Socket (vLLM)
    ↓ (ZMQ 消息)
ZmqKvEventPublisher (Dynamo Worker)
    ↓ (转换为 RouterEvent)
NATS JetStream
    ↓ (持久化)
Router Subscriber
```

### 2. TensorRT-LLM Worker 端

```python
# components/src/dynamo/trtllm/publisher.py:477-491
async def _publish_kv_cache_events_task(self):
    # 从 TensorRT-LLM engine 获取 KV events
    events = self.engine.llm.get_kv_cache_events_async(timeout=5)
    async for event in events:
        # 处理 event
        if data["type"] == "stored":
            # 发布到 ZMQ 或 NATS
            if self.zmq_kv_event_publisher:
                self.zmq_kv_event_publisher.publish_stored(...)
            elif self.kv_event_publisher:
                self.kv_event_publisher.publish_stored(...)
```

**工作流程：**
```
TensorRT-LLM Engine (GPU)
    ↓ (get_kv_cache_events_async)
Publisher._publish_kv_cache_events_task
    ↓ (转换为标准格式)
ZMQ 或 NATS
    ↓
Router Subscriber
```

### 3. Router 端接收

#### ZMQ Listener (如果使用 ZMQ)

```rust
// lib/llm/src/kv_router/publisher.rs:223-352
pub async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    tx: mpsc::UnboundedSender<KvCacheEvent>,
    ...
) {
    // 创建 ZMQ SUB socket
    let mut socket = SubSocket::new();
    socket.subscribe(&zmq_topic).await;
    socket.connect(&zmq_endpoint).await;
    
    // 接收消息
    loop {
        let msg = socket.recv().await;
        // 解析为 KvEventBatch
        // 转换为 KvCacheEvent
        // 发送到 channel
    }
}
```

#### NATS Subscriber (如果使用 NATS)

```rust
// lib/llm/src/kv_router/subscriber.rs:352-379
result = nats_queue.dequeue_task(None) => {
    match result {
        Ok(Some(bytes)) => {
            let event: RouterEvent = serde_json::from_slice(&bytes)?;
            // 转发到 indexer
            kv_events_tx.send(event).await?;
        }
    }
}
```

#### Indexer 更新 Radix Tree

```rust
// lib/llm/src/kv_router/indexer.rs:992-1024
Some(event) = event_rx.recv() => {
    // 更新 Radix Tree
    let result = trie.apply_event(event.clone());
    // ...
}
```

## 关键函数位置

### vLLM Worker 端

**vLLM 内部（vLLM 库）：**
- `vllm.distributed.kv_events.KVEventsConfig` - 配置 KV events
- vLLM 的 KV cache manager 内部自动发送 events
- **这是 vLLM 库的功能，不是 Dynamo 的代码**

**Dynamo Worker 端：**
- `components/src/dynamo/vllm/main.py:setup_kv_event_publisher()` - 设置 ZMQ publisher
- `dynamo.llm.kv_router.publisher.ZmqKvEventPublisher` - ZMQ publisher 实现

### TensorRT-LLM Worker 端

- `components/src/dynamo/trtllm/publisher.py:Publisher._publish_kv_cache_events_task()` - 获取并发布 events
- `self.engine.llm.get_kv_cache_events_async()` - 从 TensorRT-LLM engine 获取 events

### Router 端

- `lib/llm/src/kv_router/publisher.rs:start_zmq_listener()` - ZMQ listener
- `lib/llm/src/kv_router/subscriber.rs:start_kv_router_background()` - NATS subscriber
- `lib/llm/src/kv_router/indexer.rs:apply_event()` - 更新 Radix Tree

## 为什么是推送模式？

### 优势

1. **实时性**
   - Events 在 block 创建/删除时立即发送
   - Router 能实时跟踪 GPU 中的 KV cache 状态

2. **低延迟**
   - 不需要轮询查询
   - 减少网络开销

3. **可扩展性**
   - 多个 Router 可以订阅同一个 NATS stream
   - 支持多 Router 实例

### 如果使用拉取模式的问题

1. **延迟高**
   - 需要定期查询所有 workers
   - 查询频率和实时性的权衡

2. **开销大**
   - 每次 routing 决策都需要查询所有 workers
   - 增加网络负载

3. **一致性差**
   - 查询时刻的状态可能已经过时
   - 难以保证实时性

## 两种传输方式

### 方式1：直接 ZMQ (无 Consolidator)

```
vLLM Worker (ZMQ PUB)
    ↓
Router ZMQ Listener (ZMQ SUB)
    ↓
Router Indexer
    ↓
Radix Tree
```

### 方式2：ZMQ → Consolidator → NATS (有 Consolidator)

```
vLLM Worker (ZMQ PUB)
    ↓
KV Event Consolidator (ZMQ SUB → NATS PUB)
    ↓
NATS JetStream (持久化)
    ↓
Router Subscriber (NATS SUB)
    ↓
Router Indexer
    ↓
Radix Tree
```

**Consolidator 的作用：**
- 统一多个 workers 的 events
- 持久化到 NATS JetStream
- 支持 Router 重启后恢复状态

## 总结

### 信息获取方式

**不是发请求给各个 worker！**

- **Workers 主动推送** KV events（推送模式）
- **Router 被动接收** events
- **实时更新** Radix Tree

### 关键函数位置

1. **vLLM Worker:**
   - vLLM 库内部自动发送（`vllm.distributed.kv_events`）
   - Dynamo: `components/src/dynamo/vllm/main.py:setup_kv_event_publisher()`

2. **TensorRT-LLM Worker:**
   - `components/src/dynamo/trtllm/publisher.py:Publisher._publish_kv_cache_events_task()`
   - `self.engine.llm.get_kv_cache_events_async()` - 从 engine 获取

3. **Router:**
   - `lib/llm/src/kv_router/publisher.rs:start_zmq_listener()` - ZMQ listener
   - `lib/llm/src/kv_router/subscriber.rs:start_kv_router_background()` - NATS subscriber
   - `lib/llm/src/kv_router/indexer.rs:apply_event()` - 更新 Radix Tree

### 数据来源

- **GPU 内存中的 KV cache blocks**
- 当 block 被创建/删除时，worker 自动发送 event
- Router 通过 ZMQ 或 NATS 接收并更新 Radix Tree
- Router 使用 Radix Tree 进行 routing 决策






