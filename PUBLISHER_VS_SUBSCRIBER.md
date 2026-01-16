# Publisher vs Subscriber 架构说明

## 核心理解

**`publisher.rs` 和 `subscriber.rs` 运行在不同的组件中，实现了一个两层架构！**

## 架构层次

### 完整信息流

```
┌─────────────────────────────────────────────────────────────────┐
│                    vLLM Worker                                   │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  vLLM Engine (GPU)                                        │ │
│  │    ↓ (内部自动发送)                                        │ │
│  │  ZMQ PUB Socket                                            │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │ ZMQ 消息
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              Dynamo Worker (publisher.rs)                        │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  KvEventPublisher                                         │ │
│  │    ├─ start_zmq_listener()  ← ZMQ SUB (订阅 vLLM)        │ │
│  │    │   └─ convert_event()                                 │ │
│  │    └─ start_event_processor()                             │ │
│  │        └─ 发布到 NATS JetStream                           │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             │ NATS JetStream (持久化)
                             ↓
┌─────────────────────────────────────────────────────────────────┐
│              Dynamo Router (subscriber.rs)                      │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │  start_kv_router_background()                            │ │
│  │    ├─ NatsQueue (NATS SUB)                               │ │
│  │    ├─ 从 NATS JetStream 订阅 events                      │ │
│  │    └─ 发送给 Indexer                                     │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────┬────────────────────────────────────┘
                             ↓
                    Indexer → Radix Tree
```

## Publisher.rs (Worker 端)

### 作用

**`publisher.rs` 在 Worker 端运行，作为 ZMQ 和 NATS 之间的桥接器。**

### 主要功能

1. **从 ZMQ 订阅 vLLM 的 KV events**
   ```rust
   // lib/llm/src/kv_router/publisher.rs:223-352
   pub async fn start_zmq_listener(
       zmq_endpoint: String,
       zmq_topic: String,
       tx: mpsc::UnboundedSender<KvCacheEvent>,
       ...
   ) {
       let mut socket = SubSocket::new();  // ZMQ SUB
       socket.subscribe(&zmq_topic).await;
       socket.connect(&zmq_endpoint).await;
       
       // 接收 ZMQ 消息并转换为 KvCacheEvent
       while let Ok(msg) = socket.recv().await {
           let event = convert_event(raw_event, ...);
           tx.send(event).await;
       }
   }
   ```

2. **将 events 发布到 NATS JetStream**
   ```rust
   // lib/llm/src/kv_router/publisher.rs:180-207
   async fn start_event_processor<P: EventPublisher>(
       publisher: P,
       worker_id: u64,
       mut rx: mpsc::UnboundedReceiver<KvCacheEvent>,
   ) {
       while let Some(event) = rx.recv().await {
           // 封装为 RouterEvent（包含 worker_id）
           let router_event = RouterEvent::new(worker_id, event);
           // 发布到 NATS
           publisher.publish(QUEUE_NAME, &router_event).await;
       }
   }
   ```

### 关键点

- **运行位置**: Dynamo Worker 端
- **输入**: ZMQ 消息（来自 vLLM）
- **输出**: NATS JetStream（持久化）
- **作用**: 桥接 ZMQ 和 NATS，添加 worker_id

## Subscriber.rs (Router 端)

### 作用

**`subscriber.rs` 在 Router 端运行，从 NATS JetStream 订阅 events 并发送给 Indexer。**

### 主要功能

1. **从 NATS JetStream 订阅 events**
   ```rust
   // lib/llm/src/kv_router/subscriber.rs:215-446
   pub async fn start_kv_router_background(
       component: Component,
       consumer_id: String,
       kv_events_tx: mpsc::Sender<RouterEvent>,
       ...
   ) {
       // 创建 NatsQueue (NATS SUB)
       let mut nats_queue = NatsQueue::new_with_consumer(
           stream_name,
           nats_server,
           ...
       );
       
       // 从 NATS 订阅 events
       while let Ok(Some(bytes)) = nats_queue.dequeue_task(None).await {
           let event: RouterEvent = serde_json::from_slice(&bytes)?;
           // 发送给 Indexer
           kv_events_tx.send(event).await?;
       }
   }
   ```

2. **处理 Snapshot 和 State Management**
   - 下载初始状态（从 NATS object store）
   - 定期上传 snapshot
   - 清理过期的 messages

### 关键点

- **运行位置**: Dynamo Router 端
- **输入**: NATS JetStream
- **输出**: Indexer (Radix Tree)
- **作用**: 从 NATS 订阅并转发给 Indexer

## 为什么需要两层架构？

### 1. **解耦和可靠性**

- **ZMQ**: 快速、低延迟，但不持久化
- **NATS JetStream**: 持久化、可靠，支持多订阅者

### 2. **持久化**

- Router 重启后可以从 NATS JetStream 恢复状态
- 支持 snapshot 机制

### 3. **多 Router 支持**

- 多个 Router 可以订阅同一个 NATS stream
- 每个 Router 有自己的 consumer，互不干扰

### 4. **Worker ID 注入**

- Publisher 在 Worker 端知道自己的 worker_id
- 将 worker_id 封装到 RouterEvent 中
- Router 可以知道 event 来自哪个 worker

## 代码位置总结

### Publisher.rs (Worker 端)

- **ZMQ Listener**: `start_zmq_listener()` - 从 ZMQ 订阅
- **Event Processor**: `start_event_processor()` - 发布到 NATS
- **使用位置**: Worker 启动时创建 `KvEventPublisher`

### Subscriber.rs (Router 端)

- **NATS Subscriber**: `start_kv_router_background()` - 从 NATS 订阅
- **使用位置**: Router 启动时调用 `start_kv_router_background()`

## 两种模式对比

### 模式 1: 直接模式（无 Consolidator）

```
vLLM Worker
    ↓ ZMQ PUB
Dynamo Worker (publisher.rs)
    ↓ ZMQ SUB → NATS PUB
NATS JetStream
    ↓ NATS SUB
Dynamo Router (subscriber.rs)
    ↓
Indexer
```

### 模式 2: Consolidator 模式

```
vLLM Worker
    ↓ ZMQ PUB
Consolidator (去重、合并)
    ↓ ZMQ PUB
Dynamo Worker (publisher.rs)
    ↓ ZMQ SUB → NATS PUB
NATS JetStream
    ↓ NATS SUB
Dynamo Router (subscriber.rs)
    ↓
Indexer
```

## 总结

1. **`publisher.rs`**: Worker 端，ZMQ SUB → NATS PUB
2. **`subscriber.rs`**: Router 端，NATS SUB → Indexer
3. **两层架构**: 使用 NATS 作为中间层，实现持久化和可靠性
4. **解耦**: Worker 和 Router 通过 NATS 解耦，互不直接依赖





