# Dynamo Radix Tree 架构说明

## 关键理解

**Radix Tree 是在 Dynamo Router 层面维护的，不是底层 Backend（vLLM/SGLang）维护的！**

## 架构层次

```
┌─────────────────────────────────────────────────────────┐
│                    Dynamo Router                        │
│  ┌──────────────────────────────────────────────────┐  │
│  │         KvIndexer (维护 Radix Tree)               │  │
│  │  - 接收 KV Events                                 │  │
│  │  - 更新 Radix Tree                                │  │
│  │  - 提供 find_matches() 查找                       │  │
│  └──────────────────────────────────────────────────┘  │
│                          ↑                               │
│                          │ 接收 KV Events                │
└──────────────────────────┼──────────────────────────────┘
                            │
                            │ (ZMQ 或 NATS)
                            │
┌───────────────────────────┼──────────────────────────────┐
│                    Backend Workers                       │
│  ┌────────────────────┐  │  ┌────────────────────┐     │
│  │   vLLM Worker      │  │  │   SGLang Worker    │     │
│  │                    │  │  │                    │     │
│  │  - 发送 KV Events  │  │  │  - 发送 KV Events  │     │
│  │  (通过 ZMQ)        │  │  │  (通过 ZMQ/NATS)   │     │
│  └────────────────────┘  │  └────────────────────┘     │
│                          │                               │
│  ┌──────────────────────────────────────────────────┐  │
│  │  每个 Backend 维护自己的 KV Cache                │  │
│  │  (vLLM: PagedAttention, SGLang: RadixAttention)  │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## 为什么需要 Radix Tree？

### 1. Router 需要知道每个 Worker 的 KV Cache 状态

Router 需要回答这个问题：
- **"哪些 worker 有与当前请求匹配的 KV cache blocks？"**

为了回答这个问题，Router 需要：
- 跟踪每个 worker 的 KV cache 内容
- 快速查找匹配的 blocks
- 计算 overlap scores

### 2. Radix Tree 是 Router 的索引结构

Radix Tree 在 Router 层面作为**索引/元数据**存储：
- **不是**实际存储 KV cache 数据（数据在 worker 的 GPU 内存中）
- **而是**存储 KV cache 的**元数据**（哪些 blocks 在哪些 workers 上）

## 工作流程

### vLLM Backend 发送 KV Events

```python
# vLLM 配置 (components/src/dynamo/vllm/args.py)
kv_events_config = KVEventsConfig(
    enable_kv_cache_events=True,
    publisher="zmq",  # 使用 ZMQ 发布 events
    endpoint=f"tcp://*:{kv_port}",
)
```

**vLLM 发送的 Events：**
- `Stored`: 当新的 KV block 被创建时
- `Removed`: 当 KV block 被删除时
- `Cleared`: 当所有 blocks 被清除时

### Dynamo Router 接收 Events

```rust
// lib/llm/src/kv_router/publisher.rs
// ZMQ Listener 接收来自 vLLM 的 events
start_zmq_listener(
    endpoint,
    topic,
    tx,
    cancellation_token,
    kv_block_size,
)

// 然后通过 NATS 发送到 Router 的 subscriber
// lib/llm/src/kv_router/subscriber.rs
// Router 从 NATS 接收 events 并更新 Radix Tree
```

### Router 更新 Radix Tree

```rust
// lib/llm/src/kv_router/indexer.rs
// 当收到 KV event 时，更新 Radix Tree
pub fn apply_event(&mut self, event: RouterEvent) -> Result<(), KvCacheEventError> {
    match event.event.data {
        KvCacheEventData::Stored(op) => {
            // 在 Radix Tree 中添加新的 block
            // 记录该 block 属于哪个 worker
        }
        KvCacheEventData::Removed(remove) => {
            // 从 Radix Tree 中移除 block
        }
        KvCacheEventData::Cleared => {
            // 清除该 worker 的所有 blocks
        }
    }
}
```

## 为什么 vLLM 也能工作？

### vLLM 的 KV Cache 管理

vLLM 使用 **PagedAttention** 管理 KV cache：
- 将 KV cache 分成固定大小的 pages
- 每个 page 对应一个 block
- 当 block 被创建/删除时，vLLM 发送 KV events

### vLLM 发送 Events 的机制

1. **ZMQ Publisher** (`lib/llm/src/kv_router/publisher.rs`)
   - vLLM 通过 ZMQ socket 发送 KV events
   - Dynamo Router 的 ZMQ listener 接收这些 events

2. **Event 格式**
   ```rust
   pub struct KvCacheEvent {
       pub event_id: u64,
       pub data: KvCacheEventData,
       pub dp_rank: u32,
   }
   
   pub enum KvCacheEventData {
       Stored(KvCacheStoreData),  // block 被创建
       Removed(KvCacheRemoveData), // block 被删除
       Cleared,                   // 所有 blocks 被清除
   }
   ```

3. **Router 接收并处理**
   - Router 的 subscriber 从 NATS 接收 events
   - 更新自己的 Radix Tree
   - 用于后续的 routing 决策

## SGLang 的情况

SGLang 使用 **RadixAttention**：
- 内部也使用 Radix Tree 结构管理 KV cache
- **但是**，Dynamo Router 的 Radix Tree 是**独立的**
- SGLang 只需要发送 KV events，Router 就能维护自己的索引

## 两种模式对比

### Exact Mode (use_kv_events = true)

```
Backend (vLLM/SGLang)
    ↓ 发送 KV Events (ZMQ/NATS)
Router Subscriber
    ↓ 接收 Events
KvIndexer
    ↓ 更新 Radix Tree
Radix Tree (Router 的索引)
```

**特点：**
- 基于真实的 KV events
- 需要 NATS 持久化 events
- 实时反映 worker 的 KV cache 状态

### Approximate Mode (use_kv_events = false)

```
Router 记录自己的 routing 决策
    ↓
KvIndexer.process_routing_decision()
    ↓ 更新 Radix Tree (预测状态)
Radix Tree (Router 的预测索引)
```

**特点：**
- 不需要真实的 KV events
- 基于 Router 的 routing 决策预测
- 使用 TTL 和 pruning 机制
- 不需要 NATS

## 关键点总结

1. **Radix Tree 是 Router 的索引，不是 Backend 的**
   - Router 维护自己的 Radix Tree 来跟踪 worker 的 KV cache 状态
   - Backend 只需要发送 KV events

2. **无论底层是 vLLM 还是 SGLang**
   - 只要 Backend 能够发送 KV events（通过 ZMQ 或 NATS）
   - Router 就能维护自己的 Radix Tree
   - 做出智能的 routing 决策

3. **Backend 的 KV Cache 实现不影响 Router**
   - vLLM 使用 PagedAttention
   - SGLang 使用 RadixAttention
   - 但 Router 的 Radix Tree 是独立的索引结构

4. **Radix Tree 存储的是元数据，不是实际数据**
   - 实际 KV cache 数据在 worker 的 GPU 内存中
   - Radix Tree 只存储"哪些 blocks 在哪些 workers 上"的映射关系

## 代码位置

- **Radix Tree 实现**: `lib/llm/src/kv_router/indexer.rs`
- **Event Publisher (ZMQ)**: `lib/llm/src/kv_router/publisher.rs`
- **Event Subscriber (NATS)**: `lib/llm/src/kv_router/subscriber.rs`
- **vLLM 配置**: `components/src/dynamo/vllm/args.py`






