# Indexer 对 Medium 字段的处理

## 核心发现

**Indexer 会接收并 apply 所有 medium 的 events，不会筛选只 apply GPU 的 events！**

## 代码证据

### 1. convert_event 函数忽略了 medium 字段

```rust
// lib/llm/src/kv_router/publisher.rs:356-415
fn convert_event(
    raw: RawKvEvent,
    event_id: u64,
    kv_block_size: u32,
    dp_rank: u32,
    warning_count: &Arc<AtomicU32>,
) -> KvCacheEvent {
    match raw {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_id,
            ..  // ← 关键：使用 .. 忽略了 medium 字段
        } => {
            // ... 转换为 KvCacheEvent，但没有 medium 信息
        }
        // ...
    }
}
```

**关键点：**
- `RawKvEvent` 结构中有 `medium: Option<String>` 字段
- 但在 `convert_event` 中使用了 `..` 来忽略 medium 字段
- 转换后的 `KvCacheEvent` 结构中没有 medium 字段

### 2. Indexer 的 apply_event 没有筛选

```rust
// lib/llm/src/kv_router/indexer.rs:992-1024
Some(event) = event_rx.recv() => {
    let event_type = KvIndexerMetrics::get_event_type(&event.event.data);
    let result = trie.apply_event(event.clone());  // ← 直接 apply，没有筛选
    // ...
}
```

**关键点：**
- Indexer 直接调用 `trie.apply_event(event.clone())`
- 没有对 medium 进行任何检查或筛选
- 所有 events（GPU、CPU、KVBM等）都会被 apply 到 Radix Tree

### 3. KvCacheEvent 结构中没有 medium 字段

```rust
// lib/llm/src/kv_router/indexer.rs (KvCacheEvent 定义)
pub struct KvCacheEvent {
    pub event_id: u64,
    pub data: KvCacheEventData,
    pub dp_rank: u32,
    // 没有 medium 字段！
}
```

## 完整信息流

### vLLM Worker 端

```
vLLM Worker
    ├─ GPU KV Cache → BlockStored { medium: "GPU" }
    ├─ CPU KV Cache → BlockStored { medium: "CPU_TIER1" }
    └─ KVBM Storage → BlockStored { medium: "KVBM" }
        ↓
    通过 ZMQ 发布所有 events（包含 medium 字段）
```

### Router 端处理

```
ZMQ Subscriber
    ↓ 接收 RawKvEvent（包含 medium）
convert_event()
    ↓ 忽略 medium 字段
KvCacheEvent（不包含 medium）
    ↓
Indexer.apply_event()
    ↓ 直接 apply，不筛选
Radix Tree（包含所有 medium 的 blocks）
```

## 潜在问题

### 1. Routing 决策可能不准确

**场景：**
- Worker A 的 GPU 中有 block X
- Worker B 的 KVBM（远程存储）中有 block X
- 两个 workers 的 overlap score 相同

**问题：**
- Router 无法区分 block 在 GPU 还是 KVBM
- 可能路由到 Worker B，但 block 在 KVBM 中，需要先加载到 GPU
- 这会导致额外的延迟

### 2. 性能影响

**如果 block 在 CPU 或 KVBM 中：**
- Routing 到该 worker 时，虽然能找到匹配的 blocks
- 但这些 blocks 不在 GPU 中，无法直接使用
- 需要先加载到 GPU，增加延迟

**如果 block 在 GPU 中：**
- 可以直接使用，延迟最低

## Consolidator 的作用

### Consolidator 会解析 medium，但不筛选

```rust
// lib/llm/src/block_manager/kv_consolidator/subscriber.rs:281-284
let storage_tier = medium
    .as_ref()
    .and_then(|m| StorageTier::from_vllm_medium(m))
    .unwrap_or(StorageTier::Device);
```

**关键点：**
- Consolidator 会解析 medium 字段并转换为 `StorageTier`
- 但 `StorageTier` 只用于**元数据/调试**，不用于筛选
- Consolidator 的作用是**去重**（deduplication），不是筛选

### Consolidator 发布的 events 没有 medium

```rust
// lib/llm/src/block_manager/kv_consolidator/publisher.rs:117
medium: None, // Not provided by ConsolidatedEvent
```

**关键点：**
- Consolidator 发布的 events 中 `medium: None`
- 这意味着即使 consolidator 知道 medium，也不会传递给 router

## 设计考虑

### 为什么可能这样设计？

1. **简化设计**
   - Router 只需要知道"哪些 blocks 在哪些 workers 上"
   - 不需要关心 block 的具体存储位置（GPU/CPU/KVBM）

2. **灵活性**
   - Worker 可以在运行时将 blocks 从 KVBM 加载到 GPU
   - Router 不需要知道这些细节

3. **性能优化在 Worker 端**
   - Worker 负责管理 KV cache 的加载/卸载
   - Router 只负责 routing 决策

### 可能的改进方向

1. **在 Router 中区分 medium**
   - 在 `KvCacheEvent` 中添加 `medium` 字段
   - 在 routing 决策时优先选择 GPU 中的 blocks

2. **在 Scheduler 中考虑 medium**
   - 在成本计算时，考虑 block 的存储位置
   - GPU blocks 的成本更低，CPU/KVBM blocks 的成本更高

3. **在 Indexer 中筛选**
   - 只 apply GPU 的 events（如果只需要 GPU 的 routing）
   - 或者维护多个 Radix Tree（按 medium 分类）

## 总结

1. **Indexer 会接收并 apply 所有 medium 的 events**
   - 不会筛选只 apply GPU 的 events
   - 所有 events（GPU、CPU、KVBM等）都会被 apply 到 Radix Tree

2. **medium 字段在 convert_event 时被忽略**
   - `RawKvEvent` 有 medium 字段
   - 但 `KvCacheEvent` 没有 medium 字段
   - Indexer 无法区分 block 的存储位置

3. **潜在影响**
   - Routing 决策可能不准确（无法区分 GPU 和 KVBM）
   - 可能路由到有 KVBM blocks 的 worker，需要额外加载时间

4. **设计考虑**
   - 可能是为了简化设计
   - 性能优化在 Worker 端进行
   - 如果需要区分 medium，需要在 Router 中添加相关逻辑





