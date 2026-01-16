# Dynamo vs SGLang Radix Tree 对比

## 核心区别

### SGLang 的 Radix Tree
- **每个 Worker 一个独立的 Radix Tree**
- 每个 Worker 只知道自己拥有的 KV cache blocks
- 用于管理该 Worker 自己的 KV cache（RadixAttention）

### Dynamo 的 Radix Tree
- **全局共享一个 Radix Tree**
- Router 维护一个 Radix Tree，记录**所有 Workers** 的 KV cache 分布情况
- 用于 Router 的 routing 决策（找到最佳匹配的 Worker）

## 数据结构对比

### SGLang (每个 Worker 一个 Tree)

```
Worker 1:
  RadixTree_1
    ├─ Block A
    ├─ Block B
    └─ Block C

Worker 2:
  RadixTree_2
    ├─ Block D
    ├─ Block E
    └─ Block F

Worker 3:
  RadixTree_3
    ├─ Block G
    └─ Block H
```

**特点：**
- 每个 Worker 独立管理自己的 KV cache
- Trees 之间不共享信息
- 用于实际的 KV cache 存储和管理

### Dynamo (全局共享一个 Tree)

```
Router 的 RadixTree (全局)
  ├─ Block A
  │   └─ workers: [Worker1, Worker2]  ← 多个workers可以共享同一个block
  ├─ Block B
  │   └─ workers: [Worker1]
  ├─ Block C
  │   └─ workers: [Worker1, Worker3]
  └─ Block D
      └─ workers: [Worker2]
```

**特点：**
- 一个全局 Tree 记录所有 Workers 的 KV cache 分布
- 同一个 Block 可以属于多个 Workers（通过 `workers` HashMap）
- 用于 Router 的 routing 决策

## 代码证据

### Dynamo RadixBlock 结构

```rust
// lib/llm/src/kv_router/indexer.rs:202-212
struct RadixBlock {
    /// 子节点映射
    children: HashMap<LocalBlockHash, SharedRadixBlock>,
    
    /// 关键！每个block可以属于多个workers
    /// WorkerWithDpRank -> ExternalSequenceBlockHash
    workers: HashMap<WorkerWithDpRank, ExternalSequenceBlockHash>,
    
    /// 访问频率跟踪
    recent_uses: VecDeque<Instant>,
}
```

**关键点：** `workers: HashMap<WorkerWithDpRank, ExternalSequenceBlockHash>`
- 一个 Block 可以属于多个 Workers
- 当多个 Workers 有相同的 KV cache block 时，它们共享同一个 RadixBlock 节点

### Dynamo RadixTree 结构

```rust
// lib/llm/src/kv_router/indexer.rs:229-245
pub struct RadixTree {
    /// 全局唯一的root节点
    root: SharedRadixBlock,
    
    /// 按worker索引的查找表
    /// WorkerWithDpRank -> (ExternalSequenceBlockHash -> SharedRadixBlock)
    lookup: HashMap<WorkerWithDpRank, HashMap<ExternalSequenceBlockHash, SharedRadixBlock>>,
    
    expiration_duration: Option<Duration>,
}
```

**关键点：**
- 只有一个 `root` 节点（全局共享）
- `lookup` 表按 worker 索引，但指向的是**共享的** RadixBlock 节点

### find_matches 逻辑

```rust
// lib/llm/src/kv_router/indexer.rs:296-297
if let Some(block) = next_block {
    // 获取该block的所有workers
    scores.update_scores(block.borrow().workers.keys());
```

**关键点：**
- 当找到匹配的 block 时，获取该 block 的**所有 workers**
- 这意味着一个 block 可以属于多个 workers

### apply_event 逻辑

```rust
// lib/llm/src/kv_router/indexer.rs:396-412
let child = match parent_mut.children.get(&block_data.tokens_hash) {
    Some(block) => block.clone(),  // Block已存在，复用
    None => {
        // 创建新block
        let new_block = ...
        parent_mut.children.insert(block_data.tokens_hash, new_block.clone());
        new_block
    }
};

// 将该worker添加到block的workers map中
child_mut.workers.insert(worker, block_data.block_hash);
```

**关键点：**
- 如果 block 已存在（其他 worker 也有），则**复用**同一个 block 节点
- 只是将新的 worker 添加到 `workers` HashMap 中
- 这证明了多个 workers 可以共享同一个 block 节点

## 实际场景示例

### 场景：两个 Workers 有相同的 Prefix

假设两个请求有相同的 prefix tokens：

**Worker 1 处理请求 A:**
```
Request A: "Hello world, how are you?"
Blocks: [Block1("Hello world"), Block2(", how"), Block3(" are you?")]
```

**Worker 2 处理请求 B:**
```
Request B: "Hello world, I am fine."
Blocks: [Block1("Hello world"), Block2(", I"), Block3(" am fine.")]
```

### SGLang 的情况

```
Worker 1 的 RadixTree:
  Root
    └─ Block1("Hello world")
        └─ Block2(", how")
            └─ Block3(" are you?")

Worker 2 的 RadixTree:
  Root
    └─ Block1("Hello world")  ← 独立的节点
        └─ Block2(", I")
            └─ Block3(" am fine.")
```

**特点：**
- 每个 Worker 有独立的 Tree
- 即使有相同的 Block1，也是独立的节点
- Workers 之间不知道对方有什么

### Dynamo 的情况

```
Router 的全局 RadixTree:
  Root
    └─ Block1("Hello world")
        ├─ workers: [Worker1, Worker2]  ← 共享同一个节点！
        ├─ Block2(", how")
        │   └─ workers: [Worker1]
        └─ Block2(", I")
            └─ workers: [Worker2]
```

**特点：**
- 全局共享一个 Tree
- Block1 被两个 Workers 共享（同一个节点）
- Router 知道哪些 Workers 有相同的 prefix

### Routing 决策

当新请求到来：`"Hello world, tell me"`

**Dynamo Router 查找：**
```rust
find_matches(["Hello world", " tell me"])
  → 找到 Block1("Hello world")
  → 发现 workers: [Worker1, Worker2]  ← 两个workers都有这个block
  → 计算overlap scores
  → 选择最佳worker（考虑overlap和负载）
```

**优势：**
- Router 知道哪些 Workers 有匹配的 prefix
- 可以选择 overlap 最多的 Worker
- 最大化 KV cache 命中率

## 为什么这样设计？

### 1. Router 需要全局视图

Router 需要回答：
- "哪些 Workers 有与当前请求匹配的 KV cache？"
- "哪个 Worker 的匹配度最高？"

这需要**全局视图**，而不是每个 Worker 的局部视图。

### 2. 支持 Prefix 共享

多个 Workers 可能有相同的 prefix（例如系统 prompt、常见前缀等）：
- 在全局 Tree 中，这些 Workers 共享同一个 Block 节点
- Router 可以识别并利用这种共享

### 3. 高效的查找

- 一次查找就能找到所有匹配的 Workers
- 不需要查询每个 Worker 的独立 Tree

## 总结

| 特性 | SGLang | Dynamo |
|------|--------|--------|
| **Tree 数量** | 每个 Worker 一个 | 全局共享一个 |
| **作用** | 管理 Worker 的 KV cache | Router 的 routing 索引 |
| **Block 共享** | 不支持（每个 Worker 独立） | 支持（多个 Workers 共享节点） |
| **全局视图** | 无 | 有 |
| **查找范围** | 单个 Worker | 所有 Workers |

**关键理解：**
- **SGLang**: 每个 Worker 维护自己的 Radix Tree，用于管理自己的 KV cache
- **Dynamo**: Router 维护一个全局 Radix Tree，用于跟踪所有 Workers 的 KV cache 分布，支持智能 routing

两者**不冲突**，服务于不同的目的！






