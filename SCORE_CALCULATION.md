# Overlap Score 计算详解

## Score 计算逻辑

### 1. 基本计算方式

**Score = 连续匹配的 Block 数量（即前缀长度）**

```rust
// lib/llm/src/kv_router/indexer.rs:281-326
pub fn find_matches(&self, sequence: Vec<LocalBlockHash>, early_exit: bool) -> OverlapScores {
    let mut scores = OverlapScores::new();
    let mut current = self.root.clone();
    
    // 遍历请求的每个block
    for (idx, block_hash) in sequence.iter().enumerate() {
        if let Some(block) = next_block {
            // 关键：每个匹配的block，将该block的所有workers的score +1
            scores.update_scores(block.borrow().workers.keys());
            current = block;
        } else {
            // 如果找不到匹配，停止遍历（前缀匹配结束）
            break;
        }
    }
    
    scores
}
```

### 2. update_scores 实现

```rust
// lib/llm/src/kv_router/indexer.rs:741-749
pub fn update_scores<'a, I>(&mut self, workers: I)
where
    I: IntoIterator<Item = &'a WorkerWithDpRank>,
{
    for worker in workers {
        let score = self.scores.entry(*worker).or_insert(0);
        *score += 1;  // 每个匹配的block，该worker的score +1
    }
}
```

### 3. 计算示例

**场景：**
- 请求 tokens: `["Hello", "world", "how", "are", "you"]`
- Block size: 2 tokens
- 请求 blocks: `[Block1("Hello world"), Block2("how are"), Block3("you")]`

**Worker A 的 KV Cache:**
- Block1("Hello world") ✓
- Block2("how are") ✓
- Block3("you") ✗

**查找过程：**

```
1. Block1("Hello world")
   → 在Tree中找到匹配
   → Worker A的score +1
   → score = 1

2. Block2("how are")
   → 在Tree中找到匹配
   → Worker A的score +1
   → score = 2

3. Block3("you")
   → 在Tree中找不到匹配
   → 停止遍历
   → 最终score = 2
```

**结果：**
- Worker A: overlap_blocks = 2
- 这意味着 Worker A 有 2 个连续的匹配 blocks（前缀长度 = 2 blocks）

### 4. 前缀长度 vs Score

**是的，命中的前缀越长，score 越高！**

- Score = 连续匹配的 Block 数量
- 前缀越长 = 匹配的 Block 越多 = Score 越高
- 这是**前缀匹配**（prefix matching），不是部分匹配

**重要：**
- 必须是**连续的前缀匹配**
- 如果中间某个 block 不匹配，遍历会停止
- 不会计算非连续或非前缀的匹配

## 命中数量来源

### 1. KV Events 的来源

**KV Events 来自 Worker 的 GPU 内存中的 KV Cache**

```rust
// lib/llm/src/kv_router/publisher.rs:223-352
// ZMQ Listener 接收来自 vLLM/TensorRT-LLM 的 KV events
pub async fn start_zmq_listener(
    zmq_endpoint: String,
    zmq_topic: String,
    tx: mpsc::UnboundedSender<KvCacheEvent>,
    ...
) {
    // 从 ZMQ socket 接收 events
    // 这些 events 来自 worker 的 GPU 内存
}
```

### 2. Event Source 类型

```rust
// lib/llm/src/block_manager/kv_consolidator/tracker.rs:61-94
pub enum EventSource {
    /// Events from vLLM worker (G1/GPU)
    Vllm,
    /// Events from TensorRT-LLM worker (G1/GPU)
    Trtllm,
    /// Events from KVBM
    Kvbm,
}
```

**关键理解：**
- **Vllm/Trtllm**: Events 来自 GPU 内存中的 KV cache
- **Kvbm**: Events 来自 KVBM（用于 disaggregated serving），但最终也反映 GPU 中的 KV cache 状态

### 3. 不是存储 Backend 的总命中数量

**Score 反映的是 Worker GPU 中的 KV Cache 状态，不是存储 Backend 的总命中数量。**

**原因：**
1. **KV Events 的发送时机**
   - 当 Worker 的 GPU 内存中创建/删除 KV cache block 时发送
   - 反映的是**当前 GPU 内存中的 KV cache 状态**

2. **Radix Tree 的更新**
   - Router 的 Radix Tree 根据 KV Events 更新
   - 只记录**GPU 内存中存在的 blocks**
   - 不记录已从 GPU 内存中移除的 blocks

3. **查找逻辑**
   - `find_matches` 在 Radix Tree 中查找
   - 只找到**当前 GPU 内存中存在的 blocks**
   - 不查找存储 backend（如 KVBM）中的 blocks

### 4. 存储 Backend 的情况

**如果使用 KVBM（Key-Value Block Manager）进行 disaggregated serving：**

```
┌─────────────────────────────────────────┐
│         Worker GPU Memory                │
│  ┌──────────────────────────────────┐   │
│  │  Active KV Cache Blocks          │   │
│  │  (发送 KV Events)                │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│         KVBM Storage Backend            │
│  ┌──────────────────────────────────┐   │
│  │  Persistent KV Cache Blocks      │   │
│  │  (不发送 KV Events)              │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**关键点：**
- **GPU 内存中的 blocks**: 发送 KV Events → Router 跟踪 → 用于 routing
- **KVBM 存储中的 blocks**: 不发送 KV Events → Router 不跟踪 → 不用于 routing

**为什么？**
- Router 需要知道**当前可用的** KV cache（在 GPU 内存中）
- 存储 backend 中的 blocks 需要先加载到 GPU 才能使用
- 加载过程有延迟，不适合用于实时 routing 决策

## Score 在 Routing 中的使用

### 1. 转换为成本计算

```rust
// lib/llm/src/kv_router/scheduler.rs:505-526
let overlap = *overlaps.get(&worker).unwrap_or(&0);  // overlap_blocks

// 计算prefill tokens（考虑overlap）
let prefill_token = *prefill_tokens.get(&worker).unwrap_or(&isl);
let potential_prefill_block = (prefill_token as f64) / (block_size as f64);

// 计算decode blocks
let decode_block = *decode_blocks.get(&worker).unwrap_or(&potential_prefill_block) as f64;

// 计算logit（成本，越低越好）
let overlap_weight = router_config_override.overlap_score_weight 
                     .unwrap_or(kv_router_config.overlap_score_weight);
let logit = overlap_weight * potential_prefill_block + decode_block;
```

**公式：**
```
logit = overlap_weight * prefill_blocks + decode_blocks

其中：
- prefill_blocks = (isl_tokens - overlap_blocks * block_size) / block_size
- overlap_blocks = score（连续匹配的block数量）
```

### 2. Overlap 的影响

**Overlap 越多（score 越高），prefill_blocks 越少，logit 越低（成本越低）**

**示例：**
- 请求: 100 tokens, block_size = 16
- Worker A: overlap_blocks = 3 (48 tokens命中)
  - prefill_blocks = (100 - 48) / 16 = 3.25
  - logit = overlap_weight * 3.25 + decode_blocks
- Worker B: overlap_blocks = 0 (0 tokens命中)
  - prefill_blocks = 100 / 16 = 6.25
  - logit = overlap_weight * 6.25 + decode_blocks

**Worker A 的 logit 更低（成本更低），会被优先选择！**

## 总结

### Score 计算
- **Score = 连续匹配的 Block 数量**
- **前缀越长，score 越高**
- **必须是连续的前缀匹配**

### 命中数量来源
- **来自 Worker GPU 内存中的 KV Cache**
- **不是存储 Backend 的总命中数量**
- **只反映当前 GPU 内存中可用的 KV cache**

### 为什么只看 GPU 内存？
- GPU 内存中的 KV cache 可以**立即使用**
- 存储 backend 中的 blocks 需要**先加载**，有延迟
- Router 需要做**实时决策**，只能考虑当前可用的资源






