# Dynamo Routing 逻辑和调用链分析

## 概述

Dynamo的routing系统是一个基于KV cache感知的智能路由系统，它能够根据worker的KV cache状态来智能选择最佳worker处理请求。

## 主要组件

### 1. **KvRouter** (`lib/llm/src/kv_router.rs`)
   - 核心路由决策组件
   - 负责找到最佳匹配的worker
   - 主要方法：`find_best_match()`

### 2. **KvIndexer** (`lib/llm/src/kv_router/indexer.rs`)
   - 维护Radix Tree索引
   - 跟踪每个worker的KV cache状态
   - 计算overlap scores（重叠分数）

### 3. **KvScheduler** (`lib/llm/src/kv_router/scheduler.rs`)
   - 调度器，负责负载均衡
   - 综合考虑overlap scores和worker负载
   - 选择最终worker

### 4. **PrefillRouter** (`lib/llm/src/kv_router/prefill_router.rs`)
   - 处理disaggregated serving场景
   - 可选组件，用于prefill和decode分离

### 5. **KvPushRouter** (`lib/llm/src/kv_router.rs`)
   - 包装PushRouter，集成KV routing逻辑
   - 实际执行路由转发

## 完整调用链

### 简化调用链图

```
HTTP Request
    ↓
[HTTP Endpoint] handle_shared_request()
    ↓
[HTTP Service] handle_payload()
    ↓
[Pipeline Engine] generate()
    ↓
[Frontend] SegmentSource
    ↓
[Preprocessor] OpenAIPreprocessor
    ├─ Tokenize请求
    └─ 转换为PreprocessedRequest
    ↓
[Backend] Backend
    ↓
[Migration] Migration
    ↓
[PrefillRouter] PrefillRouter (可选)
    ├─ call_prefill() → KvPushRouter → find_best_match() → [Prefill Worker]
    └─ 提取disaggregated_params
    ↓
[Router] KvPushRouter (KV模式) 或 PushRouter (其他模式)
    ├─ KvPushRouter.generate()
    │   ├─ find_best_match()
    │   │   ├─ compute_block_hash_for_seq()  # 计算block hashes
    │   │   ├─ indexer.find_matches()  # 在Radix Tree中查找overlap
    │   │   ├─ scheduler.schedule()  # 计算成本并选择worker
    │   │   │   ├─ potential_blocks_and_tokens()  # 计算潜在负载
    │   │   │   ├─ select_worker()  # 计算logit并选择
    │   │   │   └─ softmax_sample()  # 采样选择
    │   │   └─ process_routing_decision()  # 更新indexer（approximate模式）
    │   ├─ add_request()  # 添加到scheduler跟踪
    │   └─ inner.direct()  # 转发到worker
    └─ [Selected Worker] 处理请求
    ↓
[Response Stream] 返回结果
    ├─ mark_prefill_completed()  # 标记prefill完成
    └─ free()  # 请求完成时清理
```

### 1. HTTP请求入口

```
HTTP Request
    ↓
lib/runtime/src/pipeline/network/ingress/http_endpoint.rs
    ├─ handle_shared_request()  # HTTP handler
    └─ service_handler.handle_payload()  # 转发到service handler
```

### 2. HTTP Service处理

```
lib/llm/src/http/service/service_v2.rs
    ├─ HttpService.handle_payload()
    └─ engine.generate()  # 调用pipeline engine
```

### 3. Pipeline构建（初始化时）

```
lib/llm/src/entrypoint/input/common.rs
    └─ build_routed_pipeline_with_preprocessor()
        ├─ 创建pipeline组件链：
        │   ├─ frontend (SegmentSource)
        │   ├─ preprocessor_op (OpenAIPreprocessor)
        │   ├─ backend (Backend)
        │   ├─ migration (Migration)
        │   ├─ prefill_op (PrefillRouter) [可选]
        │   └─ service_backend (KvPushRouter 或 PushRouter)
        └─ 链接所有组件形成pipeline
```

### 4. 请求处理流程（运行时）

```
Request Flow (Forward):
    ↓
1. Frontend (SegmentSource)
    ↓
2. Preprocessor (OpenAIPreprocessor)
    ├─ 将HTTP请求转换为PreprocessedRequest
    └─ tokenize请求
    ↓
3. Backend (Backend)
    ├─ 处理token相关逻辑
    └─ 准备PreprocessedRequest
    ↓
4. Migration (Migration)
    ├─ 处理模型迁移相关逻辑
    └─ 传递PreprocessedRequest
    ↓
5. PrefillRouter (PrefillRouter) [可选，disaggregated模式]
    ├─ call_prefill()  # 调用prefill worker
    │   └─ 如果使用KV routing:
    │       └─ KvPushRouter.generate()
    │           └─ find_best_match()  # 找到prefill worker
    ├─ 提取disaggregated_params
    └─ 注入到decode请求
    ↓
6. KvPushRouter (KvPushRouter) [KV routing模式]
    └─ generate()
        ├─ 检查query_instance_id annotation
        ├─ 检查backend_instance_id
        └─ find_best_match()  # 核心路由逻辑
            ↓
            KvRouter.find_best_match()
                ├─ compute_block_hash_for_seq()  # 计算block hashes
                ├─ compute_seq_hash_for_block()  # 计算sequence hashes
                ├─ indexer.find_matches()  # 查找匹配
                │   └─ KvIndexer.find_matches()
                │       └─ 在Radix Tree中查找overlap
                ├─ scheduler.schedule()  # 调度选择worker
                │   └─ KvScheduler.schedule()
                │       ├─ 计算每个worker的成本
                │       ├─ 考虑overlap scores
                │       ├─ 考虑worker负载
                │       └─ 选择最佳worker
                └─ process_routing_decision()  # 更新indexer状态（如果使用approximate模式）
                    └─ KvIndexer.process_routing_decision()
        ├─ add_request()  # 添加到scheduler跟踪
        ├─ inner.direct()  # 转发到选定的worker
        └─ 包装响应流，处理prefill完成和free事件
    ↓
7. PushRouter (PushRouter) [非KV模式：Random/RoundRobin/Direct]
    └─ 根据模式选择worker
    ↓
8. Worker处理请求
    └─ 返回LLMEngineOutput
    ↓
Response Flow (Backward):
    ↓
9. KvPushRouter包装响应
    ├─ mark_prefill_completed()  # 标记prefill完成
    └─ free()  # 请求完成时清理
    ↓
10. PrefillRouter (backward)
    ↓
11. Migration (backward)
    ↓
12. Backend (backward)
    ↓
13. Preprocessor (backward)
    ↓
14. Frontend
    └─ 返回HTTP响应
```

## 核心路由逻辑详解

### `KvRouter.find_best_match()` 详细流程

```rust
pub async fn find_best_match(
    &self,
    context_id: Option<&str>,
    tokens: &[u32],
    router_config_override: Option<&RouterConfigOverride>,
    update_states: bool,
) -> Result<(WorkerWithDpRank, u32)>
```

**步骤：**

1. **计算Hashes**
   ```rust
   let block_hashes = compute_block_hash_for_seq(tokens, self.block_size);
   let seq_hashes = compute_seq_hash_for_block(&block_hashes);
   ```
   - 将tokens按block_size分块
   - 计算每个block的hash
   - 计算sequence hashes（用于跟踪序列）

2. **查找匹配**
   ```rust
   let overlap_scores = self.indexer.find_matches(block_hashes.clone()).await?;
   ```
   - 在Radix Tree中查找每个worker的KV cache匹配
   - 返回每个worker的overlap blocks数量

3. **调度选择**
   ```rust
   let best_worker = self.scheduler.schedule(
       context_id,
       isl_tokens,
       maybe_seq_hashes_2,
       overlap_scores,
       router_config_override,
       update_states,
   ).await?;
   ```
   - 计算每个worker的成本：
     - Prefill成本 = (isl_tokens - overlap_blocks * block_size) * prefill_cost_per_token
     - Decode成本 = decode_blocks * decode_cost_per_block
     - 总成本 = prefill_cost + decode_cost - overlap_benefit
   - 使用softmax采样（根据router_temperature）
   - 选择成本最低的worker

4. **更新状态**（如果使用approximate模式）
   ```rust
   if needs_process_routing {
       self.indexer.process_routing_decision(
           best_worker, 
           block_hashes, 
           seq_hashes
       ).await?;
   }
   ```
   - 在Radix Tree中记录routing决策
   - 用于TTL和pruning机制

### `KvIndexer.find_matches()` 详细流程

**Radix Tree查找：**
- 从root节点开始
- 沿着block hashes路径遍历
- 收集每个worker在该路径上的block数量
- 返回overlap scores

### `KvScheduler.schedule()` 详细流程

**成本计算（Logit计算）：**

实际的成本计算公式（在`DefaultWorkerSelector.select_worker()`中）：

```rust
// 对于每个worker和dp_rank组合：
let overlap = overlaps.get(&worker).unwrap_or(0);  // 重叠的block数量

// 计算prefill tokens（考虑overlap）
let prefill_token = prefill_tokens.get(&worker).unwrap_or(&isl);
let potential_prefill_block = prefill_token as f64 / block_size as f64;

// 计算decode blocks（考虑当前active sequences）
let decode_block = decode_blocks.get(&worker).unwrap_or(&potential_prefill_block) as f64;

// 计算logit（成本，越低越好）
let overlap_weight = router_config_override.overlap_score_weight 
                     .unwrap_or(kv_router_config.overlap_score_weight);
let logit = overlap_weight * potential_prefill_block + decode_block;
```

**公式说明：**
- `logit = overlap_weight * prefill_blocks + decode_blocks`
- `overlap_weight`：控制overlap的重要性（默认1.0）
- `prefill_blocks`：需要prefill的block数量（考虑overlap后）
- `decode_blocks`：当前worker的decode负载（active sequences）

**选择策略：**
- 如果`router_temperature == 0`：确定性选择最低logit的worker（如果有tie，使用tree size作为tie-breaker）
- 如果`router_temperature > 0`：使用softmax采样，引入随机性
  - 将logits转换为概率分布
  - 根据概率采样选择worker

## 两种KV Routing模式

### 1. Exact KV Routing (use_kv_events = true)
- **特点**：基于真实的KV events
- **需要NATS**：是
- **工作原理**：
  - Workers通过NATS发送KV events（block创建/删除）
  - Indexer订阅这些events并更新Radix Tree
  - Router基于实时状态做决策

### 2. Approximate KV Routing (use_kv_events = false)
- **特点**：基于routing决策预测
- **需要NATS**：否
- **工作原理**：
  - Router记录自己的routing决策
  - 使用TTL（默认120秒）过期blocks
  - 使用pruning机制限制tree大小
  - 基于预测状态做决策

## PrefillRouter工作流程

**Disaggregated Serving场景：**

1. **Prefill阶段**
   ```
   PrefillRouter.generate()
       └─ call_prefill()
           └─ 调用prefill worker（使用KV routing）
               └─ 返回disaggregated_params
   ```

2. **Decode阶段**
   ```
   PrefillRouter.generate()
       └─ 将disaggregated_params注入decode请求
           └─ 设置overlap_score_weight = 0（强制使用prefill worker）
               └─ 转发到decode worker
   ```

## 关键数据结构

### WorkerWithDpRank
```rust
pub struct WorkerWithDpRank {
    pub worker_id: WorkerId,
    pub dp_rank: DpRank,
}
```
- 用于data parallel场景，支持同一worker的多个rank

### OverlapScores
```rust
pub struct OverlapScores {
    pub scores: HashMap<WorkerWithDpRank, u32>,  // worker -> overlap blocks
    pub frequencies: Vec<u32>,  // block frequency
    pub tree_sizes: HashMap<WorkerId, usize>,  // worker -> tree size
}
```

### SchedulingRequest
```rust
pub struct SchedulingRequest {
    pub maybe_request_id: Option<String>,
    pub token_seq: Option<Vec<SequenceHash>>,
    pub isl_tokens: usize,
    pub overlaps: OverlapScores,
    pub decode_blocks: HashMap<WorkerWithDpRank, usize>,
    pub prefill_tokens: HashMap<WorkerWithDpRank, usize>,
    pub router_config_override: Option<RouterConfigOverride>,
    pub update_states: bool,
}
```

## 配置参数

### KvRouterConfig
- `overlap_score_weight`: overlap分数权重（默认1.0）
- `router_temperature`: 选择随机性（默认0.0，确定性）
- `use_kv_events`: 是否使用KV events（默认true）
- `router_track_active_blocks`: 是否跟踪active blocks（默认true）
- `router_ttl_secs`: TTL秒数（默认120.0，仅approximate模式）
- `router_max_tree_size`: 最大tree大小（默认1024，仅approximate模式）

## 状态管理

### Request生命周期跟踪

1. **add_request()**: 请求开始时添加到scheduler
2. **mark_prefill_completed()**: Prefill完成时标记
3. **free()**: 请求完成时清理

### Active Sequences跟踪

- Scheduler维护每个worker的active sequences
- 用于计算decode负载
- 支持replica sync（多router实例同步）

## 总结

Dynamo的routing系统是一个复杂的多层系统：

1. **入口层**：HTTP/TCP/NATS endpoint接收请求
2. **Pipeline层**：Preprocessor → Backend → Migration → PrefillRouter → Router
3. **Routing层**：KvRouter → KvIndexer + KvScheduler
4. **执行层**：PushRouter转发到worker

核心优势：
- KV cache感知，最大化cache命中率
- 负载均衡，考虑worker负载
- 支持disaggregated serving
- 支持exact和approximate两种模式
- 支持data parallel场景

