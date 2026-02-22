BASE=$(conda info --base)
export LD_LIBRARY_PATH="$BASE/lib:$LD_LIBRARY_PATH"

### Single Node:
python -m dynamo.frontend --router-mode kv --router-reset-states

CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE=lmcache.yaml \
python -m dynamo.vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.8 \
  --connector lmcache \
  --kv-events-config '{"enable_kv_cache_events":"True","publisher":"zmq","topic":"kv-events"}'

### Multi Node:
#### s7:
export HEAD_NODE_IP="192.168.3.67"
export NATS_SERVER="nats://${HEAD_NODE_IP}:4222"
export ETCD_ENDPOINTS="${HEAD_NODE_IP}:2379"

python -m dynamo.frontend --router-mode kv-strata --router-reset-states --kv-cache-block-size 256

sudo -E env CUDA_VISIBLE_DEVICES=0 \
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=lmcache.yaml \
./venv/bin/python -m dynamo.vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.8 \
  --connector lmcache \
  --kv-events-config '{"enable_kv_cache_events":"True","publisher":"zmq","topic":"kv-events"}' \
  --block-size 256
#### s6:
sudo -E env CUDA_VISIBLE_DEVICES=0 \
PYTHONHASHSEED=0 \
LMCACHE_CONFIG_FILE=lmcache.yaml \
./venv/bin/python -m dynamo.vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.8 \
  --connector lmcache \
  --kv-events-config '{"enable_kv_cache_events":"True","publisher":"zmq","topic":"kv-events"}'
  --block-size 256

curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "prompt": "<|begin_of_text|><|system|>\nYou are a helpful AI assistant.\n<|user|>\nWhat is the capital of France?\n<|assistant|>",
  "max_tokens": 100,
  "temperature": 0.7
}'

curl http://localhost:8000/v1/completions \
-H "Content-Type: application/json" \
-d '{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "prompt": "<|begin_of_text|><|system|>\nYou are a helpful AI assistant. You are careful, precise, and explain things clearly when needed. You always read the full context before answering.\n\n<|user|>\nI am going to ask you a very simple factual question, but before that, please carefully read the following background information. This background is provided to test your ability to process long context correctly.\n\nBackground section 1:\nFrance is a country located primarily in Western Europe, although it also has overseas regions and territories. It is known for its long history, cultural influence, cuisine, art, philosophy, and political thought. France has played a significant role in European and world history, including during the Roman period, the Middle Ages, the Renaissance, the Enlightenment, and the modern era.\n\nBackground section 2:\nThe political system of France is a semi-presidential republic. It has multiple large cities that are economically and culturally important. These cities include Paris, Lyon, Marseille, Toulouse, Nice, Nantes, Strasbourg, and others. Among these, one city serves as the seat of government, the main administrative center, and the symbolic heart of the nation.\n\nBackground section 3:\nWhen answering the question below, you should rely on well-established geographic and political knowledge. The answer should be concise and factual. Do not include unnecessary explanation unless explicitly requested.\n\nNow, based on all the information above, answer the following question:\n\nWhat is the capital of France?\n\n<|assistant|>",
  "max_tokens": 100,
  "temperature": 0.7
}'


# Install from Source
# ===================
# 如果需要在 main branch 上从源码安装（版本尚未发布到 PyPI），请按照以下步骤操作：

## 1. 安装系统依赖

### Ubuntu:
```bash
sudo apt install -y build-essential libhwloc-dev libudev-dev pkg-config libclang-dev protobuf-compiler python3-dev cmake
```

## 4. 安装构建工具

```bash
uv pip install pip maturin
```

## 6. 构建 Rust bindings (ai-dynamo-runtime)

```bash
cd lib/bindings/python
source "$HOME/.cargo/env"
# 如果同时设置了 CONDA_PREFIX 和 VIRTUAL_ENV，需要取消设置其中一个
unset CONDA_PREFIX  # 如果使用 venv
maturin develop --uv
```

这一步会在本地构建并安装 `ai-dynamo-runtime`，无需从 PyPI 下载。

## 7. 安装主包 (ai-dynamo)

```bash
cd /path/to/dynamo
source venv/bin/activate
uv pip install -e ".[vllm]"  # 或 [sglang], [trtllm]
```

## 7. 安装LMCache

```bash
uv pip install -e ./LMCache --no-build-isolation
```

使用 `-e` 进行可编辑安装，这样修改代码后无需重新安装。

## 7. 安装vllm

```bash
VLLM_USE_PRECOMPILED=1 uv pip install -e ~/vllm
```

## 8. 验证安装

```bash
python -c "import dynamo; print('ai-dynamo installed')"
python -c "import dynamo._core; print('ai-dynamo-runtime imported successfully')"
```