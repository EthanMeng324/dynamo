BASE=$(conda info --base)
export LD_LIBRARY_PATH="$BASE/lib:$LD_LIBRARY_PATH"

python -m dynamo.frontend --router-mode kv

CUDA_VISIBLE_DEVICES=0 \
LMCACHE_CONFIG_FILE=lmcache.yaml \
python -m dynamo.vllm \
  --model Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization 0.8 \
  --connector lmcache \
  --kv-events-config '{"enable_kv_cache_events":"True","publisher":"zmq","topic":"kv-events"}'

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

### macOS:
```bash
brew install cmake protobuf
```

## 2. 安装 Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

验证安装：
```bash
rustc --version  # 应该显示 1.90.0 或更高版本
```

## 3. 创建 Python 虚拟环境

```bash
# 安装 uv（如果还没有）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 创建虚拟环境
cd /path/to/dynamo
uv venv venv
source venv/bin/activate
```

## 4. 安装构建工具

```bash
uv pip install pip maturin
```

## 5. 升级 protoc（如果版本过旧）

检查当前版本：
```bash
protoc --version
```

如果版本低于 3.12，需要升级：
```bash
cd /tmp
wget https://github.com/protocolbuffers/protobuf/releases/download/v27.1/protoc-27.1-linux-x86_64.zip -O protoc.zip
unzip -q -o protoc.zip -d /tmp/protoc_install
sudo cp /tmp/protoc_install/bin/protoc /usr/local/bin/
sudo chmod +x /usr/local/bin/protoc
rm -rf /tmp/protoc_install /tmp/protoc.zip
protoc --version  # 验证版本
```

## 6. 构建 Rust bindings (ai-dynamo-runtime)

```bash
cd lib/bindings/python
source "$HOME/.cargo/env"
# 如果同时设置了 CONDA_PREFIX 和 VIRTUAL_ENV，需要取消设置其中一个
unset CONDA_PREFIX  # 如果使用 venv
source /path/to/dynamo/venv/bin/activate
maturin develop --uv
```

这一步会在本地构建并安装 `ai-dynamo-runtime`，无需从 PyPI 下载。

## 7. 安装主包 (ai-dynamo)

```bash
cd /path/to/dynamo
source venv/bin/activate
uv pip install -e ".[vllm]"  # 或 [sglang], [trtllm]
```

使用 `-e` 进行可编辑安装，这样修改代码后无需重新安装。

## 8. 验证安装

```bash
python -c "import dynamo; print('ai-dynamo installed')"
python -c "import dynamo._core; print('ai-dynamo-runtime imported successfully')"
```

## 常见问题

### 磁盘空间不足
如果遇到 "No space left on device" 错误：
```bash
# 清理 Cargo 缓存
cargo clean
rm -rf ~/.cargo/registry/cache

# 清理构建目录
cd lib/bindings/python
cargo clean
```

### protoc 版本问题
确保 protoc 版本 >= 3.12，否则会遇到 `--experimental_allow_proto3_optional` 错误。

### CONDA_PREFIX 冲突
如果同时设置了 `CONDA_PREFIX` 和 `VIRTUAL_ENV`，maturin 会报错。在使用 venv 时，取消设置 `CONDA_PREFIX`：
```bash
unset CONDA_PREFIX
```

## 注意事项

- 从源码安装需要足够的磁盘空间（建议至少 15GB 可用空间）
- 首次构建可能需要较长时间（10-30 分钟，取决于机器性能）
- 如果修改了 Rust 代码，需要重新运行 `maturin develop --uv` 来重新构建