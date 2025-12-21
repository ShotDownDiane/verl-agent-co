# 本地模型服务使用指南

本指南说明如何使用本地部署的 Qwen3-4B 和 Qwen3VL-4B-Instruct 模型进行 Two-Level Agent 测试。

## 前置要求

1. **安装 vLLM**
   ```bash
   pip install vllm
   ```

2. **模型路径**
   - Qwen3-4B: `/root/autodl-tmp/Qwen3-4B`
   - Qwen3VL-4B-Instruct: `/root/autodl-tmp/Qwen3VL-4B-Instruct`

## 使用步骤

### 1. 启动本地模型服务

#### 方法一：使用 Python 脚本（推荐）
```bash
cd /root/autodl-tmp/verl-agent-co/examples/prompt_agent
python start_local_models.py
```

#### 方法二：使用 Shell 脚本
```bash
cd /root/autodl-tmp/verl-agent-co/examples/prompt_agent
chmod +x start_local_models.sh
./start_local_models.sh
```

#### 方法三：手动启动（分别启动两个服务）

**终端 1 - 启动 VLM 服务（端口 8000）:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/Qwen3VL-4B-Instruct \
    --port 8000 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --served-model-name Qwen3VL-4B-Instruct
```

**终端 2 - 启动 LLM 服务（端口 8001）:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /root/autodl-tmp/Qwen3-4B \
    --port 8001 \
    --host 0.0.0.0 \
    --trust-remote-code \
    --served-model-name Qwen3-4B
```

### 2. 验证服务是否启动成功

等待服务启动完成（通常需要几分钟），然后测试：

```bash
# 测试 VLM 服务
curl http://localhost:8000/v1/models

# 测试 LLM 服务
curl http://localhost:8001/v1/models
```

### 3. 运行 Two-Level Agent 测试

```bash
cd /root/autodl-tmp/verl-agent-co/examples/prompt_agent
python siliconflow_ml4cokit_two_level_local.py ml4co-kit/tsp
```

### 4. 自定义配置

#### 使用环境变量指定服务地址：
```bash
export VLM_API_BASE_URL="http://localhost:8000/v1"
export LLM_API_BASE_URL="http://localhost:8001/v1"
python siliconflow_ml4cokit_two_level_local.py ml4co-kit/tsp
```

#### 使用命令行参数指定模型：
```bash
python siliconflow_ml4cokit_two_level_local.py ml4co-kit/tsp Qwen3VL-4B-Instruct Qwen3-4B
```

## 端口配置

默认端口：
- VLM 服务: `8000`
- LLM 服务: `8001`

如需修改端口，可以在启动服务时指定：
```bash
python start_local_models.py --vlm-port 8002 --llm-port 8003
```

然后在运行测试时设置环境变量：
```bash
export VLM_API_BASE_URL="http://localhost:8002/v1"
export LLM_API_BASE_URL="http://localhost:8003/v1"
```

## 故障排查

### 1. 服务启动失败
- 检查模型路径是否正确
- 检查是否有足够的 GPU 内存
- 查看服务日志中的错误信息

### 2. 连接超时
- 确保服务已完全启动（等待几分钟）
- 检查端口是否被占用：`netstat -tuln | grep 8000`
- 检查防火墙设置

### 3. 模型加载失败
- 确保模型文件完整
- 检查 `trust-remote-code` 参数是否设置
- 查看 vLLM 的日志输出

### 4. API 调用失败
- 检查服务地址是否正确
- 确认服务正在运行：`curl http://localhost:8000/v1/models`
- 查看 agent 日志中的错误信息

## 性能优化

1. **GPU 内存不足**：使用量化或减少 batch size
2. **速度慢**：确保使用 GPU，检查 CUDA 是否正常工作
3. **并发请求**：vLLM 默认支持并发，可以根据需要调整

## 注意事项

- 确保有足够的 GPU 内存（建议至少 16GB）
- 首次启动服务需要加载模型，可能需要几分钟
- 两个服务需要分别启动，建议使用不同的终端窗口
- 服务会占用 GPU 资源，测试完成后记得停止服务

