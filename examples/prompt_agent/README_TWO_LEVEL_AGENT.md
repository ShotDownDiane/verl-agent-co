# Two-Level Agent Pipeline

这是一个使用 VLM (Vision Language Model) 和 LLM (Large Language Model) 的两级 Agent 系统，用于解决优化问题。

## 架构

1. **策略生成 (Strategy Generation)**: 使用 VLM Agent 分析问题（包括可视化图片）并生成解决策略
2. **解决方案生成 (Solution Generation)**: 使用 LLM Agent 基于策略生成具体的解决方案

## 工作流程

```
环境观察 (文本 + 可视化图片)
    ↓
VLM Agent → 策略生成
    ↓
LLM Agent → 解决方案生成
    ↓
环境执行 → 奖励评估
```

## 前置要求

1. **启动 vLLM 服务**:
   ```bash
   # 启动 VLM 服务 (端口 8000)
   bash start_vlm_with_vllm.sh
   
   # 启动 LLM 服务 (端口 8001)
   bash start_llm_with_vllm.sh
   
   # 或者同时启动两个服务
   bash start_all_with_vllm.sh
   ```

2. **检查服务状态**:
   ```bash
   curl http://localhost:8000/health  # VLM 服务
   curl http://localhost:8001/health  # LLM 服务
   ```

## 使用方法

### 方法 1: 使用运行脚本（推荐）

```bash
# 默认运行 TSP 问题
bash run_two_level_agent.sh

# 指定环境和子环境
bash run_two_level_agent.sh ml4co-kit/tsp tsp
bash run_two_level_agent.sh ml4co-kit/cvrp cvrp
bash run_two_level_agent.sh ml4co-kit/op op
```

### 方法 2: 直接运行 Python 脚本

```bash
# 基本用法
python3 two_level_agent_pipeline.py ml4co-kit/tsp tsp

# 使用环境变量配置
export VLM_API_URL="http://localhost:8000/v1"
export LLM_API_URL="http://localhost:8001/v1"
export API_KEY="token-abc123456"
export ENV_NUM=3
export TEST_TIMES=1

python3 two_level_agent_pipeline.py ml4co-kit/tsp tsp
```

## 环境变量配置

- `VLM_API_URL`: VLM API 服务地址（默认: `http://localhost:8000/v1`）
- `LLM_API_URL`: LLM API 服务地址（默认: `http://localhost:8001/v1`）
- `API_KEY`: API 密钥（默认: `token-abc123456`）
- `VLM_MODEL_NAME`: VLM 模型名称（默认: `vlm`）
- `LLM_MODEL_NAME`: LLM 模型名称（默认: `llm`）
- `ENV_NUM`: 每个测试的环境数量（默认: `3`）
- `TEST_TIMES`: 测试次数（默认: `1`）

## 支持的环境

- `ml4co-kit/tsp`: 旅行商问题
- `ml4co-kit/cvrp`: 车辆路径问题
- `ml4co-kit/op`: 定向问题
- `ml4co-kit/jssp`: 作业车间调度问题
- `ml4co-kit/pfsp`: 流水车间调度问题

## 输出

Pipeline 会生成：
1. **日志文件**: `logs/two_level_agent/ml4co-kit_{sub_env}/run_log_{timestamp}.log`
2. **控制台输出**: 包含策略、解决方案和奖励信息
3. **轨迹数据**: `logs/two_level_agent/ml4co-kit_{sub_env}/trajectories_{timestamp}/`

### 轨迹数据格式

每个轨迹包含以下文件：

#### 文本轨迹 (JSON)
- `test_{test_idx}_env_{env_idx}_trajectory.json`: 包含观察、策略、解决方案、奖励等信息
- `test_{test_idx}_summary.json`: 每个测试的汇总信息
- `final_summary.json`: 所有测试的最终汇总

#### 图片轨迹 (PNG)
- `test_{test_idx}_env_{env_idx}_image.png`: 环境可视化图片（如果生成成功）

#### 轨迹 JSON 结构示例

```json
{
  "test_idx": 0,
  "env_idx": 0,
  "timestamp": "2025-12-20T15:30:00",
  "observation": "问题描述...",
  "image_path": "test_0_env_0_image.png",
  "strategy": "策略内容...",
  "solution": "解决方案...",
  "reward": 0.8523,
  "env_reward": 0.8523,
  "format_bonus": 0.05,
  "feasibility_bonus": 0.15,
  "info": {
    "route": [0, 1, 2, 3, 0],
    ...
  }
}
```

## 示例输出

```
============================================================
Two-Level Agent Pipeline
============================================================
Environment: ml4co-kit/tsp, Sub-env: tsp
VLM API: http://localhost:8000/v1, Model: vlm
LLM API: http://localhost:8001/v1, Model: llm
Env num: 3, Test times: 1
============================================================

--- Environment 0 ---
Observation: [问题描述...]
Env 0: Generating strategy with VLM...
Env 0 Strategy: [策略内容...]
Env 0: Generating solution with LLM...
Env 0 Solution: [解决方案...]
Env 0 Final Reward: 0.8523
```

## 故障排除

1. **服务未启动**: 确保 vLLM 服务正在运行
2. **端口冲突**: 检查端口 8000 和 8001 是否被占用
3. **模型加载失败**: 检查模型路径是否正确
4. **可视化失败**: 这是正常的，系统会自动回退到仅文本模式

## 技术说明

- **VLM Agent**: 使用 `vlm_agent.py`，支持 API、vLLM 和 transformers 三种模式
- **LLM Agent**: 使用 `llm_agent.py`，支持 API、vLLM 和 transformers 三种模式
- **可视化**: 使用 `visualization.py` 模块从环境数据生成图片
- **环境管理**: 使用 `agent_system.environments.env_manager` 管理环境

