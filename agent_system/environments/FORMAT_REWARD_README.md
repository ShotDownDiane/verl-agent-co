# Format Reward 功能说明

## 概述

Format Reward 是一个奖励机制，用于鼓励模型：
1. **按照要求的格式回复**（格式正确奖励）
2. **给出可行解**（可行解奖励）
3. **优化最终性能**（环境原始奖励）

## 奖励机制：条件奖励（Conditional Reward Mechanism）

采用**方案B：条件奖励机制**，参考 LLMCoSolver 的设计，避免 reward hacking：

### 核心设计原则

1. **Format Reward**：基础奖励，始终给予
2. **Feasibility Reward**：只有格式正确才给予（条件：format_bonus = 1）
3. **Environment Reward**：只有高可行性才给予完整权重（条件：feasibility >= threshold）

### 奖励计算公式

```
# 条件奖励机制（默认）
format_reward = format_reward_weight * format_bonus
feasibility_reward = feasibility_reward_weight * feasibility_bonus * format_bonus
env_reward_weight = (feasibility >= threshold) ? full_weight : 0.1 * full_weight
env_reward = normalized_env_reward * env_reward_weight

final_reward = format_reward + feasibility_reward + env_reward
```

这种设计确保模型必须**同时满足格式、可行性和质量**才能获得高奖励，避免只优化格式而忽略解质量的问题。

## 配置选项

### Routing 环境（TSP/CVRP/OP）

在配置文件中添加以下选项：

```yaml
env:
  rl4co:
    env_name: tsp  # 或 cvrp, op
    device: cpu
    one_step: True  # 推荐使用 one-step 模式
    # Format reward 配置
    use_format_reward: True  # 启用 format reward
    format_reward_weight: 0.05  # 格式正确奖励权重（默认 0.05）
    feasibility_reward_weight: 0.15  # 可行解奖励权重（默认 0.15）
    # 条件奖励机制参数
    use_conditional_reward: True  # 使用条件奖励机制（默认 True）
    feasibility_threshold: 0.9  # 高可行性门槛（默认 0.9）
    normalize_env_reward: True  # 归一化环境 reward（默认 True）
    env_reward_range: [-20.0, 0.0]  # 可选：环境 reward 范围（用于归一化）
    fixed_scale_reference: 8.0  # 可选：固定缩放参考值（避免 batch 不稳定）
    generator_params:
      num_loc: 20
      min_loc: 0.0
      max_loc: 1.0
```

### Scheduling 环境（JSSP/FFSP）

```yaml
env:
  rl4co_scheduling:
    env_name: jssp  # 或 ffsp
    device: cpu
    one_step: True  # 推荐使用 one-step 模式
    # Format reward 配置
    use_format_reward: True  # 启用 format reward
    format_reward_weight: 0.05  # 格式正确奖励权重（默认 0.05）
    feasibility_reward_weight: 0.15  # 可行解奖励权重（默认 0.15）
    # 条件奖励机制参数
    use_conditional_reward: True  # 使用条件奖励机制（默认 True）
    feasibility_threshold: 0.9  # 高可行性门槛（默认 0.9）
    normalize_env_reward: True  # 归一化环境 reward（默认 True）
    env_reward_range: [-20.0, 0.0]  # 可选：环境 reward 范围
    fixed_scale_reference: 8.0  # 可选：固定缩放参考值
    generator_params: {}
```

## 奖励计算逻辑

### 1. 格式正确奖励（Format Bonus）

- **One-step 模式**：
  - 如果 action 能够成功解析为有效的 route/schedule，`format_bonus = 1.0`
  - 判断标准：`valids[i] == 1`（由 projection 函数返回）
  - 奖励：`format_reward = format_reward_weight * format_bonus`

- **Step-by-step 模式**：
  - 如果单个 action 能够成功解析为有效的整数索引，`format_bonus = 1.0`
  - 奖励：`format_reward = format_reward_weight * format_bonus`

### 2. 可行解奖励（Feasibility Bonus）

**条件：只有格式正确才给予可行性奖励**

#### TSP（旅行商问题）
可行解需要满足：
- 访问所有节点恰好一次（除了可能返回起点）
- 所有节点索引有效（0 <= idx < num_nodes）
- 无重复访问（除了起点）

#### CVRP（车辆路径问题）
可行解需要满足：
- 每个 route 从 depot 开始并结束于 depot
- 所有客户节点被访问恰好一次
- 所有节点索引有效

#### JSSP/FFSP（调度问题）
基本可行性检查：
- 所有索引非负
- 如果提供了 num_jobs，所有索引 < num_jobs

**注意**：完整的可行性检查需要验证约束条件，当前实现为基础检查，可以后续扩展。

### 3. 环境原始奖励（Env Reward）

**条件：只有高可行性才给予完整权重**

- 来自 rl4co 环境计算的原始奖励（通常为负数，如 -7 到 -9）
- TSP/CVRP：负的总距离
- OP：负的距离 + 正的 prize
- JSSP/FFSP：负的 makespan

**归一化处理**：
- 环境 reward 被归一化到 [0, 1] 范围
- 如果 `feasibility >= feasibility_threshold`：给予完整权重（如 0.8）
- 如果 `feasibility < feasibility_threshold`：只给予 10% 的权重（如 0.08）

这确保了只有高质量的解才能获得完整的环境奖励。

## 信息输出

在 `infos` 中会包含以下额外信息：

```python
info = {
    "route": [0, 5, 2, 3, ...],  # 或 "schedule": [...]
    "is_action_valid": True,  # 格式是否正确
    "format_bonus": 1.0,  # 格式 bonus（0 或 1）
    "feasibility_bonus": 1.0,  # 可行解 bonus（0 或 1）
    "is_feasible": True,  # 是否为可行解
    "env_reward": -15.3,  # 环境原始奖励
    # 条件奖励机制的详细信息（如果启用）
    "format_reward": 0.05,  # 格式奖励值
    "feasibility_reward": 0.15,  # 可行解奖励值（已应用格式条件）
    "scaled_env_reward": 0.64,  # 缩放后的环境奖励
    "env_reward_weight": 0.8,  # 环境奖励的权重
    "meets_feasibility_threshold": True,  # 是否满足可行性门槛
}
```

## 使用示例

### Python 代码配置

```python
from omegaconf import OmegaConf
from verl.trainer.config import ppo_trainer

cfg = OmegaConf.structured(ppo_trainer)
cfg.env.env_name = "rl4co/tsp"
cfg.env.rl4co.env_name = "tsp"
cfg.env.rl4co.one_step = True
cfg.env.rl4co.use_format_reward = True
cfg.env.rl4co.format_reward_weight = 0.1
cfg.env.rl4co.feasibility_reward_weight = 0.2
```

### 权重调优建议

1. **format_reward_weight**：
   - 建议范围：0.05 - 0.2
   - 如果模型经常格式错误，可以适当提高
   - 如果模型已经能很好遵循格式，可以降低或设为 0

2. **feasibility_reward_weight**：
   - 建议范围：0.1 - 0.5
   - 如果问题约束复杂，可行解较少，可以适当提高
   - 注意不要过大，以免掩盖环境奖励的重要性

3. **环境奖励范围**：
   - TSP 20 节点：通常 -20 到 -10
   - 确保 format reward 不会完全掩盖环境奖励
   - 建议：`format_reward_weight + feasibility_reward_weight < |env_reward| / 10`

## 注意事项

1. **条件奖励机制的优势**：
   - 避免 reward hacking：模型必须同时满足格式、可行性和质量
   - 训练稳定：使用归一化和固定参考值
   - 奖励平衡：环境 reward 占主导，format/feasibility 是辅助

2. **One-step vs Step-by-step**：
   - One-step 模式支持完整的可行解检查和条件奖励机制
   - Step-by-step 模式只检查格式正确性，可行性检查较简单

3. **性能影响**：
   - Format reward 计算开销很小
   - 可行解检查需要遍历 route/schedule，但通常很快
   - 归一化操作开销可忽略

4. **调试**：
   - 通过 `infos` 中的信息监控：
     - `format_bonus`: 格式正确率
     - `feasibility_bonus`: 可行解率
     - `meets_feasibility_threshold`: 高可行性比例
     - `env_reward_weight`: 实际环境 reward 权重
   - 如果可行解率很低，可能需要调整 prompt 或模型训练策略
   - 如果 `env_reward_weight` 经常是 0.08 而不是 0.8，说明可行性门槛可能设置过高

5. **避免 Reward Hacking**：
   - 条件奖励机制确保模型不能只优化格式而忽略解质量
   - 只有高可行性的解才能获得完整的环境 reward 权重
   - 格式错误会导致可行性 reward 为 0

## 实现细节

Format reward 实现在以下文件中：
- `agent_system/environments/format_reward.py`：核心计算逻辑
- `agent_system/environments/env_manager.py`：集成到环境管理器

