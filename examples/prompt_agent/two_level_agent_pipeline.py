"""
Two-Level Agent Pipeline
使用 VLM 进行策略生成，LLM 进行解决方案生成
"""
import os
import sys
import numpy as np
import time
import logging
import base64
import io
import json
from datetime import datetime
from agent_system.environments.env_manager import *
from vlm_agent import VLMAgent
from llm_agent import LLMAgent
import termcolor
from types import SimpleNamespace
from functools import partial
from visualization import visualize_tsp_from_td, visualize_cvrp_from_td, visualize_op_from_td


def build_env(env_name, env_num=1, sub_env="tsp"):
    """
    Build ml4co-kit environment.
    
    Args:
        env_name: Should be "ml4co-kit" or "ml4co-kit/tsp" etc.
        env_num: Number of environments
        sub_env: Sub environment name (tsp, cvrp, op, jssp, pfsp)
    """
    group_n = 1
    
    if env_name.startswith("ml4co-kit") or env_name == "ml4co-kit":
        from agent_system.environments.env_package.ml4co_kit import (
            build_ml4cokit_routing_envs,
            build_ml4cokit_scheduling_envs,
            ml4cokit_projection,
            ml4cokit_scheduling_projection,
        )
        
        # Extract sub_env from env_name if provided
        if "/" in env_name:
            sub_env = env_name.split("/", 1)[1]
        
        # Create a simple config object
        resources = SimpleNamespace(num_cpus=0.1, num_gpus=0)
        
        # Default generator params for TSP
        generator_params = {
            "num_loc": 20,  # Number of nodes
            "min_loc": 0.0,
            "max_loc": 1.0,
        }
        
        # Default ml4co_kit config
        ml4co_cfg = SimpleNamespace(
            env_name=sub_env,
            device="cpu",
            generator_params=generator_params,
            rl4co_kwargs={},
            use_format_reward=False,
            format_reward_weight=0.05,
            feasibility_reward_weight=0.15,
            use_conditional_reward=True,
            feasibility_threshold=0.9,
            normalize_env_reward=True,
            env_reward_range=None,
            fixed_scale_reference=None,
        )
        
        # Create env config (structure matches env_manager.py expectations)
        env_cfg = SimpleNamespace(
            env=SimpleNamespace(
                env_name=f"ml4co-kit/{sub_env}",
                seed=1,
                max_steps=1,  # One-shot mode
                history_length=0,
                rollout=SimpleNamespace(n=1),
                resources_per_worker=resources,
                ml4co_kit=ml4co_cfg,
                rl4co=SimpleNamespace(),  # Fallback for compatibility
                rl4co_scheduling=SimpleNamespace(),  # Fallback for compatibility
            )
        )
        
        # Build environments based on sub_env type
        if sub_env in ("tsp", "cvrp", "op"):
            _envs = build_ml4cokit_routing_envs(
                env_name=sub_env,
                seed=1,
                mode="test",
                env_num=env_num,
                group_n=group_n,
                device=ml4co_cfg.device,
                generator_params=generator_params,
                rl4co_kwargs=ml4co_cfg.rl4co_kwargs,
            )
            projection_f = partial(ml4cokit_projection, env_name=sub_env)
            env_manager = ML4COKitRoutingEnvironmentManager(_envs, projection_f, env_cfg, env_name=sub_env)
        elif sub_env in ("jssp", "pfsp"):
            _envs = build_ml4cokit_scheduling_envs(
                env_name=sub_env,
                seed=1,
                env_num=env_num,
                group_n=group_n,
                device=ml4co_cfg.device,
                generator_params=generator_params,
                rl4co_kwargs=ml4co_cfg.rl4co_kwargs,
            )
            projection_f = partial(ml4cokit_scheduling_projection, env_name=sub_env)
            env_manager = ML4COKitSchedulingEnvironmentManager(_envs, projection_f, env_cfg, env_name=sub_env)
        else:
            raise ValueError(f"Unsupported ml4co-kit sub environment: {sub_env}")
    else:
        raise ValueError(f"Unsupported environment name: {env_name}")
    return env_manager


def create_visualization_image(env_manager, env_idx=0, sub_env="tsp"):
    """
    创建环境可视化图片
    
    Args:
        env_manager: 环境管理器
        env_idx: 环境索引
        sub_env: 子环境名称 (tsp, cvrp, op)
    
    Returns:
        base64 编码的图片字符串，如果失败则返回 None
    """
    try:
        # Prefer environment-provided instance data (current state)
        if hasattr(env_manager, "get_instance_data"):
            instance_data = env_manager.get_instance_data(env_idx)
            if instance_data:
                # visualization.visualize_from_instance_data expects a dict like {'locs':..., 'env_name':...}
                try:
                    from visualization import visualize_from_instance_data
                    return visualize_from_instance_data(instance_data, idx=0)
                except Exception:
                    # Fallback to specific visualizers if available
                    if sub_env == "cvrp":
                        return visualize_cvrp_from_td(instance_data, 0)
                    elif sub_env == "op":
                        return visualize_op_from_td(instance_data, 0)
                    else:
                        return visualize_tsp_from_td(instance_data, 0)

        # Fallback: try to access underlying envs (least preferred)
        if hasattr(env_manager, "_envs") and env_manager._envs is not None:
            envs = env_manager._envs
            try:
                # If envs supports a method to get current td, prefer that
                if hasattr(env_manager, "current_td") and env_manager.current_td is not None:
                    td = env_manager.current_td
                    if sub_env == "cvrp":
                        return visualize_cvrp_from_td(td, env_idx)
                    elif sub_env == "op":
                        return visualize_op_from_td(td, env_idx)
                    else:
                        return visualize_tsp_from_td(td, env_idx)
            except Exception as e:
                logging.warning(f"Failed to visualize from current_td: {e}")

        return None
    except Exception as e:
        logging.warning(f"Failed to create visualization: {e}")
        return None


def save_trajectory(trajectory_dir, test_idx, env_idx, trajectory_data):
    """
    保存轨迹数据（文本和图片）
    
    Args:
        trajectory_dir: 轨迹保存目录
        test_idx: 测试索引
        env_idx: 环境索引
        trajectory_data: 轨迹数据字典，包含：
            - observation: 观察文本
            - image_base64: 图片 base64 编码（可选）
            - strategy: 策略文本
            - solution: 解决方案文本
            - reward: 奖励
            - env_reward: 环境奖励
            - format_bonus: 格式奖励
            - feasibility_bonus: 可行性奖励
            - info: 其他信息（可选）
    """
    os.makedirs(trajectory_dir, exist_ok=True)
    
    # 保存图片（如果有）
    image_path = None
    if trajectory_data.get('image_base64'):
        image_filename = f"test_{test_idx}_env_{env_idx}_image.png"
        image_path = os.path.join(trajectory_dir, image_filename)
        try:
            image_data = base64.b64decode(trajectory_data['image_base64'])
            with open(image_path, 'wb') as f:
                f.write(image_data)
            logging.info(f"Saved image to {image_path}")
        except Exception as e:
            logging.warning(f"Failed to save image: {e}")
    
    # 准备 JSON 数据（不包含 base64 图片，只保存路径）
    json_data = {
        'test_idx': test_idx,
        'env_idx': env_idx,
        'timestamp': datetime.now().isoformat(),
        'observation': trajectory_data.get('observation', ''),
        'image_path': image_path if image_path else None,
        'strategy': trajectory_data.get('strategy', ''),
        'solution': trajectory_data.get('solution', ''),
        'reward': trajectory_data.get('reward', 0.0),
        'env_reward': trajectory_data.get('env_reward', 0.0),
        'format_bonus': trajectory_data.get('format_bonus', 0.0),
        'feasibility_bonus': trajectory_data.get('feasibility_bonus', 0.0),
    }
    
    # 添加其他信息（如果有）
    if 'info' in trajectory_data:
        json_data['info'] = trajectory_data['info']
    
    # 保存 JSON 文件
    json_filename = f"test_{test_idx}_env_{env_idx}_trajectory.json"
    json_path = os.path.join(trajectory_dir, json_filename)
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved trajectory JSON to {json_path}")
    except Exception as e:
        logging.error(f"Failed to save trajectory JSON: {e}")


class TwoLevelAgent:
    """Two-Level Agent: VLM for strategy, LLM for solution"""
    
    def __init__(
        self,
        vlm_agent: VLMAgent,
        llm_agent: LLMAgent,
    ):
        """
        初始化 Two-Level Agent
        
        Args:
            vlm_agent: VLM agent 用于策略生成
            llm_agent: LLM agent 用于解决方案生成
        """
        self.vlm_agent = vlm_agent
        self.llm_agent = llm_agent
        logging.info("✓ Two-Level Agent initialized")
    
    def generate_strategy(self, observation: str, image_base64: str = None) -> str:
        """
        使用 VLM 生成策略
        
        Args:
            observation: 环境观察（文本描述）
            image_base64: 环境可视化图片（base64编码）
        
        Returns:
            生成的策略
        """
        strategy_prompt = f"""Role: Spatial Optimization Strategist
Input Data: {observation} + [Image provided]

Task: Analyze the city distribution and k-NN constraints to:
1. Identify spatial clusters or geometric patterns.
2. Define a high-level routing logic (e.g., "follow the perimeter then clear the center").
3. Flag critical/isolated nodes that must be prioritized.

Output: Provide a concise execution strategy for the next agent. No code. No final route."""
        
        try:
            if image_base64:
                strategy = self.vlm_agent.generate(
                    strategy_prompt,
                    image=image_base64,
                    max_tokens=1024,
                    temperature=0.7,
                )
            else:
                strategy = self.vlm_agent.generate(
                    strategy_prompt,
                    max_tokens=512,
                    temperature=0.7,
                )
            
            logging.info(f"Strategy generated: {strategy[:200]}...")
            return strategy
        except Exception as e:
            logging.error(f"Strategy generation failed: {e}")
            # 回退到简单策略
            return "使用贪心算法或启发式方法解决此优化问题。"
    
    def generate_solution(self, observation: str, strategy: str) -> str:
        """
        使用 LLM 基于策略生成解决方案
        
        Args:
            observation: 环境观察（文本描述）
            strategy: 策略（由 VLM 生成）
        
        Returns:
            生成的解决方案
        """
        solution_prompt = f"""Role: Precision Execution Engine
Inputs: {observation}, Strategy: {strategy}

Mandatory Rules:
1. Unique Visit: The "route" must include every city index exactly once.
2. Closed Loop: The path is a cycle (last node returns to first). Do not repeat the start node in the list.
3. Calculation: "objective" must be the total Euclidean distance of the entire cycle.
4. No Prose: No CoT, no explanations, no code, no markdown backticks. Output raw JSON only.

Output Format:
{{
  "route": [node_indices],
  "objective": total_distance
}}"""
        
        try:
            solution = self.llm_agent.generate(
                solution_prompt,
                max_tokens=2048,
                temperature=0.3,  # 较低温度以获得更确定的输出
            )
            
            logging.info(f"Solution generated: {solution[:200]}...")
            return solution
        except Exception as e:
            logging.error(f"Solution generation failed: {e}")
            raise


if __name__ == "__main__":
    # ======================= 参数设置 =======================
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default=os.environ.get("ENV_NAME", "ml4co-kit"))
    parser.add_argument("--sub_env", default=os.environ.get("SUB_ENV", "tsp"))
    parser.add_argument("--vlm_api_url", default=os.environ.get("VLM_API_URL", "http://localhost:8000/v1"))
    parser.add_argument("--llm_api_url", default=os.environ.get("LLM_API_URL", "http://localhost:8001/v1"))
    parser.add_argument("--api_key", default=os.environ.get("API_KEY", "token-abc123456"))
    parser.add_argument("--mode", default=os.environ.get("MODE", "train"))
    parser.add_argument("--env_num", default=os.environ.get("ENV_NUM", "3"))
    parser.add_argument("--test_times", default=os.environ.get("TEST_TIMES", "1"))
    parser.add_argument("--vlm_model_name", default=os.environ.get("VLM_MODEL_NAME", "vlm"))
    parser.add_argument("--llm_model_name", default=os.environ.get("LLM_MODEL_NAME", "llm"))
    args = parser.parse_args()

    env_name = args.env_name
    # If sub_env not provided explicitly, extract from env_name if possible, otherwise default to "tsp"
    sub_env = args.sub_env if args.sub_env is not None else (env_name.split("/", 1)[1] if "/" in env_name else "tsp")

    # API 配置
    vlm_api_url = args.vlm_api_url
    llm_api_url = args.llm_api_url
    api_key = args.api_key
    
    vlm_model_name = args.vlm_model_name
    llm_model_name = args.llm_model_name
    
    # 实验参数
    env_num = int(args.env_num)
    test_times = int(args.test_times)
    
    # 日志和轨迹目录设置
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"logs/two_level_agent/ml4co-kit_{sub_env}"
    trajectory_dir = os.path.join(log_dir, f"trajectories_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)
    
    log_fp = os.path.join(log_dir, f"run_log_{timestamp}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_fp, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    
    logging.info(f"Trajectory directory: {trajectory_dir}")
    
    # ======================= 初始化 =======================
    logging.info("="*60)
    logging.info("Two-Level Agent Pipeline")
    logging.info("="*60)
    logging.info(f"Environment: {env_name}, Sub-env: {sub_env}")
    logging.info(f"VLM API: {vlm_api_url}, Model: {vlm_model_name}")
    logging.info(f"LLM API: {llm_api_url}, Model: {llm_model_name}")
    logging.info(f"Env num: {env_num}, Test times: {test_times}")
    logging.info("="*60)
    
    # 构建环境
    logging.info(f"\nBuilding environment: {env_name}")
    env_manager = build_env(env_name, env_num, sub_env=sub_env)
    logging.info("✓ Environment built")
    
    # 初始化 Agents
    logging.info("\nInitializing agents...")
    # 支持自定义 api_client（例如使用 requests.Session 直接调用 remote API）
    api_client_type = os.environ.get("API_CLIENT_TYPE", "").lower()  # e.g. 'requests'
    api_client = None
    if api_client_type == "requests":
        try:
            import requests
            s = requests.Session()
            if api_key:
                s.headers.update({"Authorization": f"Bearer {api_key}"})
            api_client = s
            logging.info("Using requests.Session as api_client for remote API calls")
        except Exception as e:
            logging.warning(f"Failed to create requests.Session api_client: {e}")

    vlm_agent = VLMAgent(
        api_base_url=vlm_api_url,
        api_key=api_key,
        model_name=vlm_model_name,
        api_client=api_client,
    )
    logging.info("✓ VLM Agent initialized")
    
    llm_agent = LLMAgent(
        api_base_url=llm_api_url,
        api_key=api_key,
        model_name=llm_model_name,
        api_client=api_client,
    )
    logging.info("✓ LLM Agent initialized")
    
    two_level_agent = TwoLevelAgent(vlm_agent, llm_agent)
    logging.info("✓ Two-Level Agent initialized")
    
    # ======================= 主循环 =======================
    all_rewards = []
    all_env_rewards = []
    all_format_bonuses = []
    all_feasibility_bonuses = []
    all_invalid_actions = []
    
    for test_idx in range(test_times):
        logging.info(f"\n{'='*60}")
        logging.info(f"Start test {test_idx + 1}/{test_times}")
        logging.info(f"{'='*60}")
        start_time = time.time()
        
        # Count invalid actions in this test (reward == -1000 indicates invalid action)
        test_invalid_actions = 0
        
        # 重置环境
        obs, infos = env_manager.reset(kwargs={})
        logging.info(f"Reset completed. Batch size: {len(obs['text'])}")
        
        # 对每个环境生成策略和解决方案
        logging.info("\nGenerating strategies and solutions...")
        actions = []
        trajectories = []  # 保存轨迹数据
        
        for i in range(env_num):
            env_obs = obs['text'][i]
            logging.info(f"\n--- Environment {i} ---")
            logging.info(f"Observation:\n{env_obs[:500]}...")
            
            # 初始化轨迹数据
            trajectory = {
                'observation': env_obs,
                'image_base64': None,
                'strategy': None,
                'solution': None,
            }
            
            # 步骤 1: 生成可视化图片（如果可能）
            image_base64 = None
            try:
                # 尝试从环境管理器中创建可视化
                image_base64 = create_visualization_image(env_manager, i, sub_env)
                if image_base64:
                    logging.info(f"Env {i}: Visualization created successfully")
                    trajectory['image_base64'] = image_base64
            except Exception as e:
                logging.warning(f"Failed to create visualization for env {i}: {e}")
            
            # 步骤 2: 使用 VLM 生成策略
            logging.info(f"Env {i}: Generating strategy with VLM...")
            try:
                strategy = two_level_agent.generate_strategy(env_obs, image_base64)
                logging.info(f"Env {i} Strategy:\n{strategy}")
                trajectory['strategy'] = strategy
            except Exception as e:
                logging.error(f"Env {i} Strategy generation failed: {e}")
                strategy = "使用标准优化算法解决此问题。"
                trajectory['strategy'] = strategy
            
            # 步骤 3: 使用 LLM 基于策略生成解决方案
            logging.info(f"Env {i}: Generating solution with LLM...")
            try:
                solution = two_level_agent.generate_solution(env_obs, strategy)
                logging.info(f"Env {i} Solution:\n{solution}")
                trajectory['solution'] = solution
                actions.append(solution)
            except Exception as e:
                logging.error(f"Env {i} Solution generation failed: {e}")
                # 回退：直接使用观察生成
                try:
                    fallback_solution = llm_agent.generate(
                        f"请解决以下问题：\n{env_obs}\n\n解决方案：",
                        max_tokens=512,
                    )
                    trajectory['solution'] = fallback_solution
                    actions.append(fallback_solution)
                except:
                    trajectory['solution'] = "[]"
                    actions.append("[]")  # 最后的回退
            
            trajectories.append(trajectory)
        
        # 步骤 4: 环境执行
        logging.info("\nExecuting actions in environment...")
        termcolor.cprint(f"Actions: {[a[:100] + '...' if len(a) > 100 else a for a in actions]}", 'blue')
        obs, rewards, dones, infos = env_manager.step(actions)
        
        if infos:
            termcolor.cprint(f"Info keys: {list(infos[0].keys())}", 'yellow')
        
        # 步骤 5: 记录结果并保存轨迹
        test_rewards = []
        test_env_rewards = []
        test_format_bonuses = []
        test_feasibility_bonuses = []
        
        for i in range(env_num):
            reward = float(rewards[i])
            test_rewards.append(reward)
            all_rewards.append(reward)
            
            # 检测无效 action（env 返回 -1000 表示 action 无效）
            if reward == -1000:
                test_invalid_actions += 1
            
            # 更新轨迹数据
            if i < len(trajectories):
                trajectories[i]['reward'] = reward
                trajectories[i]['env_reward'] = reward
                trajectories[i]['format_bonus'] = 0.0
                trajectories[i]['feasibility_bonus'] = 0.0
                trajectories[i]['info'] = {}
            
            if infos and i < len(infos):
                info = infos[i]
                
                env_reward = info.get("env_reward", reward)
                test_env_rewards.append(env_reward)
                all_env_rewards.append(env_reward)
                
                format_bonus = info.get("format_bonus", 0.0)
                feasibility_bonus = info.get("feasibility_bonus", 0.0)
                test_format_bonuses.append(format_bonus)
                test_feasibility_bonuses.append(feasibility_bonus)
                all_format_bonuses.append(format_bonus)
                all_feasibility_bonuses.append(feasibility_bonus)
                
                # 更新轨迹数据
                if i < len(trajectories):
                    trajectories[i]['env_reward'] = env_reward
                    trajectories[i]['format_bonus'] = format_bonus
                    trajectories[i]['feasibility_bonus'] = feasibility_bonus
                    trajectories[i]['info'] = info
                
                logging.info(f'Env {i} Final Reward: {reward:.4f}')
                logging.info(f'Env {i} Env Reward: {env_reward:.4f}')
                if format_bonus > 0 or feasibility_bonus > 0:
                    logging.info(f'Env {i} Format Bonus: {format_bonus:.4f}')
                    logging.info(f'Env {i} Feasibility Bonus: {feasibility_bonus:.4f}')
                
                # Log solution if available
                if "route" in info:
                    logging.info(f'Env {i} Route: {info["route"]}')
                elif "routes" in info:
                    logging.info(f'Env {i} Routes: {info["routes"]}')
                elif "schedule" in info:
                    logging.info(f'Env {i} Schedule: {info["schedule"]}')
            
            # 保存轨迹
            if i < len(trajectories):
                try:
                    save_trajectory(trajectory_dir, test_idx, i, trajectories[i])
                except Exception as e:
                    logging.error(f"Failed to save trajectory for test {test_idx}, env {i}: {e}")
        
        elapsed_time = time.time() - start_time
        logging.info(f"\nTest {test_idx + 1} completed in {elapsed_time:.2f}s")
        
        # 保存测试汇总
        test_summary = {
            'test_idx': test_idx,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'env_num': env_num,
            'rewards': test_rewards,
            'env_rewards': test_env_rewards,
            'format_bonuses': test_format_bonuses,
            'feasibility_bonuses': test_feasibility_bonuses,
            'rewards_avg': float(np.mean(test_rewards)),
            'rewards_std': float(np.std(test_rewards)),
            'env_rewards_avg': float(np.mean(test_env_rewards)),
            'env_rewards_std': float(np.std(test_env_rewards)),
            'invalid_actions': test_invalid_actions,
            'invalid_rate': float(test_invalid_actions) / float(env_num) if env_num > 0 else 0.0,
        }
        
        summary_path = os.path.join(trajectory_dir, f"test_{test_idx}_summary.json")
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(test_summary, f, ensure_ascii=False, indent=2)
            logging.info(f"Saved test summary to {summary_path}")
        except Exception as e:
            logging.warning(f"Failed to save test summary: {e}")
        
        # Append test invalid count to overall list and log
        all_invalid_actions.append(test_invalid_actions)
        logging.info(f"Invalid actions (reward == -1000) in test {test_idx}: {test_invalid_actions} / {env_num} ({test_summary['invalid_rate']:.2%})")
        logging.info("=============== Single Test Summary ===============")
        logging.info(f"Rewards avg ± std: {np.mean(test_rewards):.4f} ± {np.std(test_rewards):.4f}")
        logging.info(f"Env Rewards avg ± std: {np.mean(test_env_rewards):.4f} ± {np.std(test_env_rewards):.4f}")
        if any(test_format_bonuses):
            logging.info(f"Format Bonuses avg ± std: {np.mean(test_format_bonuses):.4f} ± {np.std(test_format_bonuses):.4f}")
        if any(test_feasibility_bonuses):
            logging.info(f"Feasibility Bonuses avg ± std: {np.mean(test_feasibility_bonuses):.4f} ± {np.std(test_feasibility_bonuses):.4f}")
    
    # ======================= 最终总结 =======================
    logging.info("\n" + "="*60)
    logging.info("Final Summary")
    logging.info("="*60)
    logging.info(f"Total tests: {test_times} | Envs / test: {env_num} | Total envs: {test_times * env_num}")
    logging.info(f"Overall Rewards avg ± std: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    logging.info(f"Overall Env Rewards avg ± std: {np.mean(all_env_rewards):.4f} ± {np.std(all_env_rewards):.4f}")
    if any(all_format_bonuses):
        logging.info(f"Overall Format Bonuses avg ± std: {np.mean(all_format_bonuses):.4f} ± {np.std(all_format_bonuses):.4f}")
    if any(all_feasibility_bonuses):
        logging.info(f"Overall Feasibility Bonuses avg ± std: {np.mean(all_feasibility_bonuses):.4f} ± {np.std(all_feasibility_bonuses):.4f}")
    logging.info("="*60)
    
    # 保存最终汇总
    final_summary = {
        'timestamp': datetime.now().isoformat(),
        'env_name': env_name,
        'sub_env': sub_env,
        'test_times': test_times,
        'env_num': env_num,
        'total_envs': test_times * env_num,
        'vlm_api_url': vlm_api_url,
        'llm_api_url': llm_api_url,
        'vlm_model_name': vlm_model_name,
        'llm_model_name': llm_model_name,
        'all_rewards': all_rewards,
        'all_env_rewards': all_env_rewards,
        'all_format_bonuses': all_format_bonuses,
        'all_feasibility_bonuses': all_feasibility_bonuses,
        'rewards_avg': float(np.mean(all_rewards)),
        'rewards_std': float(np.std(all_rewards)),
        'env_rewards_avg': float(np.mean(all_env_rewards)),
        'env_rewards_std': float(np.std(all_env_rewards)),
    }
    
    # Overall invalid action stats
    total_invalid_actions = int(np.sum(all_invalid_actions)) if len(all_invalid_actions) > 0 else 0
    final_summary['total_invalid_actions'] = total_invalid_actions
    final_summary['invalid_rate'] = float(total_invalid_actions) / float(final_summary['total_envs']) if final_summary['total_envs'] > 0 else 0.0
    
    if any(all_format_bonuses):
        final_summary['format_bonuses_avg'] = float(np.mean(all_format_bonuses))
        final_summary['format_bonuses_std'] = float(np.std(all_format_bonuses))
    
    if any(all_feasibility_bonuses):
        final_summary['feasibility_bonuses_avg'] = float(np.mean(all_feasibility_bonuses))
        final_summary['feasibility_bonuses_std'] = float(np.std(all_feasibility_bonuses))
    
    final_summary_path = os.path.join(trajectory_dir, "final_summary.json")
    try:
        with open(final_summary_path, 'w', encoding='utf-8') as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)
        logging.info(f"Saved final summary to {final_summary_path}")
    except Exception as e:
        logging.warning(f"Failed to save final summary: {e}")
    
    logging.info(f"\nAll trajectories saved to: {trajectory_dir}")
    logging.info("="*60)

