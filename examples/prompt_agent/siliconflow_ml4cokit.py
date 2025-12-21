import os
import numpy as np
import time
import logging
from datetime import datetime
from agent_system.environments.env_manager import *
from openai import OpenAI
import sys
import termcolor
from types import SimpleNamespace
from functools import partial


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
            use_format_reward=True,
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


class Agent:
    def __init__(self, model_name="gpt-4o", api_key=None):
        self.model_name = model_name
        # 优先使用传入的 api_key，其次使用环境变量，最后使用默认值
        if api_key:
            final_api_key = api_key
            key_source = "parameter"
        elif os.environ.get("SILICONFLOW_API_KEY"):
            final_api_key = os.environ.get("SILICONFLOW_API_KEY")
            key_source = "environment variable"
        else:
            final_api_key = "sk-saxqqtlyqrpconxlgcslqhrgvhwnfmuhnimiyzfvpcxqgmkh"
            key_source = "hardcoded default"
        
        logging.info(f"Using API key from: {key_source} (key: {final_api_key[:10]}...{final_api_key[-10:]})")
        
        self.client = OpenAI(
            api_key=final_api_key,
            base_url="https://api.siliconflow.cn/v1/",
            max_retries=3,
            timeout=60.0,  # 设置超时时间为60秒
        )

    def get_action_from_gpt(self, obs):
        import time
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": obs}],
                temperature=1,
                max_tokens=128,
                n=1,
                stop=None,
            )
            action = response.choices[0].message.content.strip()
            elapsed_time = time.time() - start_time
            logging.info(f"API call completed in {elapsed_time:.2f}s")
            return action
        except Exception as e:
            elapsed_time = time.time() - start_time
            logging.error(f"API call failed after {elapsed_time:.2f}s: {e}")
            raise



if __name__ == "__main__":
    # -------- Parameters ----------
    # Default: ml4co-kit/tsp, can be changed via command line or environment variable
    env_name = "ml4co-kit/tsp"
    sub_env = "tsp"
    model_name = "deepseek-ai/DeepSeek-V3.2"
    model_name_short = "deepseekv3.2"
    
    # -------- logging ----------
    log_dir = f"logs/{model_name_short}/ml4co-kit_{sub_env}"
    os.makedirs(log_dir, exist_ok=True)
    log_fp = os.path.join(
        log_dir, f"run_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_fp, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # -------- Parameters ----------
    env_num = 3
    test_times = 1

    # -------- Environment and agent setup ----------
    logging.info(f"Building environment: {env_name}, sub_env: {sub_env}")
    env_manager = build_env(env_name, env_num, sub_env=sub_env)
    
    # Support API key from command line (4th argument) or environment variable
    api_key = sys.argv[3] if len(sys.argv) > 3 else None
    agent = Agent(model_name=model_name, api_key=api_key)  # Pass model name as command line argument

    # ======================= Main Loop =======================
    all_rewards = []
    all_env_rewards = []
    all_format_bonuses = []
    all_feasibility_bonuses = []
    
    for test_idx in range(test_times):
        logging.info(f"\n========== Start test {test_idx} ==========")
        start_time = time.time()

        obs, infos = env_manager.reset(kwargs={})
        logging.info(f"Reset completed. Batch size: {len(obs['text'])}")

        # ml4co-kit is one-shot mode, so we only need one step
        logging.info("Getting actions from agent...")
        actions = []
        
        for i in range(env_num):
            env_obs = obs['text'][i]
            logging.info(f"\nEnv {i} Observation:\n{env_obs}")
            response = agent.get_action_from_gpt(env_obs)
            logging.info(f"Env {i} Response: {response}")
            actions.append(response)

        # --- Environment stepping ---
        termcolor.cprint(f"Actions: {[a[:100] + '...' if len(a) > 100 else a for a in actions]}", 'blue')
        obs, rewards, dones, infos = env_manager.step(actions)
        
        if infos:
            termcolor.cprint(f"Info keys: {list(infos[0].keys())}", 'yellow')

        # -------- Single round results --------
        test_rewards = []
        test_env_rewards = []
        test_format_bonuses = []
        test_feasibility_bonuses = []
        
        for i in range(env_num):
            reward = float(rewards[i])
            test_rewards.append(reward)
            all_rewards.append(reward)
            
            if infos and i < len(infos):
                info = infos[i]
                
                # Log reward breakdown if available
                env_reward = info.get("env_reward", reward)
                test_env_rewards.append(env_reward)
                all_env_rewards.append(env_reward)
                
                format_bonus = info.get("format_bonus", 0.0)
                feasibility_bonus = info.get("feasibility_bonus", 0.0)
                test_format_bonuses.append(format_bonus)
                test_feasibility_bonuses.append(feasibility_bonus)
                all_format_bonuses.append(format_bonus)
                all_feasibility_bonuses.append(feasibility_bonus)
                
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

        logging.info(f"Test {test_idx} time elapsed: {time.time() - start_time:.2f}s\n")

        logging.info("=============== Single Test Summary ===============")
        logging.info(f"Rewards avg ± std: {np.mean(test_rewards):.4f} ± {np.std(test_rewards):.4f}")
        logging.info(f"Env Rewards avg ± std: {np.mean(test_env_rewards):.4f} ± {np.std(test_env_rewards):.4f}")
        if any(test_format_bonuses):
            logging.info(f"Format Bonuses avg ± std: {np.mean(test_format_bonuses):.4f} ± {np.std(test_format_bonuses):.4f}")
        if any(test_feasibility_bonuses):
            logging.info(f"Feasibility Bonuses avg ± std: {np.mean(test_feasibility_bonuses):.4f} ± {np.std(test_feasibility_bonuses):.4f}")

    # ======================= Final Summary =======================
    logging.info("\n=============== Final Summary ===============")
    logging.info(f"Total tests: {test_times} | Envs / test: {env_num} | Total envs: {env_num * test_times}")
    logging.info(f"Overall Rewards avg ± std: {np.mean(all_rewards):.4f} ± {np.std(all_rewards):.4f}")
    logging.info(f"Overall Env Rewards avg ± std: {np.mean(all_env_rewards):.4f} ± {np.std(all_env_rewards):.4f}")
    if any(all_format_bonuses):
        logging.info(f"Overall Format Bonuses avg ± std: {np.mean(all_format_bonuses):.4f} ± {np.std(all_format_bonuses):.4f}")
    if any(all_feasibility_bonuses):
        logging.info(f"Overall Feasibility Bonuses avg ± std: {np.mean(all_feasibility_bonuses):.4f} ± {np.std(all_feasibility_bonuses):.4f}")
