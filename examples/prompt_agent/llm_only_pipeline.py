"""
LLM-only Agent Pipeline
Only the LLM is used to produce the final solution (no separate VLM strategy step).
Prompts are kept similar to `two_level_agent_pipeline.py`.
"""
import os
import sys
import time
import logging
import json
import numpy as np
from datetime import datetime
from types import SimpleNamespace
from agent_system.environments.env_manager import *
from llm_agent import LLMAgent
from two_level_agent_pipeline import build_env, create_visualization_image, save_trajectory
import termcolor


def llm_solution_prompt(observation: str, image_present: bool):
    """Prompt for LLM when LLM is the only solution agent."""
    image_note = " + [Image provided]" if image_present else ""
    prompt = f"""Role: Precision Execution Engine
Inputs: {observation}{image_note}

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
    return prompt


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default=os.environ.get("ENV_NAME", "rl4co"))
    parser.add_argument("--sub_env", default=os.environ.get("SUB_ENV", "tsp"))
    parser.add_argument("--llm_api_url", default=os.environ.get("LLM_API_URL", "http://localhost:8001/v1"))
    parser.add_argument("--api_key", default=os.environ.get("API_KEY", "token-abc123456"))
    parser.add_argument("--env_num", default=os.environ.get("ENV_NUM", "1"))
    parser.add_argument("--test_times", default=os.environ.get("TEST_TIMES", "1"))
    parser.add_argument("--llm_model_name", default=os.environ.get("LLM_MODEL_NAME", "llm"))
    args = parser.parse_args()

    env_name = args.env_name
    sub_env = args.sub_env if args.sub_env is not None else (env_name.split("/", 1)[1] if "/" in env_name else "tsp")
    llm_api_url = args.llm_api_url
    api_key = args.api_key
    env_num = int(args.env_num)
    test_times = int(args.test_times)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"logs/llm_only/rl4co_{sub_env}"
    trajectory_dir = os.path.join(log_dir, f"trajectories_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("LLM-only Agent Pipeline")
    cfg = {
        "train_batch_size": env_num,
        "val_batch_size": 1,
        "env_name": "rl4co/tsp",
        "seed": 0,
        "group_n": 1,
        "device":"cpu",
        "return_topk_options": 5,
    }
    cfg = OmegaConf.create(cfg)

    env_manager, val_env = make_envs(cfg)
    logging.info("✓ Environment built")

    llm_agent = LLMAgent(api_base_url=llm_api_url, api_key=api_key, model_name=args.llm_model_name)
    logging.info("✓ LLM Agent initialized")

    all_rewards = []
    all_invalid_actions = []
    all_final_rewards = []
    for test_idx in range(test_times):
        logging.info(f"Start test {test_idx + 1}/{test_times}")
        start_time = time.time()
        # Count invalid actions this test (reward == -1000)
        test_invalid_actions = 0

        obs, infos = env_manager.reset(kwargs={})
        logging.info(f"Reset completed. Batch size: {len(obs['text'])}")

        
        trajectories = []
        max_steps = 100
        for step in range(max_steps):
            actions = []
            for i in range(env_num):
                env_obs = obs['text'][i]
                logging.info(f"Env {i} Observation:\n{env_obs}...")
                trajectory = {'observation': env_obs, 'image_base64': None, 'solution': None}

                prompt = env_obs
                try:
                    solution = llm_agent.generate(prompt, max_tokens=2048, temperature=0.3)
                    trajectory['solution'] = solution
                    actions.append(solution)
                    logging.info(f"Env {i} Solution generated")
                except Exception as e:
                    logging.error(f"LLM generation failed for env {i}: {e}")
                    actions.append("[]")
                    trajectory['solution'] = "[]"

                trajectories.append(trajectory)

            termcolor.cprint(f"Actions: {[a + '...' if len(a) > 120 else a for a in actions]}", 'blue')
            obs, rewards, dones, infos = env_manager.step(actions)
            all_final_rewards.append(rewards) 

            if infos:
                print("\nReward Breakdown:")
                for i, info in enumerate(infos):
                    env_reward = info.get("env_reward", rewards[i])
                    if "format_reward" in info:
                        print(f"  Env {i}: Format={info.get('format_reward',0):.4f}, "
                                f"Feasibility={info.get('feasibility_reward',0):.4f}, "
                                f"Env={info.get('scaled_env_reward',0):.4f}, "
                                f"Final={rewards[i]:.4f}")
                    elif "format_bonus" in info:
                        print(f"  Env {i}: FormatBonus={info.get('format_bonus',0):.4f}, "
                                f"FeasibilityBonus={info.get('feasibility_bonus',0):.4f}, "
                                f"Env={env_reward:.4f}, Final={rewards[i]:.4f}")
                    else:
                        print(f"  Env {i}: Env={env_reward:.4f}, Final={rewards[i]:.4f}")
            
            done_flag = dones.all()
            if done_flag:
                print("\n" + "=" * 80)
                print("Episode finished!")
                print("=" * 80)
                break

        elapsed_time = time.time() - start_time
        logging.info(f"Test {test_idx + 1} completed in {elapsed_time:.2f}s")
        all_final_rewards = np.array(all_final_rewards).transpose(1,0)
        all_final_rewards = all_final_rewards.sum(1)
        logging.info(f"all final rewards: {all_final_rewards}") 



