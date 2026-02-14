import os
import sys
import torch
import numpy as np
import swanlab
import time
from types import SimpleNamespace
from functools import partial
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from examples.prompt_agent.llm_agent import LLMAgent
from examples.prompt_agent.vlm_agent import VLMAgent
from agent_system.environments.env_package.rl4co.route_envs import RouteEnvs
from agent_system.environments.env_manager import RL4COEnvironmentManager
from agent_system.environments.env_package.rl4co.projection import co_projection_selected
from rl4co.envs.routing.tdtsp.env_tw import TDTSPTWGenerator

def run_single_test(env_manager, agent, batch_size):
    """Run a single episode for the entire batch"""
    observations, infos = env_manager.reset()
    
    done = False
    step = 0
    total_rewards = np.zeros(batch_size)
    env_rewards = np.zeros(batch_size)
    
    while not done:
        step += 1
        prompts = observations['text']
        images = observations.get('image', None)
        system_template = observations.get('system_template', "")

        # Agent Inference
        actions_str = agent.batch_generate(system_template, prompts, images)

        # Environment Step
        observations, rewards, dones, step_infos = env_manager.step(actions_str)
        
        total_rewards += np.array(rewards)
        done = np.all(dones)
        
        if done:
            for i in range(batch_size):
                env_rewards[i] = step_infos[i].get("raw_env_reward", 0)
            break
            
    return env_rewards, step

def main():
    # --- Configuration ---
    num_time_steps = 37
    base_data_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    num_workers = 1 
    device = "cpu"
    
    cities = ["berlin","london", "newyork", "nairobi"]    
    node_configs = [20, 50]
    
    # --- Initialize SwanLab ---
    swanlab.init(
        project="TDTSP-Baseline",
        experiment_name="vlm-multi-city-multi-node",
        config={
            "model": "vlm",
            "cities": cities,
            "node_configs": node_configs,
            "device": device
        }
    )

    # --- Initialize Agent ---
    api_base_url = "http://localhost:8000/v1"
    api_key = "token-abc123456"
    agent = VLMAgent(
        api_key=api_key,
        api_base_url=api_base_url,
        model_name="vlm"
    )

    total_results = {}

    for num_nodes in node_configs:
        print(f"\n================ CONFIG: {num_nodes} Nodes ================")
        for city in cities:
            test_data_path = f"/root/autodl-tmp/tdtsp_dataset_random/{city}_{num_nodes}_random_test.npz"
            
            if not os.path.exists(test_data_path):
                print(f"Skipping {city}_{num_nodes}: NPZ not found at {test_data_path}")
                continue

            city_node_key = f"{city}_{num_nodes}"
            print(f"\n>>> Testing VLM: {city_node_key}")
            
            # 1. Initialize Generator
            try:
                generator = TDTSPTWGenerator(
                    data_path=test_data_path,
                    base_data_path=base_data_path,
                    matrix_path=matrix_path,
                    num_matrix_steps=num_time_steps,
                    random_sample=False,
                    phase="test"
                )
            except Exception as e:
                print(f"Error loading {city_node_key}: {e}")
                continue

            num_samples = generator.num_samples
            if num_samples == 0:
                continue
            
            # 2. Setup RL4CO Config
            config = SimpleNamespace(
                env=SimpleNamespace(
                    env_name="rl4co/tdtsp",
                    seed=42,
                    device=device,
                    rl4co=SimpleNamespace(
                        use_format_reward=True,
                        format_reward_weight=0.1,
                        format_penalty=-1.0,
                        env_reward_scale=1.0
                    )
                ),
                data=SimpleNamespace(
                    train_batch_size=num_samples,
                    val_batch_size=num_samples,
                    return_topk_options=20
                ),
                model=SimpleNamespace(model_path="dummy")
            )

            # 3. Initialize Environment
            _envs = RouteEnvs(
                env_name="tdtsp",
                seed=config.env.seed,
                env_num=1,
                group_n=1,
                device=device,
                resources_per_worker={"num_cpus": 1, "num_gpus": 0},
                return_topk_options=config.data.return_topk_options,
                env_kwargs={"generator": generator, "penalty_value": 500.0}
            )

            projection_f = partial(co_projection_selected, env_name="tdtsp")
            env_manager = RL4COEnvironmentManager(_envs, projection_f, config)

            # 4. Run Test
            start_t = time.time()
            env_rewards, total_steps = run_single_test(env_manager, agent, num_samples)
            end_t = time.time()
            
            # 5. Calculate Metrics
            # Note: RL4CO reward for TSP is usually negative cost
            # In testing mode with penalty_value > 0, violations return -inf
            is_late = np.isinf(env_rewards)
            late_count = np.sum(is_late)
            late_rate = late_count / num_samples
            
            # For avg_cost, we might want to ignore inf or handle them
            # Let's follow the baseline logic: if it's inf, it's very high cost.
            # But here we can just report the rate.
            valid_rewards = env_rewards[~is_late]
            if len(valid_rewards) > 0:
                avg_cost = -np.mean(valid_rewards)
            else:
                avg_cost = float('inf')
            
            avg_time = (end_t - start_t) / num_samples
            
            print(f"  Results for {city_node_key}:")
            print(f"    Avg Cost (Valid): {avg_cost:.2f}")
            print(f"    Late Rate: {late_rate:.2%} ({late_count}/{num_samples})")
            print(f"    Avg Time: {avg_time:.4f}s")

            # 6. Log to SwanLab
            swanlab.log({
                f"{num_nodes}nodes/{city}/avg_cost": avg_cost,
                f"{num_nodes}nodes/{city}/late_rate": late_rate,
                f"{num_nodes}nodes/{city}/avg_time": avg_time,
                f"{num_nodes}nodes/{city}/total_steps": total_steps,
            })
            
            total_results[city_node_key] = avg_cost
            
            # Cleanup env for next city to free resources
            _envs.close()

    # Final Overall Summary
    for num_nodes in node_configs:
        node_costs = [v for k, v in total_results.items() if f"_{num_nodes}" in k]
        if node_costs:
            node_avg = np.mean(node_costs)
            swanlab.log({f"overall/{num_nodes}nodes_avg_cost": node_avg})
            print(f"\n>>> Final {num_nodes} Nodes Overall Avg Cost: {node_avg:.2f}")

if __name__ == "__main__":
    main()
