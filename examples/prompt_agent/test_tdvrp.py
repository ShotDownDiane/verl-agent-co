import os
import sys
import torch
import numpy as np
import swanlab
import time
from types import SimpleNamespace
from functools import partial

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_system/environments/env_package/rl4co")))
from examples.prompt_agent.llm_agent import LLMAgent
from examples.prompt_agent.vlm_agent import VLMAgent
from agent_system.environments.env_package.rl4co.route_envs import RouteEnvs
from agent_system.environments.env_manager import RL4COEnvironmentManager
from agent_system.environments.env_package.rl4co.projection import co_projection_selected
from rl4co.envs.routing.tdvrp.generator import TDVRPGenerator
from tqdm import trange

class MockAgent:
    """A mock agent that always picks Option A for testing purposes."""
    def batch_generate(self, system_prompts, texts, images=None, **kwargs):
        # Always pick Option A
        return ["<OBS>xxxx</OBS><Thought> Picking Option A to test the environment loop. </Thought>\n<Decision> \\boxed{A} </Decision>"] * len(texts)

def run_tdvrp_test():
    print("="*60)
    print("Starting TDVRP (Time-Dependent VRP) Environment Test")
    print("="*60)

    # --- Configuration ---
    cities = ["berlin"]#, "nairobi", "new", "madrid"]
    node_configs = [20, 50]
    base_data_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    
    # Initialize SwanLab
    swanlab.init(
        project="TDVRP-Agent-Test",
        experiment_name="vlm-agent-test",
        config={
            "cities": cities,
            "node_configs": node_configs,
        }
    )

    # Agent Configuration
    api_base_url = "http://localhost:8000/v1"
    api_key = "token-abc123456"
    # agent = VLMAgent(
    #     api_key=api_key,
    #     api_base_url=api_base_url,
    #     model_name="vlm"
    # )
    agent = MockAgent() # Uncomment for testing without API

    for num_nodes in node_configs:
        print(f"\n{'='*20} CONFIG: {num_nodes} Nodes {'='*20}")
        for city in cities:
            test_data_path = f"/root/autodl-tmp/tdtsp_dataset_random/{city}_{num_nodes}_random_test.npz"
            
            if not os.path.exists(test_data_path):
                print(f"Skipping {city}_{num_nodes}: NPZ not found at {test_data_path}")
                continue

            print(f"\n>>> Testing: {city}_{num_nodes}")
            start_time = time.time()
            
            try:
                # 1. Detect unique base files
                num_samples = 500
                
                all_costs = []
                all_lates = []

                if num_samples == 0:
                    continue
                
                # 2. Setup Environment Configuration
                batch_size = 10 # 根据需要调整 batch size
                config = SimpleNamespace(
                    env=SimpleNamespace(
                        env_name="rl4co/tdvrp",
                        seed=42,
                        device="cpu",
                        rl4co=SimpleNamespace(
                            use_format_reward=False,
                            format_reward_weight=0.1,
                            format_penalty=-1.0,
                            env_reward_scale=1.0
                        )
                    ),
                    data=SimpleNamespace(
                        train_batch_size=batch_size,
                        val_batch_size=1,
                        return_topk_options=20
                    ),
                    model=SimpleNamespace(
                        model_path="dummy"
                    )
                )

                
                _envs = RouteEnvs(
                    env_name="tdvrp",
                    seed=config.env.seed,
                    env_num=batch_size,
                    group_n=1,
                    device=config.env.device,
                    resources_per_worker={"num_cpus": 1, "num_gpus": 0},
                    return_topk_options=config.data.return_topk_options,
                    env_kwargs={
                        "data_path": test_data_path,
                        "base_data_path": base_data_path,
                        "matrix_path": matrix_path,
                        "penalty_value": 0.0,
                        "synchronous": False # 关键：不要同步，因为每个 worker 需要测试不同的实例
                    }
                )

                projection_f = partial(
                    co_projection_selected,
                    env_name="tdvrp",
                )

                env_manager = RL4COEnvironmentManager(_envs, projection_f, config)

                for current_idx in trange(0, num_samples-1, config.data.train_batch_size):
                    # 4. Execution Loop
                    observations, infos = env_manager.reset()
                    done = False
                    step = 0
                    
                    while not done:
                        step += 1
                        prompts = observations['text']
                        images = observations.get('image', None)
                        system_template = observations.get('system_template', "")

                        # Agent Inference
                        actions_str = agent.batch_generate(system_template, prompts, images)
                        
                        # Environment Step
                        observations, rewards, dones, step_infos = env_manager.step(actions_str)
                        done = np.all(dones)

                    # 5. Collect results for this sub-group
                    for i in range(config.data.train_batch_size):
                        # raw_env_reward is -makespan when penalty_value=0
                        cost = -step_infos[i].get("raw_env_reward", 0)
                        violations = step_infos[i].get("violations", 0)
                        was_late = 1 if violations > 0 else 0
        
                        all_costs.append(cost)
                        all_lates.append(was_late)
                    

                # 6. Calculate and Report Metrics for the city_num_nodes
                if all_costs:
                    elapsed_time = time.time() - start_time
                    avg_cost = np.mean(all_costs) * 3600
                    late_rate = sum(all_lates) / len(all_lates)
                    
                    print(f"  Results for {city}_{num_nodes}:")
                    print(f"    Avg Cost: {avg_cost:.4f}")
                    print(f"    Late Rate: {late_rate:.2%} ({sum(all_lates)}/{len(all_lates)})")
                    print(f"    Execution Time: {elapsed_time:.2f}s")
                    
                    swanlab.log({
                        f"{num_nodes}nodes/{city}/avg_cost": avg_cost,
                        f"{num_nodes}nodes/{city}/late_rate": late_rate,
                        f"{num_nodes}nodes/{city}/execution_time": elapsed_time,
                    })

            except Exception as e:
                print(f"Error testing {city}_{num_nodes}: {e}")
                import traceback
                traceback.print_exc()
                continue

if __name__ == "__main__":
    run_tdvrp_test()
