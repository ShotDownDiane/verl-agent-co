import os
import sys
import torch
import numpy as np
from types import SimpleNamespace
from functools import partial
import random

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from agent_system.environments.env_package.rl4co.route_envs import RouteEnvs
from agent_system.environments.env_manager import RL4COEnvironmentManager
from agent_system.environments.env_package.rl4co.tsp_lib_test import download_and_extract_tsplib, load_tsplib_problems, TSPLibGenerator

class DummyAgent:
    def __init__(self, num_actions=10):
        self.num_actions = num_actions

    def batch_generate(self, system_prompts, texts, images, max_tokens=256, temperature=0.7):
        batch_size = len(texts)
        # Return random integer strings as actions
        return [str(random.randint(0, self.num_actions - 1)) for _ in range(batch_size)]

def simple_projection(actions, env_name=None):
    parsed = []
    valids = []
    for a in actions:
        try:
            parsed.append(int(a))
            valids.append(1)
        except:
            parsed.append(0)
            valids.append(0)
    return parsed, valids

def main():
    print("Starting TSP TSPLIB Test...")
    
    # 1. Load Data
    # Use the same data directory as tsp_lib_test.py
    data_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../agent_system/environments/env_package/rl4co")), "tsplib_data")
    
    download_url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz"
    
    print(f"Data directory: {data_dir}")
    # try:
    #     download_and_extract_tsplib(download_url, directory=data_dir)
    # except Exception as e:
    #     print(f"Download/Extract failed: {e}")
    #     # Proceed if data exists
    #     if not os.path.exists(data_dir):
    #         return

    problems = load_tsplib_problems(data_dir)
    if not problems:
        print("No TSPLIB problems found.")
        return
        
    print(f"Loaded {len(problems)} problems.")
    
    # Filter small problems for testing (e.g., <= 100 nodes to run fast)
    # Sort by dimension first (already sorted by load_tsplib_problems)
    problems = [p for p in problems if p['dimension'] <= 100]
    print(f"Selected {len(problems)} problems with dim <= 100.")
    
    if not problems:
        print("No suitable problems found.")
        return

    # 2. Setup Agent
    # We'll update num_actions dynamically, but set a default
    agent = DummyAgent(num_actions=50)

    # 3. Run a batch
    # Let's run a small batch of 2 instances
    batch_size = 2
    num_batches = 1 # Just run one batch
    
    for i in range(0, min(len(problems), batch_size * num_batches), batch_size):
        batch_probs = problems[i:i+batch_size]
        current_bs = len(batch_probs)
        
        print(f"\nProcessing Batch {i//batch_size + 1} (Size {current_bs})")
        print(f"Problems: {[p['name'] for p in batch_probs]}")
        
        # Create generators
        generators = []
        max_dim = 0
        for p in batch_probs:
            # TSPLibGenerator needs locs [N, 2]
            # Ensure it's on CPU float32
            locs = p['node_coords'].cpu().float()
            gen = TSPLibGenerator(locs=locs)
            generators.append(gen)
            if p['dimension'] > max_dim:
                max_dim = p['dimension']
            
        # Update agent num_actions
        agent.num_actions = max_dim
        
        # Config
        config = SimpleNamespace(
            env=SimpleNamespace(
                env_name="tsp",
                rl4co=SimpleNamespace(
                    use_format_reward=False,
                    format_reward_weight=0.1,
                    format_penalty=-1.0,
                    env_reward_scale=0.1
                )
            ),
            data=SimpleNamespace(
                return_topk_options=10, 
                image_max_pixels=448,
                train_batch_size=current_bs, 
                val_batch_size=current_bs
            )
        )
        
        # Worker
        # We need to ensure generators list is passed correctly
        worker = RouteEnvs(
            env_name="tsp",
            seed=42,
            env_num=current_bs,
            group_n=1,
            device="cpu",
            resources_per_worker={},
            return_topk_options=10,
            env_kwargs={
                "generator": generators,
                "image_obs": "path", 
                "generator_params": {}, 
                "synchronous": False
            }
        )
        
        env_manager = RL4COEnvironmentManager(worker, simple_projection, config)
        
        # Run Loop
        print("Resetting environment...")
        observations, infos = env_manager.reset()
        
        done = False
        step = 0
        max_steps = max_dim + 5 # Give some buffer
        
        while not done and step < 20: # Limit to 20 steps for quick test
            # Agent act
            prompts = observations['text']
            
            # Print first prompt length to verify
            # print(f"Prompt 0 length: {len(prompts[0])}")
            
            actions = agent.batch_generate(None, prompts, None)
            print(f"Step {step} Actions: {actions}")
            
            observations, rewards, dones, infos = env_manager.step(actions)
            done = np.all(dones)
            step += 1
            
            # Check rewards
            # print(f"Step {step} Rewards: {rewards}")
            
        print("Batch Done.")
        
        # Print results
        for k, info in enumerate(infos):
             name = batch_probs[k]['name']
             reward = info.get('sum_env_reward', 0)
             print(f"Instance {k} ({name}) Final Reward: {reward}")
             
        # Cleanup
        del worker
        del env_manager

if __name__ == "__main__":
    main()
