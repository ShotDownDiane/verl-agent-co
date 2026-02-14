
import os
import sys
import torch
import numpy as np
from types import SimpleNamespace
from functools import partial
import random

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from agent_system.environments.env_package.rl4co.graph_env import GraphWorker, GraphEnvs
from agent_system.environments.env_manager import RL4COEnvironmentManager
from examples.prompt_agent.load_orlib import parse_pmed_mclp

# Reuse LoadedDataGenerator from test_mclp.py or redefine it here
# Redefining for self-contained script
class LoadedDataGenerator:
    def __init__(self, data_list, device="cpu"):
        self.data_list = data_list
        self.idx = 0
        self.device = device
        
        self.num_facility = 0
        self.num_facilities_to_select = 0
        self.num_demand = 0
        
        if len(data_list) > 0:
            td = data_list[0]['td']
            if 'demand_locs' in td.keys():
                self.num_loc = td['demand_locs'].shape[-2]
                self.num_demand = td['demand_locs'].shape[-2]
            elif 'locs' in td.keys():
                self.num_loc = td['locs'].shape[-2]
            else:
                self.num_loc = 0
                
            if 'facility_locs' in td.keys():
                self.num_facility = td['facility_locs'].shape[-2]
            
            if 'num_facilities_to_select' in td.keys():
                val = td['num_facilities_to_select']
                if hasattr(val, 'item'):
                    self.num_facilities_to_select = val.item() if val.numel() == 1 else val[0].item()
                else:
                    self.num_facilities_to_select = int(val)
        else:
            self.num_loc = 0

    def __call__(self, batch_size):
        if isinstance(batch_size, torch.Size):
            batch_size = batch_size[0] if len(batch_size) > 0 else 1
        elif isinstance(batch_size, list):
            batch_size = batch_size[0] if len(batch_size) > 0 else 1
            
        collected = []
        current_count = 0
        
        while current_count < batch_size:
            data_item = self.data_list[self.idx % len(self.data_list)]
            td = data_item['td'].clone()
            
            if self.device != "cpu":
                td = td.to(self.device)
            
            b = td.batch_size[0]
            collected.append(td)
            current_count += b
            self.idx += 1
            
        full_td = torch.cat(collected, dim=0)
        return full_td[:batch_size]

class DummyAgent:
    def __init__(self, num_actions):
        self.num_actions = num_actions

    def batch_generate(self, system_prompts, texts, images=None, max_tokens=256, temperature=0.7):
        # Generate random valid actions
        batch_size = len(texts)
        actions = []
        for _ in range(batch_size):
            # Randomly select a node index
            action = random.randint(0, self.num_actions - 1)
            actions.append(str(action))
        return actions

def simple_projection(actions, env_name=None):
    parsed = []
    valids = []
    for a in actions:
        try:
            val = int(a)
            parsed.append(val)
            valids.append(1)
        except:
            parsed.append(0)
            valids.append(0)
    return parsed, valids

def run_agent_loop(env_manager, agent, env_name="mclp"):
    # 1. Reset Environment
    observations, infos = env_manager.reset()
    
    i = 0
    total_reward = 0
    
    while True:
        prompts = observations['text']
        print(f"\n--- Step {i+1} (Batch Size: {len(prompts)}) ---")
        
        # Agent Inference
        actions_str = agent.batch_generate(
            system_prompts=observations['system_template'],
            texts=prompts
        )
        print(f"Action (first in batch): {actions_str[0]}")

        # Environment Step
        next_observations, rewards, dones, infos = env_manager.step(actions_str)
        
        if np.all(dones):
            print("All environments done.")
            for k, info in enumerate(infos):
                agent_reward = info.get('sum_env_reward', 0.0)
                print(f"Instance {k} Reward (Coverage): {agent_reward:.4f}")
            break
            
        observations = next_observations
        i += 1
        if i > 100:
            print("Max steps reached.")
            break

def main():
    # Load OR-Lib data
    data_dir = "/root/autodl-tmp/or-library"
    pmed_file = os.path.join(data_dir, "pmed1.txt")
    
    if not os.path.exists(pmed_file):
        print(f"Error: {pmed_file} not found.")
        return

    print(f"Loading {pmed_file} for MCLP...")
    mclp_data = parse_pmed_mclp(pmed_file)
    # Wrap in dict structure expected by LoadedDataGenerator
    # parse_pmed_mclp returns a list of TensorDicts, we wrap them
    formatted_data = [{'td': td} for td in mclp_data]
    
    num_loc = formatted_data[0]['td']['demand_locs'].shape[1]
    print(f"Num Locations: {num_loc}")

    # Use DummyAgent
    agent = DummyAgent(num_actions=num_loc)

    # Config
    batch_size = 2 # Test with small batch
    
    generators = [LoadedDataGenerator(formatted_data) for _ in range(batch_size)]
    
    config = SimpleNamespace(
        env=SimpleNamespace(
            env_name="mclp",
            rl4co=SimpleNamespace(
                use_format_reward=True,
                format_reward_weight=0.1,
                format_penalty=-1.0,
                env_reward_scale=1.0 # Use 1.0 to see raw coverage
            )
        ),
        data=SimpleNamespace(
            return_topk_options=10, 
            image_max_pixels=448,
            train_batch_size=batch_size, 
            val_batch_size=batch_size
        )
    )

    worker = GraphEnvs(
        env_name="mclp",
        seed=42,
        env_num=batch_size,
        group_n=1,
        device="cpu",
        resources_per_worker={}, # Empty dict for local execution without Ray overhead if possible, or just default
        return_topk_options=10,
        env_kwargs={
            "generator": generators,
            "image_obs": False, # No images for OR-Lib data
            "generator_params": {"num_loc": num_loc},
            "synchronous": False
        }
    )
    
    projection_f = partial(simple_projection, env_name="mclp")
    
    env_manager = RL4COEnvironmentManager(worker, projection_f, config)
    
    run_agent_loop(env_manager, agent)

if __name__ == "__main__":
    main()
