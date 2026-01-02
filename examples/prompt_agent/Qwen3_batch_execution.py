import os
import sys
import pickle
import torch
import numpy as np
from types import SimpleNamespace
from omegaconf import OmegaConf
import ray

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from agent_system.environments.env_manager import make_envs
from examples.prompt_agent.llm_agent import LLMAgent

class LoadedDataGenerator:
    def __init__(self, data_list, device="cpu"):
        self.data_list = data_list
        self.idx = 0
        self.device = device
        self.min_loc = 0.0
        self.max_loc = 1.0
        
        # Infer num_loc from first data item
        if len(data_list) > 0:
            first_item = data_list[0]
            if 'td' in first_item:
                td = first_item['td']
                if 'loc' in td.keys():
                    self.num_loc = td['loc'].shape[-2]
                else:
                    self.num_loc = 0
            else:
                self.num_loc = 0
        else:
            self.num_loc = 0
        self.capacity = 1.0 # Default capacity for CVRP (normalized)
        self.vehicle_capacity = 1.0 # Alias for CVRPEnv
        self.min_demand = 0.0
        self.max_demand = 1.0

    def __call__(self, batch_size):
        if isinstance(batch_size, torch.Size):
            batch_size = batch_size[0] if len(batch_size) > 0 else 1
        elif isinstance(batch_size, list):
            batch_size = batch_size[0] if len(batch_size) > 0 else 1
            
        collected = []
        current_count = 0
        
        # Loop until we have enough data
        # We cycle through the data_list if needed
        while current_count < batch_size:
            data_item = self.data_list[self.idx % len(self.data_list)]
            td = data_item['td'].clone()
            
            # Ensure device
            if self.device != "cpu":
                td = td.to(self.device)
            
            b = td.batch_size[0]
            collected.append(td)
            current_count += b
            self.idx += 1
            
        # Handle padding if needed
        if len(collected) > 1:
            # Find location key
            loc_key = None
            for k in collected[0].keys():
                if k in ['loc', 'locs', 'coords', 'coordinates']:
                    loc_key = k
                    break
            
            if loc_key is None:
                # Try to infer from shape (B, N, 2)
                for k, v in collected[0].items():
                    if isinstance(v, torch.Tensor) and v.dim() == 3 and v.shape[2] == 2:
                        loc_key = k
                        break
            
            if loc_key:
                max_loc = max([td[loc_key].shape[1] for td in collected])
                
                for i in range(len(collected)):
                    td = collected[i]
                    curr_loc = td[loc_key].shape[1]
                    
                    if curr_loc < max_loc:
                        pad_len = max_loc - curr_loc
                        
                        # Generic padding for all keys with matching dim 1
                        for key in td.keys():
                            val = td[key]
                            if isinstance(val, torch.Tensor) and val.dim() > 1 and val.shape[1] == curr_loc:
                                if val.dim() == 3: # (B, N, D)
                                    val_padded = torch.nn.functional.pad(val, (0, 0, 0, pad_len), value=0)
                                    td[key] = val_padded
                                elif val.dim() == 2: # (B, N)
                                    val_padded = torch.nn.functional.pad(val, (0, pad_len), value=0)
                                    td[key] = val_padded
                        
                        collected[i] = td
            
        # Concatenate and slice
        full_td = torch.cat(collected, dim=0)
        return full_td[:batch_size]

def load_data():
    base_path = "/root/autodl-tmp/rl4co-urban"
    with open(os.path.join(base_path, "results.pkl"), "rb") as f:
        graph_data = pickle.load(f)
    with open(os.path.join(base_path, "routing_results.pkl"), "rb") as f:
        routing_data = pickle.load(f)
    return graph_data, routing_data

def run_agent_loop(envs, agent, steps=5):
    print(f"Resetting environments...")
    obs, infos = envs.reset(kwargs={})
    
    for i in range(steps):
        print(f"\n--- Step {i+1} ---")
        # For simplicity, just print first env's text
        if len(obs['text']) > 0:
            print(f"Obs (Env 0): {obs['text'][0][:200]}...")
        
        # Agent inference
        actions = []
        for idx, text in enumerate(obs['text']):
            # In a real scenario, use agent.generate(text)
            # For this test, we just pick option 0
            actions.append("0") 
        
        print(f"Actions: {actions}")
        obs, rewards, dones, infos = envs.step(actions)
        print(f"Rewards: {rewards}")
        
        if dones.all():
            print("All environments done.")
            break

def main():
    # Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    _, routing_data = load_data()
    
    # Configuration
    api_key = "sk-saxqqtlyqrpconxlgcslqhrgvhwnfmuhnimiyzfvpcxqgmkh"
    agent = LLMAgent(
        api_key=api_key,
        api_base_url="https://api.siliconflow.cn/v1",
        model_name="Qwen/Qwen2.5-7B-Instruct"
    )

    print("\n" + "="*50)
    print("2. Batch Execution (CVRP - 2 instances)")
    print("="*50)
    
    cvrp_data = routing_data['cvrp']
    generator_cvrp = LoadedDataGenerator(cvrp_data)
    
    # Batch execution configuration
    # train_batch_size=2 means we want 2 environments in total
    # group_n=2 means we distribute them across 2 actors (if possible)
    cfg_batch = SimpleNamespace(
        train_batch_size=2,
        val_batch_size=1,
        env_name="rl4co/cvrp",
        seed=42,
        group_n=2, 
        device="cpu",
        return_topk_options=5,
        generator_params={"_generator_obj": generator_cvrp}
    )
    
    envs_batch, _ = make_envs(cfg_batch)
    run_agent_loop(envs_batch, agent, steps=3)

if __name__ == "__main__":
    main()
