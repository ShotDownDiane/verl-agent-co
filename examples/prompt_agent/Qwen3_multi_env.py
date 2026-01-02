
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

from agent_system.environments.env_manager import make_envs, RouteEnvironmentManager, GraphEnvironmentManager
from agent_system.environments.env_package.rl4co.base_env import BaseCOEnvs
from examples.prompt_agent.llm_agent import LLMAgent
from rl4co.utils.ops import get_distance_matrix

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
        start_idx = self.idx
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

class MixedEnvs(BaseCOEnvs):
    """
    A wrapper that holds a heterogeneous list of Ray actors (workers).
    """
    def __init__(self, actors, env_num):
        self.actors = actors
        self.env_num = env_num
        self.return_topk_options = True # Assume true for now
        # We don't call super().__init__ because it creates actors. 
        # We just reuse the interface.

    def reset(self, kwargs=None):
        # Broadcast reset to all actors
        # We assume kwargs is a list of kwargs or shared.
        # For simplicity, pass empty kwargs or handle per-actor if needed.
        futures = [actor.reset.remote() for actor in self.actors]
        results = ray.get(futures)
        
        # Combine results
        # results is list of (td, info)
        td_list = [r[0] for r in results]
        info_list = [r[1] for r in results]
        
        # Stack TDs
        batch_td = torch.cat(td_list, dim=0)
        # Flatten infos (if they are lists)
        flat_infos = []
        for info in info_list:
            if isinstance(info, list):
                flat_infos.extend(info)
            else:
                flat_infos.append(info)
                
        return batch_td, flat_infos

    def step(self, actions):
        # actions is a list of actions or tensor
        # We need to split actions for each actor
        # Assume each actor handles 1 env for simplicity in Mixed mode
        
        futures = []
        for i, actor in enumerate(self.actors):
            # action for this environment
            # If actions is tensor (B,), take i-th
            if isinstance(actions, torch.Tensor):
                act = actions[i].unsqueeze(0) # Keep batch dim 1
            elif isinstance(actions, list):
                act = [actions[i]]
            else:
                act = actions[i]
            
            futures.append(actor.step.remote(act))
            
        results = ray.get(futures)
        
        # results is list of (next_td, reward, done, info)
        next_tds = [r[0] for r in results]
        rewards = [r[1] for r in results]
        dones = [r[2] for r in results]
        infos = [r[3] for r in results]
        
        batch_next_td = torch.cat(next_tds, dim=0)
        batch_rewards = torch.cat(rewards, dim=0)
        batch_dones = torch.cat(dones, dim=0)
        
        flat_infos = []
        for info in infos:
            if isinstance(info, list):
                flat_infos.extend(info)
            else:
                flat_infos.append(info)
                
        return batch_next_td, batch_rewards, batch_dones, flat_infos


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
        print(f"Obs (Env 0): {obs['text'][0][:200]}...")
        
        # Agent inference
        # In a real scenario, we would batch prompt the agent.
        # Here we mock it or loop.
        actions = []
        for idx, text in enumerate(obs['text']):
            # Construct prompt
            prompt = text
            # Simple agent call (mocked or real)
            # response = agent.generate(prompt)
            # Parse response to get action. 
            # For this test, we just pick option 0 (first valid action)
            # We assume the prompt asks to select an option.
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

    graph_data, routing_data = load_data()
    
    # Configuration
    api_key = "sk-saxqqtlyqrpconxlgcslqhrgvhwnfmuhnimiyzfvpcxqgmkh" # Use key from flp_real_world_test.py
    agent = LLMAgent(
        api_key=api_key,
        api_base_url="https://api.siliconflow.cn/v1",
        model_name="Qwen/Qwen2.5-7B-Instruct" # Use a smaller model for testing if Qwen3 not available or use Qwen3 name
    )

    print("\n" + "="*50)
    print("1. Single Worker Execution (TSP)")
    print("="*50)
    
    tsp_data = routing_data['tsp']
    generator = LoadedDataGenerator(tsp_data)
    
    cfg = SimpleNamespace(
        train_batch_size=1,
        val_batch_size=1,
        env_name="rl4co/tsp",
        seed=42,
        group_n=1,
        device="cpu",
        return_topk_options=5,
        generator_params={"_generator_obj": generator} # Pass generator
    )
    
    envs, _ = make_envs(cfg)
    run_agent_loop(envs, agent, steps=3)
    
    
    print("\n" + "="*50)
    print("2. Batch Execution (CVRP - 2 instances)")
    print("="*50)
    
    cvrp_data = routing_data['cvrp']
    generator_cvrp = LoadedDataGenerator(cvrp_data)
    
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

    
    print("\n" + "="*50)
    print("3. Mixed Tasks Batch Inference (TSP + CVRP)")
    print("="*50)
    
    # We create two sets of envs and combine their actors
    # 1 TSP
    cfg_tsp = SimpleNamespace(
        train_batch_size=1,
        val_batch_size=0,
        env_name="rl4co/tsp",
        seed=100,
        group_n=1,
        device="cpu",
        return_topk_options=5,
        generator_params={"_generator_obj": LoadedDataGenerator(tsp_data)}
    )
    envs_tsp, _ = make_envs(cfg_tsp)
    
    # 1 CVRP
    cfg_cvrp = SimpleNamespace(
        train_batch_size=1,
        val_batch_size=0,
        env_name="rl4co/cvrp",
        seed=200,
        group_n=1,
        device="cpu",
        return_topk_options=5,
        generator_params={"_generator_obj": LoadedDataGenerator(cvrp_data)}
    )
    envs_cvrp, _ = make_envs(cfg_cvrp)
    
    # Extract actors
    # envs_tsp.envs is RouteEnvs
    actors = envs_tsp.envs.actors + envs_cvrp.envs.actors
    
    # Create MixedEnvs
    mixed_envs_base = MixedEnvs(actors, env_num=len(actors))
    
    # Wrap in EnvironmentManager
    # We need a projection function. Since both are Routing, we can use the one from TSP (it's generic co_projection_selected)
    projection_f = envs_tsp.projection_f
    
    # Create Manager
    # We reuse the config from TSP but it doesn't matter much as long as it has basic fields
    mixed_manager = RouteEnvironmentManager(mixed_envs_base, projection_f, cfg_tsp)
    
    run_agent_loop(mixed_manager, agent, steps=3)

if __name__ == "__main__":
    main()
