import os
import sys
import pickle
import torch
import numpy as np
from types import SimpleNamespace

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import random

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

from functools import partial
from agent_system.environments.env_package.rl4co.graph_env import GraphWorker, GraphEnvs
from agent_system.environments.env_manager import RL4COEnvironmentManager
from agent_system.environments.env_package.rl4co.projection import co_projection_selected

class LoadedDataGenerator:
    def __init__(self, data_list, device="cpu"):
        self.data_list = data_list
        self.idx = 0
        self.device = device
        self.min_loc = 0.0
        self.max_loc = 1.0
        
        self.num_facility = 0
        self.num_facilities_to_select = 0
        self.num_demand = 0
        self.num_terminals = 0
        
        if len(data_list) > 0:
            first_item = data_list[0]
            if 'td' in first_item:
                td = first_item['td']
                # Try to infer num_loc
                if 'loc' in td.keys():
                    self.num_loc = td['loc'].shape[-2]
                elif 'locs' in td.keys():
                    self.num_loc = td['locs'].shape[-2]
                elif 'orig_distances' in td.keys():
                     self.num_loc = td['orig_distances'].shape[-1]
                else:
                    self.num_loc = 0
                
                if 'to_choose' in td.keys():
                     val = td['to_choose']
                     self.num_facilities_to_select = val.item() if val.numel() == 1 else val[0].item()
                
            else:
                self.num_loc = 0
        else:
            self.num_loc = 0

        self.capacity = 1.0 
        self.vehicle_capacity = 1.0 
        self.min_demand = 0.0
        self.max_demand = 1.0

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
            
            if len(td.batch_size) > 0:
                b = td.batch_size[0]
            else:
                b = 1
                
            collected.append(td)
            current_count += b
            self.idx += 1
            
        # Check maximum dimensions for padding
        max_num_loc = 0
        for td in collected:
             if 'num_loc' in td.keys():
                 max_num_loc = max(max_num_loc, int(td['num_loc'].item()))
        
        for td in collected:
            curr_N = int(td['num_loc'].item()) if 'num_loc' in td.keys() else max_num_loc
            pad_len = max_num_loc - curr_N
            
            if pad_len > 0:
                for key, val in td.items():
                    if key == 'locs': # [1, N, 2]
                         td[key] = torch.nn.functional.pad(val, (0, 0, 0, pad_len), value=0)
                    elif key == 'distances': # [1, N]
                         # Initialize padded distances to infinity? 
                         # No, padded nodes shouldn't matter if we mask them.
                         # But FLPEnv might select them? No, action mask handles it?
                         # FLPEnv doesn't have mask in td initially.
                         # We should probably mask padded nodes in action_mask if it existed.
                         # But it doesn't exist yet.
                         # Let's just pad with inf for distances to be safe?
                         # Or 0. 
                         td[key] = torch.nn.functional.pad(val, (0, pad_len), value=float('inf'))
                    elif key == 'orig_distances': # [1, N, N]
                         td[key] = torch.nn.functional.pad(val, (0, pad_len, 0, pad_len), value=0)
                    # Handle other potential keys if they depend on N


        full_td = torch.cat(collected, dim=0)
        return full_td[:batch_size]

from examples.prompt_agent.load_orlib import parse_pmed

def load_opt_values():
    path = "/root/autodl-tmp/or-library/pmedopt.txt"
    opts = {}
    if os.path.exists(path):
        with open(path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2 and parts[0].startswith('pmed'):
                    try:
                        opts[parts[0]] = float(parts[1])
                    except:
                        pass
    return opts

def load_data():
    base_path = "/root/autodl-tmp/or-library"
    opts = load_opt_values()
    
    flp_data = []
    
    # Load first 5 instances
    for i in range(1, 6):
        filename = f"pmed{i}.txt"
        path = os.path.join(base_path, filename)
        if os.path.exists(path):
            print(f"Loading {filename}...")
            # parse_pmed returns a list of TensorDicts (usually 1)
            td_list = parse_pmed(path)
            
            # Associate opt value
            key = f"pmed{i}"
            opt_val = opts.get(key, None)
            
            for td in td_list:
                flp_data.append({'td': td, 'obj': opt_val, 'name': key})
        else:
            print(f"Warning: {filename} not found.")
            
    return {'flp': flp_data}, None

def run_agent_loop(env_manager, agent, solution_tour=None, env_name="flp", instance_idx=0, heuristic_objs=None):
    # 1. Reset Environment (Batch)
    observations, infos = env_manager.reset()
    
    trajectory = []
    obs_list = []
    
    metrics = {}
    os.makedirs(f"debug_images/{env_name}", exist_ok=True)
    
    # 2. Loop
    i = 0
    while True:
        prompts = observations['text']
        system_template = observations['system_template']
        images = observations.get('image', None)
        
        print(f"\n--- Step {i+1} (Batch Size: {len(prompts)}) ---")
        
        # 3. Agent Inference (Batch)
        batch_images = None
        if images is not None and len(images) > 0:
            batch_images = []
            from PIL import Image as PILImage
            for img_data in images:
                if isinstance(img_data, np.ndarray):
                    batch_images.append(PILImage.fromarray(img_data.astype('uint8')))
                elif isinstance(img_data, str) and os.path.exists(img_data):
                    batch_images.append(img_data)
                else:
                    batch_images.append(None)

        actions_str_raw = agent.batch_generate(
            system_prompts=system_template,
            texts=prompts,
            images=batch_images,
            max_tokens=256, 
            temperature=0.7
        )
        actions_str = [resp.strip() for resp in actions_str_raw]
            
        print(f"Action (first in batch): {actions_str[0]}")
        trajectory.append(actions_str)

        # 4. Environment Step (Batch)
        next_observations, rewards, dones, infos = env_manager.step(actions_str)
        
        if np.all(dones):
            print("All environments done.")
            
            if heuristic_objs is not None:
                # For FLP, reward is negative distance.
                # Agent obj = -reward
                
                batch_gaps = []
                batch_agent_objs = []

                print("\n" + "-"*50)
                print(f"Evaluation Results (Batch starting at Instance {instance_idx}):")
                print(f"{'Idx':<10} | {'Optimal':<10} | {'Agent':<10} | {'Gap (%)':<10}")
                print("-" * 50)

                for k, info in enumerate(infos):
                    h_obj = heuristic_objs[k] if k < len(heuristic_objs) else None
                    
                    agent_reward = info.get('sum_env_reward', 0.0)
                    agent_obj = -agent_reward 
                    
                    # Gap: (Agent - Opt) / Opt * 100
                    gap = float('inf')
                    if h_obj is not None and h_obj != 0:
                        gap = (agent_obj - h_obj) / h_obj * 100
                    
                    batch_gaps.append(gap)
                    batch_agent_objs.append(agent_obj)
                    
                    h_str = f"{h_obj:.2f}" if h_obj is not None else "N/A"
                    print(f"{instance_idx+k:<10} | {h_str:<10} | {agent_obj:<10.2f} | {gap:<10.2f}")

                print("-" * 50 + "\n")
                
                metrics['gap'] = sum([g for g in batch_gaps if g != float('inf')]) / len(batch_gaps) if batch_gaps else 0
                metrics['all_gaps'] = batch_gaps
            
            break
            
        observations = next_observations
        i += 1

    return metrics

def main():
    graph_data, routing_data = load_data()
    
    if 'flp' not in graph_data or not graph_data['flp']:
        print("No FLP data found.")
        return

    agent = DummyAgent(num_actions=100) # Placeholder

    print("\n" + "="*50)
    print("1. Batch Execution (FLP) with RL4COEnvironmentManager (Dummy Agent)")
    print("="*50)

    flp_data = graph_data['flp']
    n = len(flp_data)
    
    gaps = []
    batch_size = 5
    
    for i in range(0, n, batch_size):
        current_bs = min(batch_size, n - i)
        batch_data = flp_data[i : i + current_bs]
        
        generators = [LoadedDataGenerator([item]) for item in batch_data]
        num_loc = generators[0].num_loc
        
        agent.num_actions = num_loc

        heuristic_objs = [item['obj'] for item in batch_data]
        
        config = SimpleNamespace(
            env=SimpleNamespace(
                env_name="flp",
                rl4co=SimpleNamespace(
                    use_format_reward=True,
                    format_reward_weight=0.1,
                    format_penalty=-1.0,
                    env_reward_scale=1.0 # FLP distances are large, maybe scale? But gap calc handles it.
                )
            ),
            data=SimpleNamespace(
                return_topk_options=10, 
                image_max_pixels=448,
                train_batch_size=current_bs, 
                val_batch_size=current_bs
            )
        )

        worker = GraphEnvs(
            env_name="flp",
            seed=42,
            env_num=current_bs,
            group_n=1,
            device="cpu",
            resources_per_worker={},
            return_topk_options=10,
            env_kwargs={
                "generator": generators,
                "image_obs": "path",
                "generator_params": {"num_loc": num_loc},
                "synchronous": False
            }
        )
        
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

        projection_f = partial(simple_projection, env_name="flp")
        
        env_manager = RL4COEnvironmentManager(worker, projection_f, config)
            
        metrics = run_agent_loop(env_manager, agent, solution_tour=None, env_name="flp", instance_idx=i, heuristic_objs=heuristic_objs)
        
        if 'all_gaps' in metrics:
            gaps.extend(metrics['all_gaps'])

    if gaps:
        valid_gaps = [g for g in gaps if g != float('inf')]
        if valid_gaps:
            avg_gap = sum(valid_gaps) / len(valid_gaps)
            print("\n" + "="*50)
            print(f"Final Average Gap: {avg_gap:.2f}%")
            print("="*50 + "\n")

if __name__ == "__main__":
    main()
