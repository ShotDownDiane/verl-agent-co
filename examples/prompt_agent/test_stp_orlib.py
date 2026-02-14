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
        # For STP, actions are node indices
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
                if 'loc' in td.keys():
                    self.num_loc = td['loc'].shape[-2]
                elif 'locs' in td.keys():
                    self.num_loc = td['locs'].shape[-2]
                else:
                    self.num_loc = 0
                
                if 'num_facility' in td.keys():
                    val = td['num_facility']
                    if hasattr(val, 'item'):
                        self.num_facility = val.item() if val.numel() == 1 else val[0].item()
                    else:
                        self.num_facility = val
                else:
                     if 'facility_locs' in td.keys():
                         self.num_facility = td['facility_locs'].shape[-2]
                     else:
                         self.num_facility = 0
                
                if 'num_facilities_to_select' in td.keys():
                    val = td['num_facilities_to_select']
                    if hasattr(val, 'item'):
                        self.num_facilities_to_select = val.item() if val.numel() == 1 else val[0].item()
                    else:
                        self.num_facilities_to_select = val
                else:
                    self.num_facilities_to_select = 0

                if 'num_demand' in td.keys():
                     val = td['num_demand']
                     if hasattr(val, 'item'):
                         self.num_demand = val.item() if val.numel() == 1 else val[0].item()
                     else:
                         self.num_demand = val
                else:
                     if 'demand_locs' in td.keys():
                         self.num_demand = td['demand_locs'].shape[-2]
                     else:
                         self.num_demand = 0
                
                if 'num_terminals' in td.keys():
                    val = td['num_terminals']
                    if hasattr(val, 'item'):
                        self.num_terminals = val.item() if val.numel() == 1 else val[0].item()
                    else:
                        self.num_terminals = val
                else:
                    if 'terminals' in td.keys():
                         t = td['terminals']
                         if t.dim() >= 2:
                             self.num_terminals = t.shape[-1]
                         else:
                             self.num_terminals = 0
                    else:
                        self.num_terminals = 0

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
        max_dims = {}
        for td in collected:
            for key, val in td.items():
                if isinstance(val, torch.Tensor) and val.dim() >= 1:
                    check_dim = -2 if val.dim() >= 2 else -1
                    curr_size = val.shape[check_dim]
                    
                    if key not in max_dims:
                        max_dims[key] = curr_size
                    else:
                        max_dims[key] = max(max_dims[key], curr_size)

        for td in collected:
            for key, val in td.items():
                if key in max_dims and isinstance(val, torch.Tensor) and val.dim() >= 1:
                    check_dim = -2 if val.dim() >= 2 else -1
                    current_dim = val.shape[check_dim]
                    max_dim = max_dims[key]
                    pad_len = max_dim - current_dim
                    
                    if pad_len > 0:
                        # pad expects (pad_last_left, pad_last_right, ...)
                        
                        # Special handling for square matrices (adjacency, edge_weights)
                        if key in ['adjacency', 'edge_weights'] and val.dim() >= 2 and val.shape[-1] == val.shape[-2]:
                             # Pad both last two dimensions
                             # pad=(0, pad_len, 0, pad_len)
                             val_padded = torch.nn.functional.pad(val, (0, pad_len, 0, pad_len), value=0)
                        elif val.dim() >= 2:
                            # Assuming [N, 2] or [E, 2], we want to pad first dimension (N or E).
                            # pad=(0, 0, 0, pad_len)
                            val_padded = torch.nn.functional.pad(val, (0, 0, 0, pad_len), value=0)
                        else:
                            # Assuming [N], pad N.
                            # pad=(0, pad_len)
                            val_padded = torch.nn.functional.pad(val, (0, pad_len), value=0)
                            
                        td[key] = val_padded

        full_td = torch.cat(collected, dim=0)
        return full_td[:batch_size]

from examples.prompt_agent.load_orlib import parse_estein

def load_data():
    # Load OR-Lib estein data
    path = "/root/autodl-tmp/or-library/estein1.txt"
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        return None, {'stp': []}
        
    print(f"Loading data from {path}...")
    stp_data = parse_estein(path)
    # Wrap in expected structure
    graph_data = {'stp': stp_data}
    
    # No heuristic solutions available in this file
    routing_data = None 
    
    return graph_data, routing_data

def run_agent_loop(env_manager, agent, solution_tour=None, env_name="stp", instance_idx=0, heuristic_objs=None):
    # 1. Reset Environment (Batch)
    observations, infos = env_manager.reset()
    
    trajectory = []
    obs_list = []
    image_list = []
    candidates_list = []
    
    metrics = {}

    os.makedirs(f"debug_images/{env_name}", exist_ok=True)
    
    # 2. Loop
    i = 0
    while True:
        # Access batch data
        prompts = observations['text']
        system_template = observations['system_template']
        images = observations.get('image', None)
        
        # Log first item in batch
        print(f"\n--- Step {i+1} (Batch Size: {len(prompts)}) ---")
        
        obs_list.append(prompts[0])
        
        # Save step image if available
        current_img_path = ""
        if images and len(images) > 0 and images[0] is not None:
            img_data = images[0]
            if isinstance(img_data, np.ndarray):
                from PIL import Image as PILImage
                try:
                    img = PILImage.fromarray(img_data.astype('uint8'))
                    step_img_path = f"debug_images/{env_name}/inst_{instance_idx}_step_{i}.png"
                    img.save(step_img_path)
                    current_img_path = step_img_path
                except Exception as e:
                    print(f"Error saving numpy image: {e}")
            elif isinstance(img_data, str) and os.path.exists(img_data):
                current_img_path = img_data
        
        image_list.append(current_img_path)

        # 3. Agent Inference (Batch)
        # Prepare images for batch (convert numpy to PIL if needed)
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
            
            # if heuristic_objs is not None: # ALWAYS print results
            reward_scale = 1.0 # No scaling
            
            batch_gaps = []
            batch_agent_objs = []

            print("\n" + "-"*50)
            print(f"Evaluation Results (Batch starting at Instance {instance_idx}):")
            print(f"{'Idx':<5} | {'Heuristic':<10} | {'Agent':<10} | {'Gap (%)':<10}")
            print("-" * 45)

            for k, info in enumerate(infos):
                val = heuristic_objs[k] if heuristic_objs and k < len(heuristic_objs) else 0.0
                h_obj = val if val is not None else 0.0
                    
                agent_reward = info.get('sum_env_reward', 0.0)
                
                # Agent reward is negative cost
                if agent_reward != 0:
                        agent_obj = -agent_reward / reward_scale
                else:
                        agent_obj = 0.0 # Or inf if we expect non-zero cost
                
                # Gap for minimization: (Agent - Heuristic) / Heuristic
                if h_obj != 0:
                    gap = (agent_obj - h_obj) / h_obj * 100
                else:
                    gap = 0.0 # No heuristic to compare

                batch_gaps.append(gap)
                batch_agent_objs.append(agent_obj)
                
                print(f"{instance_idx+k:<5} | {h_obj:<10.4f} | {agent_obj:<10.4f} | {gap:<10.2f}")

            print("-" * 50 + "\n")
            
            # Add to global metrics
            if 'gaps' not in metrics: metrics['gaps'] = []
            metrics['gaps'].extend(batch_gaps)
            
            return metrics
            
        observations = next_observations
        i += 1

    return metrics

class DummyAgent:
    def __init__(self, num_actions=10):
        self.num_actions = num_actions

    def batch_generate(self, system_prompts, texts, images, max_tokens=256, temperature=0.7):
        import random
        batch_size = len(texts)
        # Return random integer strings
        return [str(random.randint(0, self.num_actions - 1)) for _ in range(batch_size)]

def main():
    graph_data, routing_data = load_data()
    
    if 'stp' not in graph_data:
        print("No STP data found.")
        return

    # Use DummyAgent instead of VLMAgent
    # api_base_url = "http://localhost:8000/v1"
    # api_key = "token-abc123456"
    # if not api_key or "sk-" not in api_key:
    #     print("Please provide a valid API key.")
        
    # agent = VLMAgent(
    #     api_key=api_key,
    #     api_base_url=api_base_url,
    #     model_name="vlm"
    # )
    
    # We will instantiate DummyAgent inside the loop to adapt to num_loc if needed, 
    # or just use a large enough range and hope for the best (or handle errors).
    # Better: initialize it with a safe upper bound or update it.
    agent = DummyAgent(num_actions=50) # Placeholder

    print("\n" + "="*50)
    print("1. Single Worker Execution (STP) with RL4COEnvironmentManager (Dummy Agent)")
    print("="*50)

    stp_data = graph_data['stp']
    n = len(stp_data)
    if n == 0:
        print("No STP instances.")
        return

    gaps = []
    
    # Run 5 instances in one batch
    batch_size = 5
    num_to_run = min(5, n)
    for i in range(0, num_to_run, batch_size):
        current_bs = min(batch_size, num_to_run - i)
        batch_data = stp_data[i : i + current_bs]
        
        # Create a list of generators, one for each environment/worker
        generators = [LoadedDataGenerator([item]) for item in batch_data]
        # Assume all instances have same num_loc or we take from first
        num_loc = generators[0].num_loc
        
        # Update agent with correct num_loc
        agent.num_actions = num_loc

        # Extract heuristic solution objectives
        heuristic_objs = []
        for item in batch_data:
            h_obj = None
            if 'objs' in item:
                objs = item['objs']
                if isinstance(objs, list) and len(objs) > 0:
                    h_obj = objs[0]
                elif hasattr(objs, 'item'):
                    h_obj = objs.item() if objs.numel() == 1 else objs[0].item()
            elif 'obj' in item:
                h_obj = item['obj']
            heuristic_objs.append(h_obj)
        
        config = SimpleNamespace(
            env=SimpleNamespace(
                env_name="stp",
                rl4co=SimpleNamespace(
                    use_format_reward=True,
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

        worker = GraphEnvs(
            env_name="stp",
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

        projection_f = partial(simple_projection, env_name="stp")
        
        env_manager = RL4COEnvironmentManager(worker, projection_f, config)
            
        metrics = run_agent_loop(env_manager, agent, solution_tour=None, env_name="stp", instance_idx=i, heuristic_objs=heuristic_objs)
        
        if 'all_gaps' in metrics:
            gaps.extend(metrics['all_gaps'])
        
        print(f"Test complete for batch starting at {i}")

    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        print("\n" + "="*50)
        print(f"Final Average Gap over {len(gaps)} instances: {avg_gap:.2f}%")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
