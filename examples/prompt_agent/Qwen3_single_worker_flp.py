import os
import sys
import pickle
import torch
import numpy as np
import re
import traceback
from types import SimpleNamespace
from omegaconf import OmegaConf
import ray
import json

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        return super(NumpyEncoder, self).default(obj)

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# Add rl4co-urban path
sys.path.append("/root/autodl-tmp/rl4co-urban")

from examples.prompt_agent.llm_agent import LLMAgent
from functools import partial
from agent_system.environments.env_package.rl4co.graph_env import GraphWorker
from agent_system.environments.env_package.rl4co.projection import co_projection_selected
from agent_system.environments.env_package.rl4co.graph_obs import render_flp_image, get_diverse_top_k, get_label, build_obs_flp

from scipy.spatial.distance import cdist
from typing import List, Any

global COUNT
COUNT = 0


class LoadedDataGenerator:
    def __init__(self, data_list, device="cpu"):
        self.data_list = data_list
        self.idx = 0
        self.device = device
        self.min_loc = 0.0
        self.max_loc = 1.0
        
        # Initialize attributes to avoid AttributeError
        self.num_facility = 0
        self.num_facilities_to_select = 0
        self.num_demand = 0
        self.num_terminals = 0
        
        # Infer num_loc from first data item
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
                
                # Support for MCLP
                if 'num_facility' in td.keys():
                    # Check if scalar or tensor
                    val = td['num_facility']
                    if hasattr(val, 'item'):
                        self.num_facility = val.item() if val.numel() == 1 else val[0].item()
                    else:
                        self.num_facility = val
                else:
                     # Default or try to infer from facility_locs if present
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
                
                # Support for STP
                if 'num_terminals' in td.keys():
                    val = td['num_terminals']
                    if hasattr(val, 'item'):
                        self.num_terminals = val.item() if val.numel() == 1 else val[0].item()
                    else:
                        self.num_terminals = val
                else:
                    # Try to infer from terminals list/mask if present
                    if 'terminals' in td.keys():
                         # terminals might be indices or mask
                         t = td['terminals']
                         if t.dim() >= 2:
                             self.num_terminals = t.shape[-1] # or count non-zeros if mask
                         else:
                             self.num_terminals = 0 # unclear
                    else:
                        self.num_terminals = 0

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

def get_solution_from_data(data_item):
    """
    Extract solution tour from data item if available.
    User indicated that data includes 'obj', so we check for explicit solution fields.
    """
    keys_to_check = ['tour', 'solution', 'actions', 'node_sequence']
    for key in keys_to_check:
        if key in data_item:
            # print(f"Found solution in key '{key}'.")
            val = data_item[key]
            if isinstance(val, torch.Tensor):
                val = val.numpy()
            else:
                val = np.array(val)
            return val.flatten()
            
    # Check if 'objs' contains the solution (unlikely but checking based on user hint)
    if 'objs' in data_item:
        objs = data_item['objs']
        # If objs is a list of integers/arrays, it might be the tour
        # But we saw it's a list of floats (costs).
        # We'll print a warning if we can't find a tour.
        print(f"Found 'objs' key with type {type(objs)}. Sample: {objs[:1] if isinstance(objs, list) else objs}")
        if isinstance(objs, (list, np.ndarray)) and len(objs) > 1 and isinstance(objs[0], (int, np.integer)):
             print("Assuming 'objs' contains the tour sequence.")
             return np.array(objs)

    print("Warning: No explicit solution tour found in data item.")
    return None

def run_agent_loop(worker, agent, solution_tour, env_name='flp'):
    envs = worker.env if hasattr(worker, 'env') else worker
    
    # Reset Environment
    obs, _ = envs.reset()
    
    # Prepare Data Storage
    steps_data = []
    
    # A. 获取全局坐标 (所有环境共享/或单实例)
    # 假设 batch_size=1，直接取第0个
    all_coords_map = {}
    all_coords = None # For legacy injection logic compatibility
    
    # 尝试从 envs._td 中提取坐标
    if hasattr(envs, '_td'):
        coords_array = None
        if 'locs' in envs._td.keys():
            coords_array = envs._td['locs'][0]
        elif 'facility_locs' in envs._td.keys():
            coords_array = envs._td['facility_locs'][0]
            
        if coords_array is not None:
            # Keep original reference for injection logic
            all_coords = coords_array
            
            # Convert to Dict {id: [x, y]} for JSON output
            temp_coords = coords_array
            if isinstance(temp_coords, torch.Tensor):
                temp_coords = temp_coords.cpu().numpy()
            
            for idx, coord in enumerate(temp_coords):
                all_coords_map[int(idx)] = coord.tolist()

    tour_idx = 0
    i = 0
    
    # =========================================================================
    # 2. 交互主循环
    # =========================================================================
    while True:
        # Robust obs handling
        curr_obs = obs[0]
        if isinstance(curr_obs, tuple):
            obs_text, img = curr_obs
        elif isinstance(curr_obs, dict):
            obs_text = curr_obs.get('text', str(curr_obs))
            img = curr_obs.get('image', None)
        else:
            obs_text = curr_obs
            img = None

        actions = []
        
        # 获取当前环境推荐的 Top-K 候选项 (Tensor 或 List)
        options_map = envs._td['topk_acts'][0]
        
        # Extract Candidates List for this step (New Requirement)
        candidates_list = []
        if isinstance(options_map, dict):
            candidates_list = [int(k) for k in options_map.keys()]
        else:
            # Tensor or Array
            c_vals = options_map.tolist() if hasattr(options_map, 'tolist') else list(options_map)
            candidates_list = [int(x) for x in c_vals]
        
        chosen_label = "0" # Default fallback
        
        if solution_tour is not None:
            try:
                # --- A. 准备标准答案集合 (Ground Truth) ---
                solution_set = set(s.item() if hasattr(s, 'item') else s for s in solution_tour)
                
                # --- B. 准备候选人列表 (Avail Candidates) ---
                if isinstance(options_map, dict):
                    avail_candidates = list(options_map.keys())
                else:
                    avail_candidates = options_map.tolist() if hasattr(options_map, 'tolist') else list(options_map)
                
                # --- C. 第一轮尝试：直接查找交集 ---
                target_cand = None
                target_idx = -1
                
                for k, cand in enumerate(avail_candidates):
                    c_val = cand.item() if hasattr(cand, 'item') else cand
                    if c_val in solution_set:
                        target_cand = cand
                        target_idx = k
                        break
                
                # --- D. 核心修改：注入与替换策略 (如果不匹配) ---
                if target_cand is None and len(solution_set) > 0 and len(avail_candidates) > 0:
                    if all_coords is None:
                        raise ValueError("Coordinates (locs) not found in `envs`. Cannot perform injection.")

                    remaining_opt_items = list(solution_set)
                    best_swap_pair = None 
                    min_dist = float('inf')

                    for k_idx, c_node in enumerate(avail_candidates):
                        c_val = c_node.item() if hasattr(c_node, 'item') else c_node
                        c_coord = all_coords[c_val]
                        
                        for o_node in remaining_opt_items:
                            o_coord = all_coords[o_node]
                            
                            # Euclidean Distance
                            if isinstance(c_coord, torch.Tensor):
                                d = torch.norm(c_coord - o_coord).item()
                            else:
                                d = np.linalg.norm(c_coord - o_coord)
                            
                            if d < min_dist:
                                min_dist = d
                                best_swap_pair = (k_idx, o_node)
                    
                    if best_swap_pair is not None:
                        swap_idx, swap_opt_item = best_swap_pair
                        
                        # --- E. 修改 Top-K 容器 ---
                        acts_container = envs._td['topk_acts'][0]
                        
                        if isinstance(acts_container, dict):
                            old_key = avail_candidates[swap_idx]
                            if old_key in acts_container:
                                val = acts_container.pop(old_key)
                            else:
                                val = 1.0 
                            acts_container[swap_opt_item] = val
                            avail_candidates[swap_idx] = swap_opt_item
                            target_idx = swap_idx
                            target_cand = swap_opt_item
                            
                            if swap_idx < len(candidates_list):
                                candidates_list[swap_idx] = int(swap_opt_item)

                        elif isinstance(acts_container, list):
                            acts_container.pop(swap_idx)
                            acts_container.append(swap_opt_item)
                            target_idx = len(acts_container) - 1 
                            target_cand = swap_opt_item
                            
                            candidates_list.pop(swap_idx)
                            candidates_list.append(int(swap_opt_item))

                        elif isinstance(acts_container, torch.Tensor):
                            new_val = swap_opt_item.item() if hasattr(swap_opt_item, 'item') else swap_opt_item
                            new_node_tensor = torch.tensor([new_val], dtype=acts_container.dtype, device=acts_container.device)
                            part_before = acts_container[:swap_idx]
                            part_after = acts_container[swap_idx+1:]
                            new_container = torch.cat([part_before, part_after, new_node_tensor]) 
                            envs._td['topk_acts'][0] = new_container
                            
                            target_idx = len(acts_container) - 1 
                            target_cand = swap_opt_item
                            
                            candidates_list.pop(swap_idx)
                            candidates_list.append(int(swap_opt_item))
                        
                        global COUNT
                        COUNT += 1
                        print(f"Count: {COUNT}")
                        obs = build_obs_flp(envs._td,1, given_topk_acts= [envs._td['topk_acts'][0]], image_obs=True)
                        obs_text = obs[0]['text']
                        img = obs[0]['image']
                
                # --- F. 生成 Label (Action String) ---
                if target_cand is not None and target_idx != -1:
                    chosen_label = chr(ord('A') + target_idx)
                else:
                     if len(options_map) > 0:
                         chosen_label = 'A'

            except Exception as e:
                print(f"Error following solution: {e}")
                traceback.print_exc()
                chosen_label = "0"
        
        # 格式化动作字符串，例如 \boxed{A}
        action_str = f"\\boxed{{{chosen_label}}}"
        actions.append(action_str)
        
        # Store Step Data (New Format)
        step_record = {
            "step_idx": i,
            "obs": obs_text,
            "image": img, # Base64 string
            "trajectory": action_str,
            "candidates": candidates_list,
            "solution_tour": [int(x) for x in solution_tour] if solution_tour is not None else []
        }
        steps_data.append(step_record)
        
        print(f"Action: {action_str}")
        
        # 调用你的投影函数 (假设存在)
        actions, valids = co_projection_selected(actions, env_name=env_name)
        
        # 环境执行一步
        obs, rewards, dones, infos = envs.step(actions)
        
        dones = np.array(dones)
        i += 1
        
        if dones.all():
            print("All environments done.")
            break

    # Unzip steps_data into parallel lists
    trajectory = [s['trajectory'] for s in steps_data]
    obs_list = [s['obs'] for s in steps_data]
    image_list = [s['image'] for s in steps_data]
    candidates_list = [s['candidates'] for s in steps_data]

    return obs_list, image_list, trajectory, candidates_list, all_coords_map

def main():
    graph_data, routing_data = load_data()
    
    # Configuration
    api_key = "sk-saxqqtlyqrpconxlgcslqhrgvhwnfmuhnimiyzfvpcxqgmkh"
    agent = LLMAgent(
        api_key=api_key,
        api_base_url="https://api.siliconflow.cn/v1",
        model_name="Qwen/Qwen2.5-7B-Instruct"
    )

    print("\n" + "="*50)
    print("1. Single Worker Execution (FLP)")
    print("="*50)

    # Environments to process
    target_envs = ['flp']

    for env_name in target_envs:
        if env_name not in graph_data:
            print(f"Skipping {env_name} (not in graph_data)")
            continue
            
        print(f"\n>>> Running Environment: {env_name}")
        data = graph_data[env_name] 
        n = 1#len(data)
        
        # Limit to first item for testing as per original code structure
        # Or iterate all if intended. The original code looped i in range(n)
        json_container = []
        for i in range(n):
            print(f"  -- Instance {i} --")
            generator = LoadedDataGenerator(data[i:i+1])
            
            # Try to get solution from data
            solution_tour = get_solution_from_data(data[i])
            if solution_tour is None:
                print("No solution tour available in data.")

            # Setup Environment
            worker = GraphWorker(
                env_name=env_name,  
                seed=42,
                env_num=1,
                device="cpu",
                num_loc=generator.num_loc,
                return_topk_options=24, # Unified topk
                image_obs=True,
                env_kwargs={"generator": generator}
            )
        
            # Define projection function
            projection_f = partial(co_projection_selected, env_name=env_name)
        
            # Ensure generator starts from 0 for the agent loop
            generator.idx = 0
            
            obs_list, image_list, trajectory, candidates_list, node_coords = run_agent_loop(worker, agent, solution_tour, env_name=env_name)

            json_container.append({
                "node_coords": node_coords,
                "trajectory": trajectory,
                "obs_list": obs_list,
                "image_list": image_list,
                "candidates": candidates_list,
                "solution_tour": [int(x) for x in solution_tour] if solution_tour is not None else []
            })
        
        with open(f"{env_name}_agent_output.json", "w") as f:
            json.dump(json_container, f, indent=4, cls=NumpyEncoder)

if __name__ == "__main__":
    main()
