import os
import sys
import pickle
import torch
import numpy as np
import re
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

from agent_system.environments.env_manager import make_envs
from examples.prompt_agent.llm_agent import LLMAgent
from functools import partial
from agent_system.environments.env_package.rl4co.route_obs import build_obs_tsp
from agent_system.environments.env_package.rl4co.route_envs import RouteWorker
from agent_system.environments.env_manager import RouteEnvironmentManager
from agent_system.environments.env_package.rl4co.projection import co_projection_selected

global COUNT
COUNT = 0

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
            if isinstance(val[0][0], list):
                # Flatten nested list
                val = val[0]
                val = [item for sublist in val for item in sublist]
                val = [val]
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


import torch
import numpy as np
import traceback

def run_agent_loop(envs, agent, solution_tour=None, env_name="cvrp"):
    # print(f"Resetting environments...")
    obs, infos = envs.reset()
    trajectory = []
    obs_list = []
    solution_tour_list = []
    image_list = []
    
    # =========================================================================
    # 1. 获取全局坐标 (用于计算几何距离)
    # =========================================================================
    all_coords = None
    if hasattr(envs, '_td'):
        if 'locs' in envs._td.keys():
            all_coords = envs._td['locs'][0]
        elif 'facility_locs' in envs._td.keys():
            all_coords = envs._td['facility_locs'][0]
            
    tour_idx = 0
    i = 0
    
    while True:
        # print(f"\n--- Step {i+1} ---")
        obs_text, img = obs[0]['text'], obs[0]['image']

        actions = []
        
        # Parse observation
        current_node = envs._td['current_node'][0]
        # 注意：获取引用，以便修改
        options_map = envs._td['topk_acts'][0]
        
        chosen_label = "0" # Default
        
        if solution_tour is not None and current_node is not None:
            # Sequential matching for Routing (TSP/CVRP)
            try:
                # Normalize current_node to scalar
                c_node = current_node.item() if hasattr(current_node, 'item') else current_node
                
                # Check if we are on track (同步当前位置)
                if tour_idx < len(solution_tour):
                    expected_node = solution_tour[tour_idx]
                    e_node = expected_node.item() if hasattr(expected_node, 'item') else expected_node
                    
                    if c_node != e_node:
                        # print(f"Mismatch! Current {c_node} != Expected {e_node} (index {tour_idx})")
                        # Recovery: scan forward
                        future_tour = solution_tour[tour_idx:]
                        matches = np.where(future_tour == c_node)[0]
                        if len(matches) > 0:
                            # print(f"Recovered: Found {c_node} at offset {matches[0]}")
                            tour_idx += matches[0]
                        else:
                            # print("Lost track of tour. Continuing blindly.")
                            pass
                            
                    # --- 核心逻辑：寻找下一步的目标节点 ---
                    if tour_idx < len(solution_tour) - 1:
                        target_node = solution_tour[tour_idx + 1]
                        # Normalize target_node
                        t_node = target_node.item() if hasattr(target_node, 'item') else target_node
                        
                        found_opt = False
                        target_idx_in_opts = -1
                        
                        # A. 检查 options_map 类型并提取列表
                        if isinstance(options_map, dict):
                            avail_opts = list(options_map.keys())
                        else:
                            avail_opts = options_map.tolist() if hasattr(options_map, 'tolist') else list(options_map)
                        
                        # B. 在候选中查找 Target
                        if t_node in avail_opts:
                            target_idx_in_opts = avail_opts.index(t_node)
                            found_opt = True
                        
                        # ====================================================
                        # C. 【注入逻辑】如果没找到，且不是 Dict 类型 (Dict难改序)
                        # ====================================================
                        if not found_opt and not isinstance(options_map, dict):
                            if all_coords is None:
                                # print("Warning: Coordinates missing, cannot perform geometric injection.")
                                pass
                            elif len(avail_opts) > 0:
                                # print(f"[{env_name.upper()}] Target {t_node} not in Top-K. Calculating replacement...")
                                
                                # 1. 随机置换最后5个中的一个
                                num_opts = len(avail_opts)
                                pool_size = min(5, num_opts)
                                start_idx = num_opts - pool_size
                                swap_idx = np.random.randint(start_idx, num_opts)

                                # 2. 获取 Top-K 容器引用
                                acts_container = envs._td['topk_acts'][0]

                                # 3. 覆盖 (Overwrite)
                                acts_container[swap_idx] = t_node
                                target_idx_in_opts = swap_idx

                                found_opt = True
                                global COUNT
                                COUNT += 1
                                print(f"COUNT: {COUNT}")  
                                print("old obs:", obs_text)                                  
                                obs_new = build_obs_tsp(envs._td, 1, trajectory=envs.actions,given_topk_acts=[acts_container], image_obs=True)
                                obs_text = obs_new[0]['text']
                                img = obs_new[0]['image']
                                print("new obs:", obs_text)
                            

                        # D. 生成 Label
                        if found_opt:
                            if isinstance(options_map, dict):
                                chosen_label = options_map[t_node]
                            else:
                                chosen_label = chr(ord('A') + target_idx_in_opts)
                            
                            # print(f"Planned move: {t_node} -> Option {chosen_label}")
                            tour_idx += 1 
                        else:
                             # print(f"Target node {t_node} NOT in options! Available: {avail_opts}")
                             # Fallback: Pick first option
                             if isinstance(options_map, dict) and options_map:
                                 chosen_label = list(options_map.values())[0]
                             elif not isinstance(options_map, dict) and len(options_map) > 0:
                                 chosen_label = 'A'
                        
                    else:
                        # print("At end of tour.")
                        chosen_label = "0"
                else:
                     # print("Tour index out of bounds.")
                     pass
            except Exception as e:
                print(f"Error following tour: {e}")
                traceback.print_exc()
                chosen_label = "0"
        
        # Format action for projection
        action_str = f"\\boxed{{{chosen_label}}}"
        print(f"Action: {action_str}")
        actions.append(action_str)
        trajectory.append(action_str)
        obs_list.append(obs_text)
        image_list.append(img)
        solution_tour_list.append(solution_tour)
        
        # print(f"Action: {action_str}")
        actions, valids = co_projection_selected(actions, env_name=env_name)
        obs, rewards, dones, infos = envs.step(actions)
        dones = np.array(dones)
        # print(f"Rewards: {rewards}")
        i += 1
        
        if dones.all():
            # print("All environments done.")
            break
            
    return obs_list, image_list, solution_tour_list, trajectory

def main():
    _, routing_data = load_data()
    
    # Configuration
    api_key = "sk-saxqqtlyqrpconxlgcslqhrgvhwnfmuhnimiyzfvpcxqgmkh"
    agent = LLMAgent(
        api_key=api_key,
        api_base_url="https://api.siliconflow.cn/v1",
        model_name="Qwen/Qwen2.5-7B-Instruct"
    )

    print("\n" + "="*50)
    print("1. Single Worker Execution (TSP)")
    print("="*50)

    tsp_data = routing_data['tsp']
    n = len(tsp_data)
    json_container = []

    for i in range(n):
        generator = LoadedDataGenerator(tsp_data[i:i+1])
        
        # Try to get solution from data
        solution_tour = get_solution_from_data(tsp_data[i])
        if solution_tour is None:
            print("No solution tour available in data.")

        # 2. Setup Environment directly (Bypassing Ray)
        # print("Initializing RouteWorker directly...")
        
        # Create worker directly
        worker = RouteWorker(
            env_name="tsp",
            seed=42,
            env_num=1,
            device="cpu",
            num_loc=generator.num_loc,
            return_topk_options=20,
            image_obs=True,
            env_kwargs={"generator": generator}
        )
    
        # Define projection function
        projection_f = partial(co_projection_selected, env_name="tsp")
    
    
        # Ensure generator starts from 0 for the agent loop
        generator.idx = 0
        
        obs_list, image_list, solution_tour_list, trajectory = run_agent_loop(worker, agent, solution_tour, env_name="tsp")

        json_container.append({
            "obs_list": obs_list,
            "image_list": image_list,
            "solution_tour_list": solution_tour_list,
            "trajectory": trajectory,
        })
    
    with open("tsp_agent_output.json", "w") as f:
        json.dump(json_container, f, indent=4, cls=NumpyEncoder)

if __name__ == "__main__":
    main()
