import os
import sys
import pickle
import torch
import numpy as np
import re
import base64
from types import SimpleNamespace
from omegaconf import OmegaConf
import ray
import math
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
from agent_system.environments.env_package.rl4co.route_obs import build_obs_cvrp, render_cvrp_image
from agent_system.environments.env_package.rl4co.route_envs import RouteWorker
from agent_system.environments.env_manager import RouteEnvironmentManager
from agent_system.environments.env_package.rl4co.projection import co_projection_selected

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

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


def get_solution_from_data(data_item, locs, depot_loc):
    """
    Extract solution tour from data item and SORT routes by polar angle.
    
    Args:
        data_item: dict containing solver output
        locs: torch.Tensor, shape (N, 2) - Customer coordinates
        depot_loc: torch.Tensor, shape (1, 2) - Depot coordinate
    
    Returns:
        np.array: Flattened, sorted node sequence.
    """
    keys_to_check = ['tour', 'solution', 'actions', 'node_sequence']
    raw_routes = None

    # 1. Extract the raw routes (List of Lists)
    for key in keys_to_check:
        if key in data_item:
            val = data_item[key]
            # Handle standard nested list structure [[route1], [route2]]
            if isinstance(val, list):
                # Check if it's double nested like [[ [r1], [r2] ]] (batch dim artifact)
                if len(val) > 0 and isinstance(val[0], list) and len(val[0]) > 0 and isinstance(val[0][0], list):
                    val = val[0] 
                raw_routes = val
            elif isinstance(val, torch.Tensor):
                # If it's a tensor, we assume it might be padded or needs careful handling
                # Converting to list of lists for easier manipulation
                # This part depends on your tensor shape, assuming it separates routes somehow
                # or is just a flat sequence that we can't easily sort without delimiters.
                # For safety, let's convert to numpy if it's simple.
                raw_routes = val.tolist()
            elif isinstance(val, np.ndarray):
                raw_routes = val.tolist()
            
            break # Found a key
            
    if raw_routes is None:
        print("Warning: No explicit solution tour found.")
        return None

    # Ensure raw_routes is a list of lists: [[1, 5, 2], [3, 4], ...]
    # If it was already flattened, we can't sort by route. 
    # We assume the solver output maintains route separation (e.g., LKH usually does).
    if not isinstance(raw_routes[0], list):
        # Fallback: If data is already flattened, we can't sort petals. Return as is.
        return np.array(raw_routes)

    # 2. Define Helper to calculate Route Angle
    def get_route_angle(route):
        """Calculates the polar angle of the route's centroid relative to depot."""
        if not route: return 0
        
        # Gather coordinates for all nodes in this route
        # Assumption: route indices map to 'locs'. 
        # Check if indices are 1-based (common in VRPLIB) or 0-based.
        # If your locs includes depot at 0, then index is direct. 
        # If locs is ONLY customers, and route uses 1-based index: idx-1.
        # Here we assume standard Python 0-based index into 'locs' for simplicity.
        # You may need to adjust (n-1) if your solver uses 1-based indexing.
        
        route_coords = []
        for node_idx in route:
            # Safety check for index bounds
            if node_idx < len(locs):
                route_coords.append(locs[node_idx-1])
                break
        
        if not route_coords: return 0
        
        # Convert to tensor/numpy for mean calc
        if isinstance(route_coords[0], torch.Tensor):
            stack_coords = torch.stack(route_coords)
            centroid = torch.mean(stack_coords, dim=0)
            cx, cy = centroid[0].item(), centroid[1].item()
        else:
            stack_coords = np.array(route_coords)
            centroid = np.mean(stack_coords, axis=0)
            cx, cy = centroid[0], centroid[1]

        # Depot coordinates
        if isinstance(depot_loc, torch.Tensor):
            dx, dy = depot_loc[0][0].item(), depot_loc[0][1].item()
        else:
            dx, dy = depot_loc[0][0], depot_loc[0][1]

        # Calculate Angle (atan2 returns -pi to pi)
        return -math.atan2(cy - dy, cx - dx)

    # 3. Sort Routes based on Angle
    # This aligns the routes in a circular sweep order (e.g., -pi to pi)
    sorted_routes = sorted(raw_routes, key=get_route_angle)

    # 4. Flatten the result for the Agent
    # Now the sequence is: Petal 1 -> Petal 2 -> Petal 3 (sequentially adjacent)
    flat_solution = [node for route in sorted_routes for node in route]

    return np.array(flat_solution)


import torch
import numpy as np
import traceback

def run_agent_loop(envs, agent, solution_tour=None, env_name="cvrp", instance_idx=0):
    # print(f"Resetting environments...")
    obs, infos = envs.reset()
    trajectory = []
    obs_list = []
    solution_tour_list = []
    image_list = []
    candidates_list = []
    
    # Capture static info
    all_demand = envs._td['demand'][0].cpu().numpy().tolist()
    vehicle_capacity = envs._td['vehicle_capacity'][0].item()
    load_list = []
    
    os.makedirs(f"debug_images/{env_name}", exist_ok=True)
    
    
    # Generate Pure Obs Image (Start)
    pure_obs_image = ""
    if hasattr(envs, '_td'):
        pure_obs_path = f"debug_images/{env_name}/inst_{instance_idx}_pure_obs.png"
        try:
            render_cvrp_image(
                locs=envs._td['locs'][0].cpu().numpy(),
                demands=envs._td['demand'][0].cpu().numpy(),
                visited_mask=np.zeros(envs._td['locs'][0].shape[0], dtype=bool),
                current_node_idx=0, # Depot
                path_history=[0],
                used_capacity=0.0,
                vehicle_capacity=envs._td['vehicle_capacity'][0].item(),
                top_candidates=[],
                debug_save_path=pure_obs_path
            )
            pure_obs_image = pure_obs_path
        except Exception as e:
            print(f"Error generating pure obs image: {e}")
            traceback.print_exc()
    
    # =========================================================================
    # 1. 获取全局坐标 (用于计算几何距离)
    # =========================================================================
    all_coords = None
    all_coords_map = {}
    if hasattr(envs, '_td'):
        if 'locs' in envs._td.keys():
            all_coords = envs._td['locs'][0]
        elif 'facility_locs' in envs._td.keys():
            all_coords = envs._td['facility_locs'][0]
            
        if all_coords is not None:
            temp_coords = all_coords
            if isinstance(temp_coords, torch.Tensor):
                temp_coords = temp_coords.cpu().numpy()
            for idx, coord in enumerate(temp_coords):
                all_coords_map[int(idx)] = coord.tolist()
            
    tour_idx = 0
    i = 0
    solution_tour = np.insert(solution_tour,0,0)
    while True:
        # Record current load
        current_load = envs._td['used_capacity'][0].item()
        load_list.append(current_load)

        # print(f"\n--- Step {i+1} ---")
        obs_text, img = obs[0]['text'], obs[0]['image']
        actions = []
        
        # Parse observation
        current_node = envs._td['current_node'][0]
        # 注意：获取引用，以便修改
        options_map = envs._td['topk_acts'][0]
        
        # Extract Candidates List for this step
        current_candidates = []
        if isinstance(options_map, dict):
            current_candidates = [int(k) for k in options_map.keys()]
        else:
            # Tensor or Array
            c_vals = options_map.tolist() if hasattr(options_map, 'tolist') else list(options_map)
            current_candidates = [int(x) for x in c_vals]
        candidates_list.append(current_candidates)
        
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
                            print(f"Recovered: Found {c_node} at offset {matches[0]}. New index: {tour_idx}")
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
                                print(f"[{env_name.upper()}] Target {t_node} not in Top-K. Calculating replacement...")
                                # 随机替换最后4个之一
                                target_idx_in_opts = np.random.randint(0, 4) + len(avail_opts) - 4
                                chosen_label = str(target_idx_in_opts)
                                acts_container = envs._td['topk_acts'][0]
                                acts_container[target_idx_in_opts] = t_node
                                found_opt = True
                                obs_new = build_obs_cvrp(envs._td, 1, envs.actions, given_topk_acts=[acts_container], image_obs=True)
                                global COUNT
                                COUNT += 1     
                                print(f"error {COUNT}")
                                # 更新 obs_text 和 img
                                obs_text = obs_new[0]['text']
                                img = obs_new[0]['image']

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
                        print("At end of tour.")
                        raise ValueError("Tour index out of bounds.")
                else:
                     # print("Tour index out of bounds.")
                     pass
            except Exception as e:
                print(f"Error following tour: {e}")
                traceback.print_exc()
                chosen_label = "0"
        
        # Format action for projection
        action_str = f"\\boxed{{{chosen_label}}}"
        action_str_clean = f"{chosen_label}"
        print(f"Action: {action_str}")
        actions.append(action_str)
        trajectory.append(action_str_clean)
        obs_list.append(obs_text)
        solution_tour_list.append(solution_tour)
        
        # Save step image from obs
        if img:
            try:
                img_data = base64.b64decode(img)
                step_img_path = f"debug_images/{env_name}/inst_{instance_idx}_step_{i}.png"
                with open(step_img_path, "wb") as f:
                    f.write(img_data)
                image_list.append(img)
            except Exception as e:
                print(f"Error saving step image: {e}")
                image_list.append("")
        else:
            image_list.append("")
        
        # print(f"Action: {action_str}")
        actions, valids = co_projection_selected(actions, env_name=env_name)
        obs, rewards, dones, infos = envs.step(actions)
        dones = np.array(dones)
        # print(f"Rewards: {rewards}")
        i += 1
        
        if dones.all():
            # print("All environments done.")
            break

    # Generate Final Solution Image
    final_solution_image = ""
    if hasattr(envs, '_td'):
        final_sol_path = f"debug_images/{env_name}/inst_{instance_idx}_final_solution.png"
        try:
            # Reconstruct path from trajectory and candidates
            path_indices = [0] # Start at depot
            for i, action_str in enumerate(trajectory):
                lbl = action_str
                if 'A' <= lbl <= 'Z':
                    idx = ord(lbl) - ord('A')
                    if i < len(candidates_list):
                        cands = candidates_list[i]
                        if idx < len(cands):
                            path_indices.append(cands[idx])
            
            # Calculate visited mask based on path
            visited_mask = np.zeros(envs._td['locs'][0].shape[0], dtype=bool)
            for node_idx in path_indices:
                if node_idx != 0:
                    visited_mask[node_idx] = True
            path_indices.append(0)
            render_cvrp_image(
                locs=envs._td['locs'][0].cpu().numpy(),
                demands=envs._td['demand'][0].cpu().numpy(),
                visited_mask=visited_mask,
                current_node_idx=path_indices[-1],
                path_history=path_indices,
                used_capacity=envs._td['used_capacity'][0].item(),
                vehicle_capacity=envs._td['vehicle_capacity'][0].item(),
                top_candidates=[],
                debug_save_path=final_sol_path
            )
            final_solution_image = final_sol_path
        except Exception as e:
            print(f"Error generating final solution image: {e}")
            traceback.print_exc()

    return obs_list, image_list, trajectory, candidates_list, all_coords_map, pure_obs_image, final_solution_image, all_demand, load_list, vehicle_capacity

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
    print("1. Single Worker Execution (CVRP)")
    print("="*50)

    cvrp_data = routing_data['cvrp']
    n = len(cvrp_data)
    json_container = []
    for i in range(n):
        generator = LoadedDataGenerator(cvrp_data[i:i+1])
        
        # Try to get solution from data
        locs = cvrp_data[i]['td']['locs'][0]
        depot_loc = cvrp_data[i]['td']['depot']
        solution_tour = get_solution_from_data(cvrp_data[i], locs, depot_loc)
        if solution_tour is None:
            print("No solution tour available in data.")

        # 2. Setup Environment directly (Bypassing Ray)
        # print("Initializing RouteWorker directly...")
        
        # Create worker directly
        worker = RouteWorker(
            env_name="cvrp",
            seed=42,
            env_num=1,
            device="cpu",
            num_loc=generator.num_loc,
            return_topk_options=26,
            image_obs=True,
            env_kwargs={"generator": generator}
        )
        # Define projection function
        projection_f = partial(co_projection_selected, env_name="cvrp")
    
    
        # Ensure generator starts from 0 for the agent loop
        generator.idx = 0
        
        obs_list, image_list, trajectory, candidates_list, node_coords, pure_obs_image, final_solution_image, demand, load_list, capacity = run_agent_loop(worker, agent, solution_tour, instance_idx=i)

        json_container.append({
            "node_coords": node_coords,
            "trajectory": trajectory,
            "obs_list": obs_list,
            "image_list": image_list,
            "candidates": candidates_list,
            "solution_tour": [int(x) for x in solution_tour] if solution_tour is not None else [],
            "pure_obs_image_path": pure_obs_image,
            "final_solution_image_path": final_solution_image,
            "demand": demand,
            "load_list": load_list,
            "capacity": capacity
        })
    
    with open("cvrp_agent_output.json", "w") as f:
        json.dump(json_container, f, indent=4, cls=NumpyEncoder)
        

if __name__ == "__main__":
    main()
