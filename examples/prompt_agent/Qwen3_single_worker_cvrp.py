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
from agent_system.environments.env_package.rl4co.route_obs import build_obs_cvrp
from agent_system.environments.env_package.rl4co.route_envs import RouteWorker
from agent_system.environments.env_manager import RouteEnvironmentManager
from agent_system.environments.env_package.rl4co.projection import co_projection_selected

def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

global COUNT
COUNT = 0

# def build_obs_cvrp(
#     td, 
#     env_num: int, 
#     trajectory: list = None, 
#     top_k: int = 5, 
#     given_topk_acts = None,
#     image_obs: bool = True,
#     **kwargs
# ) -> list:
#     """
#     构建 CVRP 任务的 Observation (Prompt + Image)。
#     支持 'given_topk_acts' 用于 Teacher Forcing / Injection。

#     Args:
#         td (TensorDict): 包含 'locs', 'demand', 'used_capacity', 'vehicle_capacity' 等。
#         env_num (int): Batch size。
#         trajectory (list): 历史轨迹。
#         given_topk_acts (Tensor/Array): [Batch, K] 强制指定的候选集。

#     Returns:
#         obs_list (list): [(prompt_text, image_base64), ...]
#     """
#     obs_list = []
    
#     # --- 1. 数据提取 (Batch处理) ---
#     locs = _to_numpy(td["locs"])                 # (B, N, 2)
#     demands = _to_numpy(td["demand"])            # (B, N) - 注意：通常不包含Depot需求，或Depot为0
#     current_node = _to_numpy(td["current_node"]) # (B,)
#     used_capacity = _to_numpy(td["used_capacity"]) # (B, 1) or (B,)
#     vehicle_capacity = _to_numpy(td["vehicle_capacity"]) # (B, 1) or (B,)
#     topk_acts = []
    
#     # 访问掩码 (1=visited/invalid)
#     if "action_mask" in td.keys():
#         visited = _to_numpy(td["action_mask"])   
#     else:
#         visited = np.zeros((env_num, locs.shape[1]))

#     # 处理 given_topk_acts
#     if given_topk_acts is not None:
#         given_topk_acts = _to_numpy(given_topk_acts)

#     # --- 2. 遍历每个环境 ---
#     for idx in range(env_num):
#         # 基础状态
#         curr_locs = locs[idx]          # (N, 2)
#         curr_demands = demands[idx]    # (N,)
#         curr_idx = int(current_node[idx])
#         curr_visited = visited[idx]    # (N,)
        
#         # 载重状态
#         curr_used = float(used_capacity[idx])
#         curr_cap = float(vehicle_capacity[idx])
#         remaining_cap = curr_cap - curr_used
        
        
#         # --- A. 轨迹处理 ---
#         path_history = []
#         if trajectory is not None and len(trajectory) > 0:
#             for t_step in trajectory:
#                 val = t_step[idx]
#                 if hasattr(val, 'item'): val = val.item()
#                 path_history.append(int(val))
        
#         if len(path_history) == 0 or path_history[-1] != curr_idx:
#             path_history.append(curr_idx)

#         # --- B. 生成 Top-K 候选集 ---
#         candidates = []
#         curr_pos = curr_locs[curr_idx]
        
#         # 分支 1: 注入模式 (Injection)
#         if given_topk_acts is not None:
#             topk_indices = given_topk_acts[idx]
            
#             for cand_idx in topk_indices:
#                 cand_idx = int(cand_idx)
                
#                 # 获取该点的需求量
#                 # 注意处理索引偏移：通常 locs[0] 是 depot, demands[0] 是 customer 1
#                 # 假设 demands 长度比 locs 少 1 (不含 depot) 或 长度相等 (含 depot=0)
#                 # 这里假设通用情况：demands 对应 locs 索引
#                 node_demand = 0.0
#                 if cand_idx < len(curr_demands): 
#                     node_demand = float(curr_demands[cand_idx])
#                 # 如果是 Depot (Node 0)，需求强制为 0
#                 if cand_idx == 0: node_demand = 0.0

#                 dist_val = np.linalg.norm(curr_locs[cand_idx] - curr_pos)
                
#                 candidates.append({
#                     "id": cand_idx,
#                     "dist": dist_val,
#                     "demand": node_demand,
#                     "is_depot": (cand_idx == 0),
#                     "x": curr_locs[cand_idx][0],
#                     "y": curr_locs[cand_idx][1]
#                 })

#         # 分支 2: 自动模式 (Greedy KNN)
#         else:
#             dists = cdist(curr_pos.reshape(1, 2), curr_locs, metric='euclidean').flatten()
            
#             # Mask 逻辑：
#             # 1. 已访问的点不能去 (visited==1)
#             # 2. 除非是 Depot (Node 0)，Depot 总是可以回去 (即使 visited 标记了)
#             #    但是，通常 CVRP 环境的 mask 会处理好 Depot 的可见性。
#             #    如果 mask[0] == 1 表示不能回库（比如刚出来）。
#             # 这里简单起见，信赖 env 给的 mask，但手动修正 mask[0] 的距离
            
#             # 先全部 mask 掉已访问
#             dists[curr_visited == 1] = np.inf
#             # 自身设为无穷大
#             dists[curr_idx] = np.inf
            
#             # 如果 mask 允许回库，确保距离计算正确
#             if curr_visited[0] == 0:
#                 # Depot 可选
#                 pass
            
#             # 排序
#             sorted_indices = np.argsort(dists)
#             topk_indices = sorted_indices[:top_k]
#             topk_acts.append(topk_indices)
            
#             for cand_idx in topk_indices:
#                 dist_val = dists[cand_idx]
#                 if dist_val == np.inf: break
#                 if len(candidates) >= top_k: break
                
#                 cand_idx = int(cand_idx)
#                 node_demand = 0.0
#                 if cand_idx < len(curr_demands): node_demand = float(curr_demands[cand_idx])
#                 if cand_idx == 0: node_demand = 0.0

#                 candidates.append({
#                     "id": cand_idx,
#                     "dist": dist_val,
#                     "demand": node_demand,
#                     "is_depot": (cand_idx == 0),
#                     "x": curr_locs[cand_idx][0],
#                     "y": curr_locs[cand_idx][1]
#                 })

#         # --- C. 绘图 (需实现 render_cvrp_image) ---
#         # 传入 capacity info 用于在图上显示当前载重
#         img_b64 = "PLACEHOLDER"
#         try:
#              # 如果你有 render 函数，取消注释
#              # img_b64 = render_cvrp_image(
#              #     curr_locs, curr_visited, curr_idx, path_history, candidates, 
#              #     capacity_status=(curr_used, curr_cap)
#              # )
#              pass
#         except:
#             pass

#         # --- D. 构建文本 Prompt ---
#         cand_str_list = []
#         for rank, cand in enumerate(candidates):
#             label = chr(65 + rank)
            
#             # 特殊显示 Depot
#             node_type = "**DEPOT (Refill)**" if cand['is_depot'] else f"Customer {cand['id']}"
            
#             # 需求显示逻辑
#             demand_info = f", Demand: {cand['demand']:.2f}" if not cand['is_depot'] else ""
            
#             # 辅助判断：能不能装下？
#             # 仅作为 Info 给 LLM，不做硬性过滤，让 LLM 自己学
#             feasible_mark = "" 
#             if not cand['is_depot'] and cand['demand'] > remaining_cap:
#                 feasible_mark = " [OVERLOAD!]" # 提示 LLM 这个点去了会超载（除非环境允许部分配送）

#             cand_str_list.append(
#                 f"Option {label} [{node_type}]: "
#                 f"Dist: {cand['dist']*100:.1f}{demand_info}{feasible_mark}"
#             )
#         cand_section = "\n".join(cand_str_list)
        
#         # 统计剩余未访问客户数 (排除 Depot)
#         # 假设 mask[0] 不代表“所有客户都送完了”，只统计 index 1..N
#         unvisited_customers = np.sum(curr_visited[1:] == 0)
#         step_val = len(path_history)
#         obs = (
#             f"### Task: Capacitated Vehicle Routing Problem (CVRP)\n"
#             f"Step: {step_val}\n\n"
#             f"### Status:\n"
#             f"- Current Location: Node {curr_idx} ({'Depot' if curr_idx==0 else 'Customer'})\n"
#             f"- Vehicle Load: {curr_used:.2f} / {curr_cap:.2f} (Remaining: {remaining_cap:.2f})\n"
#             f"- Unvisited Customers: {unvisited_customers}\n"
#             f"- Path History (Last 10): {path_history[-10:]}\n\n"
#             f"### Candidate Options:\n"
#             f"{cand_section}\n\n"
#             f"### Instruction:\n"
#             f"Select the Option Label (A, B...) to visit next. "
#             f"Prioritize visiting customers if capacity allows. "
#             f"Return to Depot (Node 0) ONLY if capacity is insufficient or all customers are served."
#         )
        
#         if image_obs and img_b64:
#             obs_list.append({"text": obs, "image": img_b64})
#         else:
#             obs_list.append(obs)

#     return obs_list

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
    candidates_list = []
    
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
                                obs_new = build_obs_cvrp(envs._td, 1, envs.actions,given_topk_acts=[acts_container], image_obs=True)
                                global COUNT
                                COUNT += 1     
                                print(f"error {COUNT}")

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
        solution_tour_list.append(solution_tour)
        image_list.append(img)
        
        # print(f"Action: {action_str}")
        actions, valids = co_projection_selected(actions, env_name=env_name)
        obs, rewards, dones, infos = envs.step(actions)
        dones = np.array(dones)
        # print(f"Rewards: {rewards}")
        i += 1
        
        if dones.all():
            # print("All environments done.")
            break

    return obs_list, image_list, trajectory, candidates_list, all_coords_map

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
        solution_tour = get_solution_from_data(cvrp_data[i])
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
        
        obs_list, image_list, trajectory, candidates_list, node_coords = run_agent_loop(worker, agent, solution_tour)

        json_container.append({
            "node_coords": node_coords,
            "trajectory": trajectory,
            "obs_list": obs_list,
            "image_list": image_list,
            "candidates": candidates_list,
            "solution_tour": [int(x) for x in solution_tour] if solution_tour is not None else []
        })
    
    with open("cvrp_agent_output.json", "w") as f:
        json.dump(json_container, f, indent=4, cls=NumpyEncoder)
        

if __name__ == "__main__":
    main()
