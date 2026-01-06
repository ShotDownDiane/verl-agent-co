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
# Add rl4co-urban path
sys.path.append("/root/autodl-tmp/rl4co-urban")

from examples.prompt_agent.llm_agent import LLMAgent
from functools import partial
from agent_system.environments.env_package.rl4co.graph_obs import build_obs_mclp
from agent_system.environments.env_package.rl4co.graph_env import GraphWorker
from agent_system.environments.env_package.rl4co.projection import co_projection_selected
from agent_system.environments.env_package.rl4co.graph_obs import get_label


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

import torch
import numpy as np
import traceback

def run_agent_loop(envs, agent, solution_tour=None, env_name="flp"):
    """
    SFT 数据生成主循环
    包含：Teacher Forcing, Geometric Injection, Swap-to-Last 策略
    """
    # print(f"Resetting environments...")
    obs, infos = envs.reset()
    
    trajectory = []
    obs_list = []
    solution_tour_list = []
    image_list = []
    candidates_list = []
    
    # =========================================================================
    # 1. 静态数据准备 (坐标 & 边列表)
    # =========================================================================
    
    # A. 获取全局坐标 (所有任务都需要计算几何距离)
    all_coords = None
    all_coords_map = {}
    # 根据 tensordict 的不同 key 尝试获取
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
    
    # B. 获取全局边列表 (仅 STP/GSTP 需要)
    # 我们需要知道 edge_idx 到底连接了哪两个点
    global_edge_list = None
    if env_name in ['stp', 'gstp'] and hasattr(envs, '_td'):
        # 尝试获取 edge_index 或 edge_list
        if 'edge_index' in envs._td.keys():
            edges = envs._td['edge_index'][0]
        elif 'edge_list' in envs._td.keys():
            edges = envs._td['edge_list'][0]
        else:
            edges = None
            
        if edges is not None:
            # 统一转置为 [Num_Edges, 2] 格式，方便用 idx 索引
            if edges.shape[0] == 2 and edges.shape[1] != 2: 
                global_edge_list = edges.T 
            else:
                global_edge_list = edges

    tour_idx = 0
    i = 0
    
    # =========================================================================
    # 2. 交互主循环
    # =========================================================================
    while True:
        # print(f"\n--- Step {i+1} ---")
        
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
        # 注意：这里获取的是引用，稍后修改它会影响环境状态
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
        
        chosen_label = "0" # Default fallback
        
        if solution_tour is not None:
            try:
                # --- A. 准备标准答案集合 (Ground Truth) ---
                # 将 tensor 转换为 python scalar，放入 set 方便 O(1) 查找
                solution_set = set(s.item() if hasattr(s, 'item') else s for s in solution_tour)
                
                # --- B. 准备候选人列表 (Avail Candidates) ---
                if isinstance(options_map, dict):
                    avail_candidates = list(options_map.keys())
                else:
                    # 转为 list 方便后续操作
                    avail_candidates = options_map.tolist() if hasattr(options_map, 'tolist') else list(options_map)
                
                # --- C. 第一轮尝试：直接查找交集 ---
                target_cand = None
                target_idx = -1
                
                for k, cand in enumerate(avail_candidates):
                    c_val = cand.item() if hasattr(cand, 'item') else cand
                    if c_val in solution_set:
                        target_cand = cand
                        target_idx = k
                        # print(f"Found target {c_val} in solution set.")
                        break
                
                # --- D. 核心修改：注入与替换策略 (如果不匹配) ---
                # 如果 Top-K 全军覆没，且我们要强行教学
                if target_cand is None and len(solution_set) > 0 and len(avail_candidates) > 0:
                    # print(f"[{env_name.upper()}] Target not in Top-K. Calculating geometric replacement...")
                    
                    if all_coords is None:
                        # 如果没有坐标，无法计算距离，只能报错或跳过
                        raise ValueError("Coordinates (locs) not found in `envs`. Cannot perform injection.")

                    remaining_opt_items = list(solution_set)
                    best_swap_pair = None 
                    min_dist = float('inf')


                    for k_idx, c_node in enumerate(avail_candidates):
                        c_val = c_node.item() if hasattr(c_node, 'item') else c_node
                        c_coord = all_coords[c_val]
                        
                        for o_node in remaining_opt_items:
                            o_coord = all_coords[o_node]
                            
                            if isinstance(c_coord, torch.Tensor):
                                dist = torch.norm(c_coord - o_coord).item()
                            else:
                                dist = np.linalg.norm(c_coord - o_coord)
                            
                            if dist < min_dist:
                                min_dist = dist
                                best_swap_pair = (k_idx, o_node)
                    
                    # --- E. 执行注入 & 移至末位 (Swap to Last) ---
                    if best_swap_pair is not None:
                        swap_idx, swap_opt_item = best_swap_pair
                        
                        # print(f"Injection ({env_name}): Removing Idx {swap_idx}, Appending Optimal {swap_opt_item} to Last.")

                        # 获取容器引用
                        acts_container = envs._td['topk_acts'][0]

                        # -------------------------------------------------
                        # 情况 1: PyTorch Tensor 处理
                        # -------------------------------------------------
                        if isinstance(acts_container, torch.Tensor):
                            # 1. 构造新节点的 Tensor (保持与原容器相同的 Device 和 Dtype)
                            # 注意：swap_opt_item 可能是 int 或 scalar tensor
                            new_val = swap_opt_item.item() if hasattr(swap_opt_item, 'item') else swap_opt_item
                            new_node_tensor = torch.tensor([new_val], dtype=acts_container.dtype, device=acts_container.device)

                            # 2. 拼接操作：[0...swap_idx-1] + [swap_idx+1...end] + [new_node]
                            # 即：跳过 swap_idx 处的元素，将其余拼接，最后加上新元素
                            part_before = acts_container[:swap_idx]
                            part_after = acts_container[swap_idx+1:]
                            
                            new_container = torch.cat([part_before, part_after, new_node_tensor])
                            
                            # 3. 更新环境中的数据
                            envs._td['topk_acts'][0] = new_container
                            
                            # 4. 更新目标索引指向最后一位
                            target_idx = len(new_container) - 1
                            target_cand = swap_opt_item

                        # -------------------------------------------------
                        # 情况 2: Python List 处理
                        # -------------------------------------------------
                        elif isinstance(acts_container, list):
                            # 1. 弹出指定位置的元素 (Remove)
                            acts_container.pop(swap_idx)
                            
                            # 2. 追加新元素到末尾 (Append)
                            acts_container.append(swap_opt_item)
                            
                            # 3. 更新目标索引
                            target_idx = len(acts_container) - 1
                            target_cand = swap_opt_item
                        
                        else:
                            # 兜底：如果是 Dict 或其他类型，暂时无法处理顺序注入
                            pass
                        global COUNT
                        COUNT += 1
                        print(f"COUNT: {COUNT}")
                        print("old obs:", obs_text)
                        obs_new = build_obs_mclp(envs._td, 1, given_topk_acts= [acts_container], image_obs=True)
                        obs_text = obs_new[0]['text']
                        img = obs_new[0]['image']
                        print("new obs:", obs_text)
                # --- F. 生成 Label (Action String) ---
                if target_cand is not None and target_idx != -1:
                    # 0->A, 1->B, ...
                    chosen_label = chr(ord('A') + target_idx)
                else:
                     # 兜底：如果实在没法生成（极罕见），默认选 A
                     if len(options_map) > 0:
                         chosen_label = 'A'

            except Exception as e:
                print(f"Error following solution: {e}")
                traceback.print_exc()
                chosen_label = "0"
        
        # 格式化动作字符串，例如 \boxed{A}
        action_str = f"\\boxed{{{chosen_label}}}"
        actions.append(action_str)
        trajectory.append(action_str)
        obs_list.append(obs_text)
        image_list.append(img)
        solution_tour_list.append(solution_tour)
        
        # print(f"Action: {action_str}")
        
        # 调用你的投影函数 (假设存在)
        actions, valids = co_projection_selected(actions, env_name=env_name)
        
        # 环境执行一步
        # 关键：因为我们在步骤 E 中修改了 _td['topk_acts']，
        # 所以这里的 step 会根据 action_str 正确映射到我们注入的 opt_item
        obs, rewards, dones, infos = envs.step(actions)
        
        dones = np.array(dones)
        # print(f"Rewards: {rewards}")
        i += 1
        
        if dones.all():
            print("All environments done.")
            break

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
    print("1. Single Worker Execution (MCLP)")
    print("="*50)

    # Environments to process
    target_envs = ['mclp']

    for env_name in target_envs:
        if env_name not in graph_data:
            print(f"Skipping {env_name} (not in graph_data)")
            continue
            
        print(f"\n>>> Running Environment: {env_name}")
        data = graph_data[env_name]
        n = len(data)
        
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
