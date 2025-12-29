import torch
import cv2
import os
import base64
from tensordict.tensordict import TensorDict
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from typing import List, Any
import numpy as np

def _to_numpy(x: Any):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def get_label(i: int) -> str:
    """Generate option labels: A, B, ..., Z, AA, AB, ..."""
    if 0 <= i < 26:
        return chr(65 + i)
    else:
        # Fallback for > 26: AA, AB... (Simplified to Opt{i} for now or extend logic)
        return f"Opt{i}"

def render_flp_image(locs, chosen_indices, top_candidates, img_size=512, debug_save_path=None):
    """
    生成 FLP 状态图像 (Base64编码)。
    视觉编码:
    - 背景: 白色
    - 客户点: 根据"当前最小距离"着色。越远越红(Red)，越近越淡(Fade to Grey)。
    - 已选设施: 蓝色方块 (Blue Square)
    - 候选设施: 绿色圆圈 + 黑色文字标签 (A, B, C...)
    """
    # 1. 初始化画布 (H, W, 3) - BGR 格式
    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # 坐标映射: [0, 1] -> [padding, size-padding]
    padding = 20
    scale = img_size - 2 * padding
    
    def to_xy(coords):
        return (int(coords[0] * scale + padding), int(coords[1] * scale + padding))

    # 2. 计算每个点的"痛苦程度" (距离最近设施的距离) 用于热力图着色
    # 如果是第一步 (chosen_indices为空)，所有点距离为无穷(最大痛感)
    num_nodes = locs.shape[0]
    if len(chosen_indices) > 0:
        chosen_locs = locs[chosen_indices]
        dists = cdist(locs, chosen_locs, metric='euclidean')
        min_dists = np.min(dists, axis=1)
        
        # [优化点] 调整归一化阈值，增强对比度
        # 距离小于 0.1 的认为已经被很好覆盖，距离大于 0.4 的认为是严重未覆盖
        norm_dists = np.clip((min_dists - 0.05) / 0.35, 0, 1)
    else:
        norm_dists = np.ones(num_nodes) # 第一步全红

    # 3. 绘制所有客户点 (Customer Nodes)
    for i in range(num_nodes):
        if i in chosen_indices: continue # 已选点稍后画

        pt = to_xy(locs[i])
        intensity = norm_dists[i]
        
        # [优化点] 颜色映射：近处接近白色(240)，远处纯红(0,0,255)
        val = int(240 * (1 - intensity)) 
        color = (val, val, 255) # BGR
        
        # [优化点] 远处点画大一点，近处点画小一点
        radius = 5 if intensity > 0.5 else 3
        cv2.circle(canvas, pt, radius, color, -1)

    # 4. 绘制已选设施 (Existing Facilities) -> 蓝色方块
    for idx in chosen_indices:
        pt = to_xy(locs[idx])
        # 画大一点的蓝色方块，加黑色边框使其醒目
        cv2.rectangle(canvas, (pt[0]-8, pt[1]-8), (pt[0]+8, pt[1]+8), (200, 100, 0), -1)
        cv2.rectangle(canvas, (pt[0]-8, pt[1]-8), (pt[0]+8, pt[1]+8), (50, 50, 50), 2)

    # 5. 绘制 Top-K 候选点 (Candidates) -> 绿色圆圈 + 标签
    for rank, cand in enumerate(top_candidates):
        pt = to_xy((cand['x'], cand['y']))
        label = get_label(rank) # A, B, C...
        
        # 画绿色空心圆圈，线宽2
        cv2.circle(canvas, pt, 10, (0, 180, 0), 2)
        
        # 绘制标签背景 (为了文字清晰)
        # text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        # cv2.rectangle(canvas, (pt[0]+8, pt[1]-18), (pt[0]+8+text_size[0], pt[1]-18+text_size[1]), (255,255,255), -1)

        # 绘制标签文字 (放在点的右上方)
        cv2.putText(canvas, label, (pt[0]+8, pt[1]-5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 0), 2)

    # 6. 转 Base64
    _, buffer = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    b64_str = base64.b64encode(buffer).decode('utf-8')

    if debug_save_path is not None:
        # 确保目录存在
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        # OpenCV 保存图片 (canvas 已经是 BGR 格式，直接存即可)
        cv2.imwrite(debug_save_path, canvas)
        print(f"[Debug] Image saved to: {debug_save_path}")
    
    return f"data:image/jpeg;base64,{b64_str}"

def get_diverse_top_k(candidates, dist_matrix, top_k, exclusion_radius=0.1):
    """
    使用贪心策略 + 空间抑制 (NMS) 选择 Top-K。
    
    Args:
        candidates: list of dict, 每个 dict 包含 'id', 'gain'/'sort_val' 等。
                    必须已经按优劣排序完毕（最好的在第一个）。
        dist_matrix: (N, N) numpy array, 全距离矩阵。
        top_k: int, 需要选出的数量。
        exclusion_radius: float, 互斥半径。在此半径内的次优解会被剔除。
                          (建议值为地图尺寸的 5%~10%，例如 0.05 ~ 0.1)
    
    Returns:
        selected_candidates: list of dict, 筛选后的 Top-K。
    """
    selected_candidates = []
    
    # 用一个 mask 标记哪些点已经被“抑制”或者是“已选”
    # 初始时，所有在 candidates 列表里的点都是可用的 (True)
    # 但为了快速索引，我们用 candidates 列表本身做遍历
    
    # 这里的 candidates 应该是全量的（或者数量远大于 top_k 的）排序后的列表
    # 为了效率，我们只处理前 50-100 个最好的，没必要处理几千个垃圾解
    pool = candidates[:100] 
    
    while len(selected_candidates) < top_k and len(pool) > 0:
        # 1. 既然已经排好序，pool[0] 就是当前最好的
        best_cand = pool.pop(0)
        selected_candidates.append(best_cand)
        
        # 2. 空间抑制：从 pool 中剔除靠得太近的点
        # best_cand['id'] 是它在 dist_matrix 里的索引
        best_id = best_cand['id']
        
        # 过滤 pool
        # 保留条件：距离 > exclusion_radius
        new_pool = []
        for cand in pool:
            cand_id = cand['id']
            dist = dist_matrix[best_id, cand_id]
            
            if dist > exclusion_radius:
                new_pool.append(cand)
            # else: 
            #   该点被抑制了（因为它离最优解太近，且收益不如最优解）
        
        pool = new_pool
        
    return selected_candidates

def build_obs_flp(td, env_num: int, top_k: int = 10) -> List[str]:
    obs_list = []

    # 1. 提取数据
    locs = _to_numpy(td["locs"])        
    chosen = _to_numpy(td["chosen"])    
    i_step = _to_numpy(td["i"])         
    
    # 获取目标 K 值
    if "to_choose" in td.keys():
        to_choose = _to_numpy(td["to_choose"])
    else:
        to_choose = np.full((env_num,), 3) # Fallback

    # Initialize tensor to store Top-K candidates for action projection
    # Shape: (B, K) filled with -1
    action_candidates = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)

    for idx in range(env_num):
        current_locs = locs[idx]       # (N, 2)
        current_mask = chosen[idx]     # (N,)
        step = i_step[idx].item() if hasattr(i_step[idx], "item") else i_step[idx]
        k_target = to_choose[idx].item() if hasattr(to_choose[idx], "item") else to_choose[idx]
        
        num_locs = current_locs.shape[0]
        
        # --- 2. 状态计算 ---
        dist_matrix = cdist(current_locs, current_locs, metric='euclidean')
        chosen_indices = np.where(current_mask == 1)[0]
        
        is_first_step = (len(chosen_indices) == 0)

        # 计算当前状态下的 Cost (Min Dists)
        if not is_first_step:
            dists_to_chosen = dist_matrix[:, chosen_indices]
            min_dists = np.min(dists_to_chosen, axis=1) # (N,) 每个点到最近设施的距离
            current_total_cost = np.sum(min_dists)
        else:
            # 第一步，还没选，视为无穷大
            min_dists = np.full(num_locs, np.inf)
            current_total_cost = np.inf

        # --- 3. 候选点评估 (计算收益) ---
        candidates = []
        unchosen_indices = np.where(current_mask == 0)[0]
        
        for cand_idx in unchosen_indices:
            dist_to_cand = dist_matrix[:, cand_idx]
            
            # 模拟：假如选了这个点，新的最小距离是多少？
            new_min_dists = np.minimum(min_dists, dist_to_cand)
            new_total_cost = np.sum(new_min_dists)
            
            cand_info = {
                "id": cand_idx,
                "x": current_locs[cand_idx][0],
                "y": current_locs[cand_idx][1],
                "new_cost": new_total_cost
            }

            if is_first_step:
                # 第一步：没有“减少量”，指标是绝对总成本 (越小越好)
                cand_info["sort_val"] = new_total_cost # 升序排序
                cand_info["desc"] = f"Expected Total Distance: {new_total_cost:.2f}"
            else:
                # 后续步：指标是减少量 (越大越好)
                reduction = current_total_cost - new_total_cost
                cand_info["sort_val"] = -reduction # 负号用于升序排序达到降序效果
                # *** 这里的文本直接告诉模型收益 ***
                cand_info["desc"] = f"Reduces Total Distance by: {reduction:.2f}"
            
            candidates.append(cand_info)

        # --- 4. 排序与生成文本 ---
        # 根据 sort_val 排序 (第一步是 Cost 升序，后续是 Reduction 降序)
        candidates.sort(key=lambda x: x["sort_val"])
        radius_threshold = 0.08
        top_candidates = get_diverse_top_k(
            candidates, 
            dist_matrix, # 之前计算好的 (N, N) 矩阵
            top_k, 
            exclusion_radius=radius_threshold
        )
        
        # Store indices for this batch
        indices = [c['id'] for c in top_candidates]
        if indices:
            action_candidates[idx, :len(indices)] = torch.tensor(indices, device=td.device)
        
        # --- 5. 可视化 ---
        # 渲染图像 (Base64 字符串)
        debug_path = None
        if idx == 0: # 只看第一个环境
             if step == 0 or step % 5 == 0: # 比如每 5 步存一张
                 debug_path = f"./debug_images/env0_step{step}.jpg"
        
        # 调用渲染函数，传入 debug_path
        img_b64 = render_flp_image(
            current_locs, 
            chosen_indices, 
            top_candidates, 
            debug_save_path=debug_path  # <--- 传入路径触发保存
        )

        cand_str_list = []
        for rank, cand in enumerate(top_candidates):
            label = get_label(rank)
            cand_str_list.append(
                f"Option {label} [Node {cand['id']}]: "
                f"**{cand['desc']}** " # 加粗收益部分
                f"(Coords: {cand['x']:.2f}, {cand['y']:.2f})"
            )
        cand_section = "\n".join(cand_str_list)

        # 状态描述
        if is_first_step:
            status_desc = "No facilities open yet."
        else:
            status_desc = (f"Open Facilities: [{', '.join(map(str, chosen_indices))}]\n"
                           f"Current Total Distance: {current_total_cost:.2f}")

        # --- 5. 最终 Obs 组装 ---
        obs = (
            f"### Task: Facility Location Problem\n"
            f"Step: {step + 1} / {k_target}\n" # 人类通常从1开始计数
            f"Status:\n{status_desc}\n\n"
            f"### Top {top_k} Candidates Analysis:\n"
            f"Here are the estimated outcomes for the best available locations:\n"
            f"{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the option that minimizes total distance. Return the Option Label (e.g., A, B, C)."
        )
        obs_list.append(obs)
        
    # Store candidates in TensorDict for action projection
    td["action_candidates"] = action_candidates
    td["topk_acts"] = action_candidates
    return obs_list

def build_obs_mclp(td, env_num: int, top_k: int = 10) -> List[str]:
    """
    Build structured observations for Maximal Covering Location Problem (MCLP).
    Key Logic: Geometric Pre-calculation -> Marginal Gain Analysis (New Coverage).
    """
    obs_list = []

    # 1. 提取基础数据
    chosen = _to_numpy(td["chosen"])             # (B, N_facilities)
    i_step = _to_numpy(td["i"])
    
    # 半径通常是标量或 Batch 标量
    coverage_radius = _to_numpy(td["coverage_radius"]) 
    
    # 目标数
    if "num_facilities_to_select" in td.keys():
        num_facilities_to_select = _to_numpy(td["num_facilities_to_select"])
    else:
        num_facilities_to_select = np.full((env_num,), 3) # Fallback

    # 提取位置信息
    fac_locs = _to_numpy(td["facility_locs"]) if "facility_locs" in td.keys() else _to_numpy(td["locs"])
    dem_locs = _to_numpy(td["demand_locs"]) if "demand_locs" in td.keys() else _to_numpy(td["locs"])
    
    # 提取权重 (Demand Weights)
    if "demand_weights" in td.keys():
        weights = _to_numpy(td["demand_weights"]) # (B, N_demand)
    else:
        weights = np.ones((env_num, dem_locs.shape[1])) 

    # Initialize tensor
    action_candidates = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)

    for idx in range(env_num):
        # --- 数据准备 ---
        current_fac_locs = fac_locs[idx]    # (N_fac, 2)
        current_dem_locs = dem_locs[idx]    # (N_dem, 2)
        current_weights = weights[idx]      # (N_dem,)
        current_chosen_mask = chosen[idx]   # (N_fac,)
        
        step = i_step[idx].item() if hasattr(i_step[idx], "item") else i_step[idx]
        total_steps = num_facilities_to_select[idx].item() if hasattr(num_facilities_to_select[idx], "item") else num_facilities_to_select[idx]
        
        radius = coverage_radius[idx].item() if hasattr(coverage_radius[idx], "item") else coverage_radius[idx]
        
        n_fac = current_fac_locs.shape[0]
        n_dem = current_dem_locs.shape[0]

        # --- 2. 几何预计算 (Critical Step) ---
        dist_matrix = cdist(current_fac_locs, current_dem_locs, metric='euclidean')
        coverage_matrix = (dist_matrix <= radius) # (N_fac, N_dem) Boolean

        # --- 3. 计算当前覆盖状态 ---
        chosen_indices = np.where(current_chosen_mask == 1)[0]
        
        if len(chosen_indices) > 0:
            covered_demand_mask = np.any(coverage_matrix[chosen_indices], axis=0)
        else:
            covered_demand_mask = np.zeros(n_dem, dtype=bool)

        current_covered_val = np.sum(current_weights[covered_demand_mask])
        total_val = np.sum(current_weights)
        progress_pct = (current_covered_val / total_val) * 100 if total_val > 0 else 0

        # --- 4. 计算 Top-K 候选点 (边际收益分析) ---
        candidates = []
        unchosen_fac_indices = np.where(current_chosen_mask == 0)[0]
        
        for fac_idx in unchosen_fac_indices:
            can_cover_mask = coverage_matrix[fac_idx] # (N_dem,)
            
            newly_covered_mask = can_cover_mask & (~covered_demand_mask)
            gain = np.sum(current_weights[newly_covered_mask])
            
            redundant_mask = can_cover_mask & covered_demand_mask
            redundancy = np.sum(current_weights[redundant_mask])
            
            num_new_points = np.sum(newly_covered_mask)
            
            candidates.append({
                "id": fac_idx,
                "gain": gain,
                "redundancy": redundancy,
                "num_new": num_new_points,
                "x": current_fac_locs[fac_idx][0],
                "y": current_fac_locs[fac_idx][1]
            })

        # --- 5. 排序与生成文本 ---
        candidates.sort(key=lambda x: x["gain"], reverse=True)
        top_candidates = candidates[:top_k]
        
        # Store indices
        indices = [c['id'] for c in top_candidates]
        if indices:
            action_candidates[idx, :len(indices)] = torch.tensor(indices, device=td.device)

        cand_str_list = []
        for rank, cand in enumerate(top_candidates):
            label = get_label(rank)
            if cand['redundancy'] > 0:
                note = f"(Overlaps {cand['redundancy']:.1f} existing weight)"
            else:
                note = "(Pure expansion: No overlap)"
                
            cand_str_list.append(
                f"Option {label} [Facility {cand['id']}]: "
                f"**Gain: {cand['gain']:.2f} New Weight** | "
                f"Coords: ({cand['x']:.2f}, {cand['y']:.2f}) | {note}"
            )
            
        cand_section = "\n".join(cand_str_list)
        
        chosen_str = ", ".join(map(str, chosen_indices)) if len(chosen_indices) > 0 else "None"

        # --- 6. 最终 Obs 组装 ---
        obs = (
            f"### Task: Maximum Covering Location Problem (MCLP)\n"
            f"Goal: Select {total_steps} facilities to maximize covered demand weight within Radius {radius:.4f}.\n"
            f"Step: {step + 1} / {total_steps}\n"
            f"Status:\n"
            f"- Open Facilities: [{chosen_str}]\n"
            f"- Current Coverage: {current_covered_val:.2f} / {total_val:.2f} ({progress_pct:.1f}%)\n\n"
            f"### Top {top_k} Candidate Analysis (Marginal Gain):\n"
            f"I have identified facilities that cover the most **previously unserved** demand:\n"
            f"{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Option Label that maximizes **Gain**. Return only the Option Label."
        )
        obs_list.append(obs)
        
    td["action_candidates"] = action_candidates
    td["topk_acts"] = action_candidates
    return obs_list

def build_obs_mcp(td, env_num: int, top_k: int = 10) -> List[str]:
    """
    Build structured observations for MCP.
    Key Logic: Calculate 'Marginal Gain' (Weight of NEW unique items covered).
    """
    obs_list = []
    
    # 1. 提取数据
    chosen = _to_numpy(td["chosen"])             # (B, N_sets) - binary mask
    i_step = _to_numpy(td["i"])
    
    # membership: (B, N_sets, Max_Set_Size) - 存储的是 Item Indices
    membership = _to_numpy(td["membership"])
    # weights: (B, N_items) - 每个 Item 的权重
    weights = _to_numpy(td["weights"])
    
    # 获取目标选择数 K
    if "n_sets_to_choose" in td.keys():
        n_sets_to_choose = _to_numpy(td["n_sets_to_choose"])
    else:
        n_sets_to_choose = np.full((env_num,), 5) # Fallback default

    # Initialize tensor
    action_candidates = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)

    for idx in range(env_num):
        # --- 数据准备 ---
        current_membership = membership[idx] # (N_sets, Max_Size)
        current_weights = weights[idx]       # (N_items,)
        current_chosen_mask = chosen[idx]    # (N_sets,)
        
        step = i_step[idx].item() if hasattr(i_step[idx], "item") else i_step[idx]
        total_steps = n_sets_to_choose[idx].item() if hasattr(n_sets_to_choose[idx], "item") else n_sets_to_choose[idx]
        
        n_sets = current_membership.shape[0]
        n_items = current_weights.shape[0]

        # --- 2. 计算当前覆盖状态 (Current Coverage) ---
        chosen_set_indices = np.where(current_chosen_mask == 1)[0]
        
        is_item_covered = np.zeros(n_items, dtype=bool)
        
        for s_idx in chosen_set_indices:
            items_in_set = current_membership[s_idx]
            valid_items = items_in_set[(items_in_set > 0) & (items_in_set < n_items)].astype(int)
            is_item_covered[valid_items] = True
            
        current_covered_weight = np.sum(current_weights[is_item_covered])
        total_possible_weight = np.sum(current_weights)
        progress_pct = (current_covered_weight / total_possible_weight) * 100 if total_possible_weight > 0 else 0

        # --- 3. 计算候选集合的边际收益 (Marginal Gain) ---
        candidates = []
        unchosen_set_indices = np.where(current_chosen_mask == 0)[0]
        
        for s_idx in unchosen_set_indices:
            items_in_set = current_membership[s_idx]
            valid_items = items_in_set[(items_in_set > 0) & (items_in_set < n_items)].astype(int)
            
            this_set_mask = np.zeros(n_items, dtype=bool)
            this_set_mask[valid_items] = True
            new_items_mask = this_set_mask & (~is_item_covered)
            
            gain = np.sum(current_weights[new_items_mask])
            raw_weight = np.sum(current_weights[valid_items])
            
            candidates.append({
                "id": s_idx,
                "gain": gain,
                "raw_weight": raw_weight,
                "overlap_loss": raw_weight - gain,
                "num_new_items": np.sum(new_items_mask)
            })

        # --- 4. 排序与生成 Top-K 文本 ---
        candidates.sort(key=lambda x: x["gain"], reverse=True)
        top_candidates = candidates[:top_k]
        
        # Store indices
        indices = [c['id'] for c in top_candidates]
        if indices:
            action_candidates[idx, :len(indices)] = torch.tensor(indices, device=td.device)

        cand_str_list = []
        for rank, cand in enumerate(top_candidates):
            label = get_label(rank)
            if cand['overlap_loss'] > 0:
                overlap_desc = f"(Total weight {cand['raw_weight']:.1f}, but {cand['overlap_loss']:.1f} is redundant)"
            else:
                overlap_desc = "(Perfect efficiency: No overlap)"
            
            cand_str_list.append(
                f"Option {label} [Set ID {cand['id']}]: "
                f"**Gain: {cand['gain']:.0f}** | "
                f"Covers {cand['num_new_items']} new items. {overlap_desc}"
            )
        
        cand_section = "\n".join(cand_str_list)
        
        chosen_str = ", ".join(map(str, chosen_set_indices)) if len(chosen_set_indices) > 0 else "None"

        # --- 5. 组装最终 Obs ---
        obs = (
            f"### Task: Maximum Coverage Problem (MCP)\n"
            f"Goal: Select {total_steps} sets to maximize the total weight of UNIQUE covered items.\n"
            f"Step: {step + 1} / {total_steps}\n"
            f"Status:\n"
            f"- Chosen Sets: [{chosen_str}]\n"
            f"- Current Covered Weight: {current_covered_weight} / {total_possible_weight} ({progress_pct:.1f}%)\n\n"
            f"### Top {top_k} Recommendations (Marginal Gain Analysis):\n"
            f"I have calculated the weight of **NEW, UNCOVERED** items each set would add:\n"
            f"{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Set ID (via Option Label) that provides the highest **Gain**. Return only the Option Label."
        )
        obs_list.append(obs)
        
    td["action_candidates"] = action_candidates
    td["topk_acts"] = action_candidates
    return obs_list

def build_obs_stp(td, env_num: int, top_k: int = 10) -> List[str]:
    """
    Build structured observations for Steiner Tree Problem (STP).
    Key Logic: Analyze Connected Components & Edge 'Bridge' Value.
    """
    obs_list = []
    
    # 1. 提取数据
    locs = _to_numpy(td["locs"])              # (B, N, 2)
    edge_list = _to_numpy(td["edge_list"])    # (B, M, 2) - M edges, pairs of (u, v)
    terminals = _to_numpy(td["terminals"])    # (B, T) - indices of terminals
    
    # 已选边的 Mask (0 or 1)
    if "selected_edge_indices" in td.keys():
        selected_mask = _to_numpy(td["selected_edge_indices"]) 
    else:
        selected_mask = np.zeros((env_num, edge_list.shape[1]))

    i_step = _to_numpy(td["i"]) if "i" in td.keys() else [0]*env_num

    # Initialize tensor
    action_candidates = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)

    for idx in range(env_num):
        # --- Context Setup ---
        curr_locs = locs[idx]
        curr_edges = edge_list[idx] # (M, 2)
        curr_terminals = terminals[idx]
        curr_mask = selected_mask[idx]
        step = i_step[idx].item() if hasattr(i_step[idx], "item") else i_step[idx]
        
        num_nodes = curr_locs.shape[0]
        
        # 2. 计算边权重 (Edge Weights) - Euclidean
        u_coords = curr_locs[curr_edges[:, 0].astype(int)]
        v_coords = curr_locs[curr_edges[:, 1].astype(int)]
        edge_weights = np.linalg.norm(u_coords - v_coords, axis=1)

        # 3. 分析当前连通性 (Connected Components)
        selected_indices = np.where(curr_mask == 1)[0]
        
        if len(selected_indices) > 0:
            sel_u = curr_edges[selected_indices, 0]
            sel_v = curr_edges[selected_indices, 1]
            data = np.ones(len(selected_indices))
            adj_matrix = csr_matrix((data, (sel_u, sel_v)), shape=(num_nodes, num_nodes))
        else:
            adj_matrix = csr_matrix((num_nodes, num_nodes)) 
            
        n_comps, labels = connected_components(adj_matrix, directed=False)
        
        # 4. 分析 Terminal 的分布状态
        term_groups = {} # label -> list of terminal indices
        for t in curr_terminals:
            l = labels[t]
            if l not in term_groups: term_groups[l] = []
            term_groups[l].append(t)
            
        max_connected_terminals = 0
        if term_groups:
            max_connected_terminals = max(len(g) for g in term_groups.values())
            
        total_terminals = len(curr_terminals)
        is_solved = (max_connected_terminals == total_terminals)

        # 5. 候选边评估 (Heuristic: Kruskal-like Logic)
        candidates = []
        unchosen_indices = np.where(curr_mask == 0)[0]
        
        for e_idx in unchosen_indices:
            u, v = int(curr_edges[e_idx, 0]), int(curr_edges[e_idx, 1])
            w = edge_weights[e_idx]
            
            lab_u = labels[u]
            lab_v = labels[v]
            
            if lab_u == lab_v:
                # 环 (Cycle) - 毫无价值
                priority = -1
                desc = "Redundant (Cycle)"
                gain_type = "None"
            else:
                has_term_u = lab_u in term_groups
                has_term_v = lab_v in term_groups
                
                if has_term_u and has_term_v:
                    priority = 100 - w # 权重越小越好
                    desc = f"**Merges 2 Terminal Groups** (Cost: {w:.2f})"
                    gain_type = "Critical"
                elif has_term_u or has_term_v:
                    priority = 50 - w
                    desc = f"Extends Terminal Group (Cost: {w:.2f})"
                    gain_type = "Expand"
                else:
                    priority = 0 - w
                    desc = f"Connects empty areas (Cost: {w:.2f})"
                    gain_type = "Low"
            
            candidates.append({
                "id": e_idx,
                "priority": priority,
                "desc": desc,
                "cost": w,
                "u": u, 
                "v": v,
                "type": gain_type
            })

        # 6. 排序与生成文本
        valid_candidates = [c for c in candidates if c['type'] != "None"]
        if not valid_candidates: 
             valid_candidates = candidates
             
        valid_candidates.sort(key=lambda x: x["priority"], reverse=True)
        top_candidates = valid_candidates[:top_k]
        
        # Store indices
        indices = [c['id'] for c in top_candidates]
        if indices:
            action_candidates[idx, :len(indices)] = torch.tensor(indices, device=td.device)

        cand_str_list = []
        for rank, cand in enumerate(top_candidates):
            label = get_label(rank)
            cand_str_list.append(
                f"Option {label} [Edge {cand['id']}]: "
                f"{cand['desc']} | Connects Node {cand['u']} <-> {cand['v']}"
            )
        cand_section = "\n".join(cand_str_list)
        
        current_cost = np.sum(edge_weights[selected_indices]) if len(selected_indices) > 0 else 0.0

        # 7. Obs 组装
        status_line = "SOLVED!" if is_solved else "In Progress"
        
        obs = (
            f"### Task: Steiner Tree Problem (STP)\n"
            f"Goal: Connect all {total_terminals} Terminals with minimum total edge weight.\n"
            f"Step: {step}\n"
            f"Status: {status_line}\n"
            f"- Connected Terminals: Max group has {max_connected_terminals} / {total_terminals} terminals.\n"
            f"- Current Total Weight: {current_cost:.2f}\n"
            f"- Disconnected Groups: {len(term_groups)} separate groups containing terminals.\n\n"
            f"### Top {top_k} Recommended Edges (Heuristic Analysis):\n"
            f"I have analyzed edges that merge separate components:\n"
            f"{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Edge ID (via Option Label) that merges Terminal groups with low cost. Return only the Option Label."
        )
        obs_list.append(obs)
        
    td["action_candidates"] = action_candidates
    td["topk_acts"] = action_candidates
    return obs_list
