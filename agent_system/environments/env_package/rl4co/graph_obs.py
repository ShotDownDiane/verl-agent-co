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


def render_flp_image(locs, chosen_indices, top_candidates, img_height=600, debug_save_path=None):
    """
    智能双视图渲染 (图层顺序修正版)：
    - 左图：全局视图 + 红框指示器
    - 右图：智能聚焦放大
    - 修正：候选点绘制顺序反转，确保 Option A (Rank 0) 在最上层，不被遮挡。
    """
    
    # --- 1. 参数配置 ---
    COLOR_BG = (255, 255, 255)
    COLOR_CUSTOMER = (190, 140, 80)      # Steel Blue
    COLOR_CHOSEN = (34, 34, 200)         # Deep Red
    COLOR_CONNECTION = (50, 160, 50)     # Green
    COLOR_CANDIDATE_BORDER = (50, 50, 50)
    COLOR_TEXT = (20, 20, 20)
    COLOR_ZOOM_BOX = (0, 0, 255)

    img_width = img_height * 2 
    combined_canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    left_roi = combined_canvas[:, :img_height] 
    right_roi = combined_canvas[:, img_height:]
    
    # --- 2. 坐标变换帮助函数 ---
    def get_transform(center, span, output_size, padding=40):
        half_span = span / 2.0
        min_xy = center - half_span
        max_xy = center + half_span
        
        available_size = output_size - 2 * padding
        scale = available_size / max(span, 1e-6)
        canvas_center = output_size / 2.0
        
        def transform_fn(coords):
            coords = np.array(coords)
            centered = coords - center
            scaled = centered * scale
            final = scaled.copy()
            final[..., 0] += canvas_center
            final[..., 1] = canvas_center - final[..., 1] 
            return final.astype(int)
            
        return transform_fn, (min_xy, max_xy)

    # --- 3. 计算全局视图 (Global View) ---
    all_points = locs
    if top_candidates:
        cand_coords = np.array([[c['x'], c['y']] for c in top_candidates])
        all_points = np.vstack([locs, cand_coords])
        
    g_min = np.min(all_points, axis=0)
    g_max = np.max(all_points, axis=0)
    g_center = (g_min + g_max) / 2.0
    g_span = np.max(g_max - g_min)
    
    global_transform, _ = get_transform(g_center, g_span, img_height, padding=50)

    # --- 4. 智能聚焦逻辑 ---
    if top_candidates:
        cand_points = np.array([[c['x'], c['y']] for c in top_candidates])
        focus_center = np.median(cand_points, axis=0)
        dists = np.linalg.norm(cand_points - focus_center, axis=1)
        
        if len(dists) > 1:
            radius = np.percentile(dists, 75)
            max_allowed_span = g_span * 0.35 
            min_allowed_span = g_span * 0.05
            zoom_span = max(min(radius * 2.5, max_allowed_span), min_allowed_span)
        else:
            zoom_span = g_span * 0.1

        zoom_transform, (z_real_min, z_real_max) = get_transform(focus_center, zoom_span, img_height, padding=40)
    else:
        zoom_transform = global_transform
        z_real_min, z_real_max = g_min, g_max

    # --- 5. 统一绘图函数 ---
    def draw_scene(canvas, transform_fn, view_bounds=None, is_zoomed=False):
        
        def is_visible(pt):
            if view_bounds is None: return True
            x, y = pt
            vmin, vmax = view_bounds
            return (x >= vmin[0]-0.1) and (x <= vmax[0]+0.1) and (y >= vmin[1]-0.1) and (y <= vmax[1]+0.1)

        # A. 连线
        if len(chosen_indices) > 0:
            chosen_locs = locs[chosen_indices]
            dists = cdist(locs, chosen_locs, metric='euclidean')
            nearest_idx = np.argmin(dists, axis=1)
            pts_start = transform_fn(locs)
            pts_end = transform_fn(chosen_locs[nearest_idx])
            
            for i in range(len(locs)):
                if i in chosen_indices: continue
                if is_visible(locs[i]):
                    cv2.line(canvas, tuple(pts_start[i]), tuple(pts_end[i]), COLOR_CONNECTION, 1, cv2.LINE_AA)

        # B. 客户点
        radius = 7 if is_zoomed else 3 
        pts = transform_fn(locs)
        for i, pt in enumerate(pts):
            if i in chosen_indices: continue
            if is_visible(locs[i]):
                cv2.circle(canvas, tuple(pt), radius, COLOR_CUSTOMER, -1, cv2.LINE_AA)

        # C. 已选设施
        box_s = 14 if is_zoomed else 6
        for idx in chosen_indices:
            if is_visible(locs[idx]):
                pt = tuple(transform_fn(locs[idx]))
                cv2.rectangle(canvas, (pt[0]-box_s, pt[1]-box_s), 
                              (pt[0]+box_s, pt[1]+box_s), (255,255,255), -1, cv2.LINE_AA)
                cv2.rectangle(canvas, (pt[0]-box_s+2, pt[1]-box_s+2), 
                              (pt[0]+box_s-2, pt[1]+box_s-2), COLOR_CHOSEN, -1, cv2.LINE_AA)

        # D. 候选设施 (图层修正)
        cand_s = 12 if is_zoomed else 8
        font_scale = 0.7 if is_zoomed else 0.45
        thickness = 2 if is_zoomed else 1
        
        # [Critical Fix]: 使用 reversed 反向遍历
        # 这样先画 rank=10, rank=9... 最后画 rank=0 (Option A)
        # 从而保证 A 永远在最上层
        # list(enumerate(top_candidates)) 将生成 [(0, candA), (1, candB)...]
        candidate_list = list(enumerate(top_candidates))
        
        for rank, cand in reversed(candidate_list):
            c_loc = np.array([cand['x'], cand['y']])
            if is_visible(c_loc):
                pt = tuple(transform_fn(c_loc))
                label = get_label(rank)
                
                # 绘制白底方块
                cv2.rectangle(canvas, (pt[0]-cand_s, pt[1]-cand_s), 
                              (pt[0]+cand_s, pt[1]+cand_s), (255,255,255), -1, cv2.LINE_AA)
                # 绘制边框
                cv2.rectangle(canvas, (pt[0]-cand_s, pt[1]-cand_s), 
                              (pt[0]+cand_s, pt[1]+cand_s), COLOR_CANDIDATE_BORDER, 2, cv2.LINE_AA)
                
                # 文字居中
                (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                text_x = pt[0] - text_w // 2
                text_y = pt[1] + text_h // 2 
                
                cv2.putText(canvas, label, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)

    # --- 6. 执行绘制 ---
    draw_scene(left_roi, global_transform, is_zoomed=False)
    
    if top_candidates:
        box_p1 = global_transform(z_real_min)
        box_p2 = global_transform(z_real_max)
        x1, y1 = min(box_p1[0], box_p2[0]), min(box_p1[1], box_p2[1])
        x2, y2 = max(box_p1[0], box_p2[0]), max(box_p1[1], box_p2[1])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_height-1, x2), min(img_height-1, y2)
        
        if x2 > x1 and y2 > y1:
            cv2.rectangle(left_roi, (x1, y1), (x2, y2), COLOR_ZOOM_BOX, 3, cv2.LINE_AA)
            cv2.line(combined_canvas, (x2, y1), (img_height, 0), COLOR_ZOOM_BOX, 1, cv2.LINE_AA)
            cv2.line(combined_canvas, (x2, y2), (img_height, img_height), COLOR_ZOOM_BOX, 1, cv2.LINE_AA)

    draw_scene(right_roi, zoom_transform, view_bounds=(z_real_min, z_real_max), is_zoomed=True)
    
    cv2.rectangle(right_roi, (0,0), (img_height-1, img_height-1), (180,180,180), 4)
    cv2.putText(left_roi, "Global View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,50,50), 2, cv2.LINE_AA)
    cv2.putText(right_roi, "Smart Zoom", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (50,50,50), 2, cv2.LINE_AA)

    # --- 7. 添加图例 (New Feature) ---
    def draw_legend(img):
        # 布局参数
        start_x, start_y = 20, img_height - 20
        line_height = 25
        font_scale = 0.5
        font_color = (60, 60, 60)
        
        def draw_item(y, text, draw_icon_fn):
            # 绘制图标
            draw_icon_fn(start_x, y - 8)
            # 绘制文字
            cv2.putText(img, text, (start_x + 20, y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1, cv2.LINE_AA)
            return y - line_height

        current_y = start_y
        
        # 1. Focus Area (红框)
        def icon_focus(x, y):
            cv2.rectangle(img, (x-6, y-5), (x+6, y+5), COLOR_ZOOM_BOX, 2, cv2.LINE_AA)
        current_y = draw_item(current_y, "Focus Area", icon_focus)
        
        # 2. Chosen Facility (红方块)
        def icon_chosen(x, y):
            cv2.rectangle(img, (x-5, y-5), (x+5, y+5), COLOR_CHOSEN, -1, cv2.LINE_AA)
        current_y = draw_item(current_y, "Chosen Facility", icon_chosen)
        
        # 3. Candidate (白方块)
        def icon_cand(x, y):
            cv2.rectangle(img, (x-5, y-5), (x+5, y+5), (255,255,255), -1, cv2.LINE_AA)
            cv2.rectangle(img, (x-5, y-5), (x+5, y+5), COLOR_CANDIDATE_BORDER, 2, cv2.LINE_AA)
        current_y = draw_item(current_y, "Candidate Facility", icon_cand)
        
        # 4. Connection (绿线)
        def icon_conn(x, y):
            cv2.line(img, (x-8, y), (x+8, y), COLOR_CONNECTION, 2, cv2.LINE_AA)
        current_y = draw_item(current_y, "Active Connection", icon_conn)
        
        # 5. Customer (蓝点)
        def icon_cust(x, y):
            cv2.circle(img, (x, y), 4, COLOR_CUSTOMER, -1, cv2.LINE_AA)
        current_y = draw_item(current_y, "Customer", icon_cust)

    # 在左图绘制图例
    draw_legend(left_roi)
    
    # --- 7. 输出 ---
    _, buffer = cv2.imencode('.png', combined_canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')

    if debug_save_path is not None:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, combined_canvas)
    
    return b64_str

def get_diverse_top_k(candidates, dist_matrix, top_k, exclusion_radius=0.08):
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
    pool = candidates 
    
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

def get_hybrid_top_k(candidates, dist_matrix, k, greedy_ratio=0.6):
    """
    混合采样策略：
    - 前 k * ratio 个：保留绝对收益最高的 (Greedy)
    - 后 k * (1-ratio) 个：从剩余中选空间差异最大的 (Diverse)
    """
    # 1. 先按收益排序
    # 假设 candidates 已经按 sort_val 排好序了 (最优在前)
    sorted_cands = sorted(candidates, key=lambda x: x["sort_val"]) # 升序或降序根据你的逻辑
    
    num_greedy = int(k * greedy_ratio)
    num_diverse = k - num_greedy
    
    # --- A. Greedy Part (保命符) ---
    final_selection = sorted_cands[:num_greedy]
    
    # 记录已选 ID 用于去重
    selected_ids = {c['id'] for c in final_selection}
    
    # --- B. Diverse Part (开眼界) ---
    # 从剩下的里面选
    remaining = [c for c in sorted_cands if c['id'] not in selected_ids]
    
    if num_diverse > 0 and remaining:
        # 使用你现有的 get_diverse_top_k 逻辑，但只选 num_diverse 个
        # 注意：这里需要传入已经选了的点作为 "exclusion_mask" 的基础，
        # 或者是简单地在 remaining 里跑多样性
        diverse_picks = get_diverse_top_k(
            remaining, 
            dist_matrix, 
            top_k=num_diverse, 
            exclusion_radius=0.1
        )
        final_selection.extend(diverse_picks)
    
    # 再次按收益排序，或者保持 Greedy 在前 Diverse 在后
    # 通常建议最后统一按收益排个序，让 Option A 是最好的
    final_selection.sort(key=lambda x: x["sort_val"]) 
    
    return final_selection

def build_obs_flp(td, env_num: int, top_k: int = 10, image_obs: bool = False, given_topk_acts: torch.Tensor = None) -> List[Any]:
    obs_list = []
    if "topk_acts" not in td.keys():
        td["topk_acts"] = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)

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
    if given_topk_acts is not None:
        given_topk_acts = _to_numpy(given_topk_acts)

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

        # 定义一个内部函数来计算收益，避免代码重复
        def calc_cand_info(c_idx):
            # 模拟选了这个点后的新 Cost
            dist_to_cand = dist_matrix[:, c_idx]
            new_min_dists = np.minimum(min_dists, dist_to_cand)
            new_total_cost = np.sum(new_min_dists)
            
            c_info = {
                "id": int(c_idx),
                "x": current_locs[c_idx][0],
                "y": current_locs[c_idx][1],
                "new_cost": new_total_cost
            }
            
            if is_first_step:
                # 第一步：指标是绝对 Cost (越小越好)
                c_info["sort_val"] = new_total_cost 
                c_info["desc"] = f"Expected Total Distance: {new_total_cost:.2f}"
            else:
                # 后续步：指标是 Reduction (越大越好)
                reduction = current_total_cost - new_total_cost
                c_info["sort_val"] = -reduction # 负号用于统一升序排序
                c_info["desc"] = f"Reduces Total Distance by: {reduction:.2f}"
            return c_info

        # 【分支 A: Injection 模式 (使用给定 acts)】
        top_candidates = []
        if given_topk_acts is not None:
            indices_to_use = given_topk_acts[idx]
            
            # 遍历给定的 ID，按顺序计算 info
            for cand_id in indices_to_use:
                # 如果是 padding 的 -1，跳过
                if cand_id == -1: continue
                
                # 计算并存入
                info = calc_cand_info(cand_id)
                top_candidates.append(info)
            
            # 这里不需要再排序，也不需要 diverse filter
            # 必须严格保留 SFT 数据注入的顺序 (例如最优解在最后)

        # 【分支 B: 自动模式 (Greedy Calculation)】
        else:
            candidates = []
            unchosen_indices = np.where(current_mask == 0)[0]
            
            for cand_idx in unchosen_indices:
                candidates.append(calc_cand_info(cand_idx))

            # 排序
            candidates.sort(key=lambda x: x["sort_val"])
            
            # 多样性筛选
            radius_threshold = 0.01
            top_candidates = get_hybrid_top_k(
                candidates, 
                dist_matrix, 
                top_k
            )
            
            # 更新 action_candidates (仅在自动模式下需要回写)
            indices = [c['id'] for c in top_candidates]
            if indices:
                # Pad to top_k with -1 if needed
                valid_len = len(indices)
                padded_indices = indices + [-1] * (top_k - valid_len)
                action_candidates[idx] = torch.tensor(padded_indices, device=td.device)
        
        # --- 5. 可视化 ---
        # 渲染图像 (Base64 字符串)
        img_b64 = None
        debug_path = None
        # 保持原有调试逻辑：第一个环境每 5 步存一张
        if idx == 0 and (step == 0 or step % 5 == 0):
             debug_path = f"./debug_images/flp/env0_step{step}.jpg"
        
        # 如果开启了 image_obs，或者需要保存调试图，就渲染
        if image_obs or debug_path:
            img_b64 = render_flp_image(
                current_locs, 
                chosen_indices, 
                top_candidates, 
                debug_save_path=debug_path
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
        obs_text = (
            f"### Task: Facility Location Problem\n"
            f"Step: {step + 1} / {k_target}\n" # 人类通常从1开始计数
            f"Status:\n{status_desc}\n\n"
            f"### Top {top_k} Candidates Analysis:\n"
            f"Here are the estimated outcomes for the best available locations:\n"
            f"{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the option that minimizes total distance. Return the Option Label (e.g., A, B, C)."
        )
        
        if image_obs and img_b64:
            # Return dict if image requested
            obs_list.append({
                "text": obs_text,
                "image": img_b64
            })
        else:
            # Return text only
            obs_list.append(obs_text)
        
    # Store candidates in TensorDict for action projection
    if given_topk_acts is None:
        td["topk_acts"] = action_candidates
        td["action_candidates"] = action_candidates # 兼容某些旧代码命名
    return obs_list


def render_mclp_image(fac_locs, dem_locs, chosen_indices, top_candidates, radius, img_size=800, debug_save_path=None):
    """
    MCLP 专用渲染：辐射范围视图 (Radiation Range View).
    
    Visual Logic:
    - 背景: 纯白
    - 覆盖圆 (Radius): 
        - 已选: 浅蓝色半透明填充 (Active Zone)
        - 候选: 绿色空心圆环 (Potential Zone)
    - 需求点 (Demand):
        - 红色: Uncovered (亟待解决)
        - 灰色: Covered (已解决)
    - 设施点: 方块 (已选实心，候选空心)
    """
    
    # --- 1. 配色方案 ---
    COLOR_BG = (255, 255, 255)
    
    # 状态色
    COLOR_DEMAND_UNCOVERED = (50, 50, 220)  # Bright Red (Alert)
    COLOR_DEMAND_COVERED = (200, 200, 200)  # Light Grey (Ignored)
    
    # 设施色
    COLOR_CHOSEN_FILL = (34, 34, 200)       # Deep Red
    COLOR_CANDIDATE_BORDER = (50, 50, 50)   # Black
    COLOR_TEXT = (20, 20, 20)
    
    # 覆盖圆色 (BGR)
    # Active Coverage: Light Blue-ish
    COLOR_COVERAGE_FILL = (230, 216, 173)   # Light Blue (Fill)
    COLOR_COVERAGE_BORDER = (120, 60, 20)  # Steel Blue (Border)
    
    # Potential Coverage: Green
    # COLOR_POTENTIAL_BORDER = (50, 180, 50)  # Green

    # --- 2. 坐标计算 ---
    # 画布初始化 (支持透明度需要先画在 overlay 上)
    base_canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    overlay = base_canvas.copy() # 用于画半透明圆
    
    padding = 60
    
    # 计算全图范围
    all_points_list = [fac_locs, dem_locs]
    if top_candidates:
        cand_coords = np.array([[c['x'], c['y']] for c in top_candidates])
        all_points_list.append(cand_coords)
    all_points = np.vstack(all_points_list)
        
    min_xy = np.min(all_points, axis=0)
    max_xy = np.max(all_points, axis=0)
    
    data_range = max_xy - min_xy
    data_center = (max_xy + min_xy) / 2.0
    max_range = max(np.max(data_range), 1e-6)
    
    available_size = img_size - 2 * padding
    scale = available_size / max_range
    canvas_center = img_size / 2.0
    
    def to_xy(coords):
        coords = np.array(coords)
        centered = coords - data_center
        scaled = centered * scale
        final = scaled.copy()
        # 居中 & Y轴翻转
        final[..., 0] += canvas_center
        final[..., 1] = canvas_center - final[..., 1]
        return final.astype(int)

    # 关键：计算像素单位的半径
    pixel_radius = int(radius * scale)

    # --- 3. 计算覆盖状态 (用于给需求点着色) ---
    num_dem = dem_locs.shape[0]
    is_covered = np.zeros(num_dem, dtype=bool)
    
    if len(chosen_indices) > 0:
        chosen_locs = fac_locs[chosen_indices]
        dists = cdist(dem_locs, chosen_locs, metric='euclidean')
        # 只要有一个设施距离 < radius，即为覆盖
        min_dists = np.min(dists, axis=1)
        is_covered = min_dists <= (radius + 1e-5)

    # --- 4. 绘图层级 1: 覆盖圆 (Circles) ---
    
    # A. 画已选设施的覆盖圆 (半透明填充)
    for idx in chosen_indices:
        pt = tuple(to_xy(fac_locs[idx]))
        # 实心填充圆 (画在 overlay)
        cv2.circle(overlay, pt, pixel_radius, COLOR_COVERAGE_FILL, -1, cv2.LINE_AA)
        # 边框 (画在 base)
        cv2.circle(base_canvas, pt, pixel_radius, COLOR_COVERAGE_BORDER, 1, cv2.LINE_AA)
        
    # B. 画候选设施的覆盖圆 (空心圆/虚线效果)
    # 注意：OpenCV 原生不支持虚线圆，用细实线代替，或者用特定颜色表示"Potential"
    # for cand in top_candidates:
    #     pt = tuple(to_xy([cand['x'], cand['y']]))
    #     # 画绿色空心圆，表示潜力
    #     cv2.circle(base_canvas, pt, pixel_radius, COLOR_POTENTIAL_BORDER, 2, cv2.LINE_AA)

    # 合并图层 (应用透明度)
    alpha = 0.2 # 覆盖层透明度
    cv2.addWeighted(overlay, alpha, base_canvas, 1 - alpha, 0, base_canvas)

    # --- 5. 绘图层级 2: 需求点 (Nodes) ---
    pts_dem = to_xy(dem_locs)
    
    # 批量绘制需要性能优化，或者直接循环
    for i in range(num_dem):
        pt = tuple(pts_dem[i])
        if is_covered[i]:
            # 已覆盖：灰色小点
            cv2.circle(base_canvas, pt, 3, COLOR_DEMAND_COVERED, -1, cv2.LINE_AA)
        else:
            # 未覆盖：红色醒目点
            cv2.circle(base_canvas, pt, 4, COLOR_DEMAND_UNCOVERED, -1, cv2.LINE_AA)

    # --- 6. 绘图层级 3: 设施与标签 ---
    
    # A. 已选设施 (红方块)
    box_s = 8
    for idx in chosen_indices:
        pt = tuple(to_xy(fac_locs[idx]))
        # 白描边 + 红实心
        cv2.rectangle(base_canvas, (pt[0]-box_s-2, pt[1]-box_s-2), 
                      (pt[0]+box_s+2, pt[1]+box_s+2), (255,255,255), -1, cv2.LINE_AA)
        cv2.rectangle(base_canvas, (pt[0]-box_s, pt[1]-box_s), 
                      (pt[0]+box_s, pt[1]+box_s), COLOR_CHOSEN_FILL, -1, cv2.LINE_AA)

    # B. 候选设施 (白方块 + 标签)
    cand_s = 10
    font_scale = 0.5
    thickness = 1
    
    # 反向遍历，保证 Top-1 在最上层
    for rank, cand in reversed(list(enumerate(top_candidates))):
        pt = tuple(to_xy([cand['x'], cand['y']]))
        label = get_label(rank)
        
        # 白底
        cv2.rectangle(base_canvas, (pt[0]-cand_s, pt[1]-cand_s), 
                      (pt[0]+cand_s, pt[1]+cand_s), (255,255,255), -1, cv2.LINE_AA)
        # 黑框
        cv2.rectangle(base_canvas, (pt[0]-cand_s, pt[1]-cand_s), 
                      (pt[0]+cand_s, pt[1]+cand_s), COLOR_CANDIDATE_BORDER, 2, cv2.LINE_AA)
        
        # 文字居中
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = pt[0] - text_w // 2
        text_y = pt[1] + text_h // 2 
        cv2.putText(base_canvas, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, thickness, cv2.LINE_AA)

    # --- 7. 图例 (Legend) ---
    def draw_legend(img):
        start_x, start_y = 20, img_size - 20
        line_height = 25
        font_scale = 0.5
        font_color = (60, 60, 60)
        
        def draw_item(y, text, draw_icon_fn):
            draw_icon_fn(start_x, y - 8)
            cv2.putText(img, text, (start_x + 25, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1, cv2.LINE_AA)
            return y - line_height

        current_y = start_y
        
        # Uncovered Demand
        current_y = draw_item(current_y, "Uncovered Demand", lambda x, y: cv2.circle(img, (x, y), 4, COLOR_DEMAND_UNCOVERED, -1, cv2.LINE_AA))
        # Covered Demand
        current_y = draw_item(current_y, "Covered Demand", lambda x, y: cv2.circle(img, (x, y), 3, COLOR_DEMAND_COVERED, -1, cv2.LINE_AA))
        # Potential Zone
        # current_y = draw_item(current_y, "Candidate Range", lambda x, y: cv2.circle(img, (x, y), 8, COLOR_POTENTIAL_BORDER, 2, cv2.LINE_AA))
        # Active Zone
        def icon_active(x, y):
            # 模拟半透明效果，这里直接画实色表示意思
            cv2.circle(img, (x, y), 8, COLOR_COVERAGE_BORDER, 1, cv2.LINE_AA)
            cv2.circle(img, (x, y), 7, COLOR_COVERAGE_FILL, -1, cv2.LINE_AA)
        current_y = draw_item(current_y, "Active Coverage", icon_active)
        
        current_y = draw_item(current_y, "Candidate", lambda x, y: (cv2.rectangle(img, (x-5, y-5), (x+5, y+5), (255,255,255), -1, cv2.LINE_AA), cv2.rectangle(img, (x-5, y-5), (x+5, y+5), COLOR_CANDIDATE_BORDER, 2, cv2.LINE_AA)))
        current_y = draw_item(current_y, "Open Facility", lambda x, y: cv2.rectangle(img, (x-5, y-5), (x+5, y+5), COLOR_CHOSEN_FILL, -1, cv2.LINE_AA))

    draw_legend(base_canvas)
    
    # 标题
    cv2.putText(base_canvas, "MCLP Coverage View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2, cv2.LINE_AA)

    # --- 8. 输出 ---
    _, buffer = cv2.imencode('.png', base_canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')

    if debug_save_path is not None:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, base_canvas)
    
    return b64_str

def build_obs_mclp(
    td, 
    env_num: int, 
    top_k: int = 10, 
    image_obs: bool = False,
    given_topk_acts = None
) -> List[Any]:
    """
    MCLP Observation Builder (Final Style).
    Features:
    - Smart Dual-View Rendering (Global + Zoom).
    - SFT Injection Support (given_topk_acts).
    - Marginal Gain Analysis.
    """
    obs_list = []

    # Init containers
    if "topk_acts" not in td.keys():
        td["topk_acts"] = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)
    
    if "action_candidates" not in td.keys():
         td["action_candidates"] = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)
         
    action_candidates = td["action_candidates"] # Get reference

    # 1. Extract Data
    chosen = _to_numpy(td["chosen"])             
    i_step = _to_numpy(td["i"])
    coverage_radius = _to_numpy(td["coverage_radius"]) 
    
    if "num_facilities_to_select" in td.keys():
        num_facilities_to_select = _to_numpy(td["num_facilities_to_select"])
    else:
        num_facilities_to_select = np.full((env_num,), 3)

    fac_locs = _to_numpy(td["facility_locs"]) if "facility_locs" in td.keys() else _to_numpy(td["locs"])
    dem_locs = _to_numpy(td["demand_locs"]) if "demand_locs" in td.keys() else _to_numpy(td["locs"])
    
    if "demand_weights" in td.keys():
        weights = _to_numpy(td["demand_weights"]) 
    else:
        weights = np.ones((env_num, dem_locs.shape[1])) 

    if given_topk_acts is not None:
        given_topk_acts = _to_numpy(given_topk_acts)

    for idx in range(env_num):
        # --- Context ---
        current_fac_locs = fac_locs[idx]    
        current_dem_locs = dem_locs[idx]    
        current_weights = weights[idx]      
        current_chosen_mask = chosen[idx]   
        
        step = i_step[idx].item() if hasattr(i_step[idx], "item") else i_step[idx]
        total_steps = num_facilities_to_select[idx].item() if hasattr(num_facilities_to_select[idx], "item") else num_facilities_to_select[idx]
        radius = coverage_radius[idx].item() if hasattr(coverage_radius[idx], "item") else coverage_radius[idx]
        
        n_dem = current_dem_locs.shape[0]

        # --- 2. Geometric Pre-calculation ---
        dist_matrix = cdist(current_fac_locs, current_dem_locs, metric='euclidean')
        coverage_matrix = (dist_matrix <= radius) 

        # --- 3. Current State ---
        chosen_indices = np.where(current_chosen_mask == 1)[0]
        
        if len(chosen_indices) > 0:
            covered_demand_mask = np.any(coverage_matrix[chosen_indices], axis=0)
        else:
            covered_demand_mask = np.zeros(n_dem, dtype=bool)

        current_covered_val = np.sum(current_weights[covered_demand_mask])
        total_val = np.sum(current_weights)
        progress_pct = (current_covered_val / total_val) * 100 if total_val > 0 else 0

        # --- Internal Function: Calculate Gain ---
        def calc_cand_info(f_idx):
            f_idx = int(f_idx)
            can_cover_mask = coverage_matrix[f_idx] 
            
            newly_covered_mask = can_cover_mask & (~covered_demand_mask)
            gain = np.sum(current_weights[newly_covered_mask])
            
            redundant_mask = can_cover_mask & covered_demand_mask
            redundancy = np.sum(current_weights[redundant_mask])
            
            return {
                "id": f_idx,
                "gain": gain,
                "redundancy": redundancy,
                "x": current_fac_locs[f_idx][0],
                "y": current_fac_locs[f_idx][1]
            }

        # --- 4. Candidate Generation ---
        top_candidates = []

        # [Branch A: Injection]
        if given_topk_acts is not None:
            indices_to_use = given_topk_acts[idx]
            for cand_id in indices_to_use:
                if cand_id == -1: continue
                top_candidates.append(calc_cand_info(cand_id))

        # [Branch B: Auto-Greedy]
        else:
            candidates = []
            unchosen_fac_indices = np.where(current_chosen_mask == 0)[0]
            
            for fac_idx in unchosen_fac_indices:
                candidates.append(calc_cand_info(fac_idx))

            # Sort by Gain
            candidates.sort(key=lambda x: x["gain"], reverse=True)
            top_candidates = candidates[:top_k]
            
            # Update tensor
            indices = [c['id'] for c in top_candidates]
            if indices:
                valid_len = len(indices)
                padded_indices = indices + [-1] * (top_k - valid_len)
                action_candidates[idx] = torch.tensor(padded_indices, device=td.device)

        # --- 5. Visualization ---
        img_b64 = None
        # Debug logic or Image Obs logic
        debug_path = None
        if idx == 0 and (step == 0 or step % 5 == 0):
             debug_path = f"./debug_images/mclp/env0_step{step}.png"
            #  pass

        if image_obs or debug_path:
            img_b64 = render_mclp_image(
                current_fac_locs, current_dem_locs, chosen_indices, top_candidates, radius,
                debug_save_path=debug_path
            )

        # --- 6. Text Prompt ---
        cand_str_list = []
        for rank, cand in enumerate(top_candidates):
            label = chr(ord('A') + rank)
            if cand['redundancy'] > 0:
                note = f"(Overlaps {cand['redundancy']:.1f})"
            else:
                note = "(No overlap)"
                
            cand_str_list.append(
                f"Option {label} [Facility {cand['id']}]: "
                f"**Gain: {cand['gain']:.2f}** | "
                f"Coords: ({cand['x']:.2f}, {cand['y']:.2f}) | {note}"
            )
        cand_section = "\n".join(cand_str_list)
        chosen_str = ", ".join(map(str, chosen_indices)) if len(chosen_indices) > 0 else "None"

        obs_text = (
            f"### Task: Maximum Covering Location Problem (MCLP)\n"
            f"Goal: Select {total_steps} facilities to maximize covered demand weight within Radius {radius:.4f}.\n"
            f"Step: {step + 1} / {total_steps}\n"
            f"Status:\n"
            f"- Open Facilities: [{chosen_str}]\n"
            f"- Current Coverage: {current_covered_val:.2f} / {total_val:.2f} ({progress_pct:.1f}%)\n\n"
            f"### Top Candidates Analysis (Marginal Gain):\n"
            f"Refer to the 'Smart Zoom' view (right) for details on the Focus Area (red box in Global View).\n"
            f"I have analyzed the potential coverage gain for the recommended facilities:\n"
            f"{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Option Label that maximizes **Gain**. Return only the Option Label."
        )

        if image_obs and img_b64:
            obs_list.append({"text": obs_text, "image": img_b64})
        else:
            obs_list.append(obs_text)
            
    if given_topk_acts is None:
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


def render_stp_image(
    locs, edge_list, terminals, selected_edge_indices, top_candidates, 
    img_height=600, debug_save_path=None
):
    """
    STP 智能双视图渲染 (Layer Order Fixed & Border Fix).
    
    Changes:
    - Ensures the border around the right zoom view is drawn correctly by
      drawing directly on the combined canvas.
    - Layer Order: Tree Edges -> Nodes -> Candidate Edges (Topmost).
    - Ensures Candidate Labels (A, B...) are never occluded by nodes.
    """
    
    # --- 1. 参数与配色 ---
    COLOR_BG = (255, 255, 255)
    
    # 节点
    COLOR_TERMINAL_FILL = (34, 34, 200)    # Deep Red
    COLOR_STEINER_FILL = (255, 255, 255)   # White
    COLOR_NODE_BORDER = (20, 20, 20)       # Black
    
    # 边
    COLOR_TREE_EDGE = (220, 100, 50)       # Royal Blue
    COLOR_CANDIDATE_EDGE = (50, 180, 50)   # Forest Green
    
    # 辅助
    COLOR_TEXT = (10, 10, 10)
    COLOR_ZOOM_BOX = (0, 0, 255) 
    COLOR_BORDER = (180, 180, 180)         # Grey for UI border

    img_width = img_height * 2 
    combined_canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    left_roi = combined_canvas[:, :img_height] 
    right_roi = combined_canvas[:, img_height:]
    
    term_set = set(terminals) if isinstance(terminals, (list, np.ndarray)) else set()

    # --- 2. 坐标变换 ---
    def get_transform(center, span, output_size, padding=40):
        half_span = span / 2.0
        min_xy = center - half_span
        max_xy = center + half_span
        
        available_size = output_size - 2 * padding
        scale = available_size / max(span, 1e-6)
        canvas_center = output_size / 2.0
        
        def transform_fn(coords):
            coords = np.array(coords)
            centered = coords - center
            scaled = centered * scale
            final = scaled.copy()
            final[..., 0] += canvas_center
            final[..., 1] = canvas_center - final[..., 1] 
            return final.astype(int)
        return transform_fn, (min_xy, max_xy)

    # --- 3. 全局视图 ---
    g_min = np.min(locs, axis=0)
    g_max = np.max(locs, axis=0)
    g_center = (g_min + g_max) / 2.0
    g_span = np.max(g_max - g_min)
    
    global_transform, _ = get_transform(g_center, g_span, img_height, padding=60)

    # --- 4. 智能聚焦 ---
    if top_candidates:
        cand_points = []
        for cand in top_candidates:
            cand_points.append(locs[cand['u']])
            cand_points.append(locs[cand['v']])
        cand_points = np.array(cand_points)
        
        focus_center = np.median(cand_points, axis=0)
        dists = np.linalg.norm(cand_points - focus_center, axis=1)
        
        if len(dists) > 1:
            radius = np.percentile(dists, 80)
            max_allowed_span = g_span * 0.40 
            min_allowed_span = g_span * 0.10
            zoom_span = max(min(radius * 2.5, max_allowed_span), min_allowed_span)
        else:
            zoom_span = g_span * 0.15

        zoom_transform, (z_real_min, z_real_max) = get_transform(focus_center, zoom_span, img_height, padding=40)
    else:
        zoom_transform = global_transform
        z_real_min, z_real_max = g_min, g_max

    # --- 5. 绘图函数 ---
    def draw_scene(canvas, transform_fn, view_bounds=None, is_zoomed=False):
        
        def is_visible(pt):
            if view_bounds is None: return True
            x, y = pt
            vmin, vmax = view_bounds
            return (x >= vmin[0]-0.05) and (x <= vmax[0]+0.05) and (y >= vmin[1]-0.05) and (y <= vmax[1]+0.05)

        pts = transform_fn(locs)
        
        # === Layer 1: Tree Edges (Bottom) ===
        sel_indices = np.where(selected_edge_indices == 1)[0]
        line_width_tree = 4 if is_zoomed else 2
        
        for e_idx in sel_indices:
            u, v = int(edge_list[e_idx, 0]), int(edge_list[e_idx, 1])
            if is_visible(locs[u]) or is_visible(locs[v]):
                cv2.line(canvas, tuple(pts[u]), tuple(pts[v]), COLOR_TREE_EDGE, line_width_tree, cv2.LINE_AA)

        # === Layer 2: Nodes (Middle) ===
        term_size = 12 if is_zoomed else 7
        steiner_radius = 6 if is_zoomed else 4
        id_font_scale = 0.5 if is_zoomed else 0.35
        
        for i in range(len(locs)):
            if not is_visible(locs[i]): continue
            pt = tuple(pts[i])
            
            if i in term_set:
                # Terminal
                cv2.rectangle(canvas, (pt[0]-term_size-2, pt[1]-term_size-2), 
                              (pt[0]+term_size+2, pt[1]+term_size+2), (255,255,255), -1, cv2.LINE_AA)
                cv2.rectangle(canvas, (pt[0]-term_size, pt[1]-term_size), 
                              (pt[0]+term_size, pt[1]+term_size), COLOR_TERMINAL_FILL, -1, cv2.LINE_AA)
                # ID
                id_str = str(i)
                (w, h), _ = cv2.getTextSize(id_str, cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, 1)
                cv2.putText(canvas, id_str, (pt[0] - w//2, pt[1] + h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, id_font_scale, (255,255,255), 1, cv2.LINE_AA)
            else:
                # Steiner
                cv2.circle(canvas, pt, steiner_radius, (255,255,255), -1, cv2.LINE_AA)
                cv2.circle(canvas, pt, steiner_radius, COLOR_NODE_BORDER, 1, cv2.LINE_AA)

        # === Layer 3: Candidate Edges & Labels (Top) ===
        cand_line_width = 3 if is_zoomed else 2
        cand_label_box = 14 if is_zoomed else 10
        font_scale = 0.7 if is_zoomed else 0.5
        
        candidate_list = list(enumerate(top_candidates))
        for rank, cand in reversed(candidate_list):
            u, v = cand['u'], cand['v']
            if is_visible(locs[u]) or is_visible(locs[v]):
                pt1, pt2 = tuple(pts[u]), tuple(pts[v])
                
                # 1. 绿线
                cv2.line(canvas, pt1, pt2, COLOR_CANDIDATE_EDGE, cand_line_width, cv2.LINE_AA)
                
                # 2. 标签
                mid_x = int((pt1[0] + pt2[0]) / 2)
                mid_y = int((pt1[1] + pt2[1]) / 2)
                label = get_label(rank)
                
                cv2.rectangle(canvas, (mid_x-cand_label_box, mid_y-cand_label_box), 
                              (mid_x+cand_label_box, mid_y+cand_label_box), (255,255,255), -1, cv2.LINE_AA)
                cv2.rectangle(canvas, (mid_x-cand_label_box, mid_y-cand_label_box), 
                              (mid_x+cand_label_box, mid_y+cand_label_box), (20,20,20), 1, cv2.LINE_AA)
                
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
                text_x = mid_x - w // 2
                text_y = mid_y + h // 2
                cv2.putText(canvas, label, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, 2, cv2.LINE_AA)

    # --- 6. 执行绘制 ---
    draw_scene(left_roi, global_transform, is_zoomed=False)
    
    if top_candidates:
        box_p1 = global_transform(z_real_min)
        box_p2 = global_transform(z_real_max)
        x1, y1 = min(box_p1[0], box_p2[0]), min(box_p1[1], box_p2[1])
        x2, y2 = max(box_p1[0], box_p2[0]), max(box_p1[1], box_p2[1])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_height-1, x2), min(img_height-1, y2)
        
        if x2 > x1 and y2 > y1:
            cv2.rectangle(left_roi, (x1, y1), (x2, y2), COLOR_ZOOM_BOX, 2, cv2.LINE_AA)
            cv2.line(combined_canvas, (x2, y1), (img_height, 0), COLOR_ZOOM_BOX, 1, cv2.LINE_AA)
            cv2.line(combined_canvas, (x2, y2), (img_height, img_height), COLOR_ZOOM_BOX, 1, cv2.LINE_AA)

    draw_scene(right_roi, zoom_transform, view_bounds=(z_real_min, z_real_max), is_zoomed=True)

    # --- 7. 图例 & UI ---
    def draw_legend(img):
        start_x, start_y = 20, img_height - 20
        line_height = 25
        font_scale = 0.5
        font_color = (60, 60, 60)
        
        def draw_item(y, text, draw_icon_fn):
            draw_icon_fn(start_x, y - 8)
            cv2.putText(img, text, (start_x + 25, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_color, 1, cv2.LINE_AA)
            return y - line_height

        current_y = start_y
        current_y = draw_item(current_y, "Terminal Node", lambda x, y: cv2.rectangle(img, (x-6, y-6), (x+6, y+6), COLOR_TERMINAL_FILL, -1, cv2.LINE_AA))
        current_y = draw_item(current_y, "Steiner Node", lambda x, y: (cv2.circle(img, (x, y), 6, (255,255,255), -1, cv2.LINE_AA), cv2.circle(img, (x, y), 6, COLOR_NODE_BORDER, 1, cv2.LINE_AA)))
        current_y = draw_item(current_y, "Connected Tree", lambda x, y: cv2.line(img, (x-10, y), (x+10, y), COLOR_TREE_EDGE, 3, cv2.LINE_AA))
        def icon_cand(x, y):
            cv2.line(img, (x-10, y), (x+10, y), COLOR_CANDIDATE_EDGE, 2, cv2.LINE_AA)
            cv2.rectangle(img, (x-3, y-3), (x+3, y+3), (255,255,255), -1)
            cv2.rectangle(img, (x-3, y-3), (x+3, y+3), (20,20,20), 1)
        current_y = draw_item(current_y, "Candidate Edge", icon_cand)
        current_y = draw_item(current_y, "Focus Area", lambda x, y: cv2.rectangle(img, (x-8, y-6), (x+8, y+6), COLOR_ZOOM_BOX, 2, cv2.LINE_AA))

    draw_legend(left_roi)
    
    # [FIX]: 直接在 combined_canvas 上绘制右侧边框
    cv2.rectangle(combined_canvas, (img_height, 0), (img_width-1, img_height-1), COLOR_BORDER, 4)
    
    cv2.putText(left_roi, "Global View (STP)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2, cv2.LINE_AA)
    cv2.putText(right_roi, "Smart Zoom", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2, cv2.LINE_AA)

    # --- 8. 输出 ---
    _, buffer = cv2.imencode('.png', combined_canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')

    if debug_save_path is not None:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, combined_canvas)
    
    return b64_str

def build_obs_stp(
    td, 
    env_num: int, 
    top_k: int = 10,
    image_obs: bool = False,   # 新增
    given_topk_acts = None
) -> List[Any]:
    
    obs_list = []
    
    # Init containers
    if "topk_acts" not in td.keys():
        td["topk_acts"] = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)
    if "action_candidates" not in td.keys():
         td["action_candidates"] = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)
    
    action_candidates = td["action_candidates"]

    # 1. 提取数据
    locs = _to_numpy(td["locs"])              
    edge_list = _to_numpy(td["edge_list"])    
    terminals = _to_numpy(td["terminals"])    
    
    if "selected_edge_indices" in td.keys():
        selected_mask = _to_numpy(td["selected_edge_indices"]) 
    else:
        selected_mask = np.zeros((env_num, edge_list.shape[1]))

    i_step = _to_numpy(td.get("i", [0]*env_num))

    if given_topk_acts is not None:
        given_topk_acts = _to_numpy(given_topk_acts)

    for idx in range(env_num):
        curr_locs = locs[idx]         
        curr_edges = edge_list[idx]   
        curr_terminals = terminals[idx]
        curr_mask = selected_mask[idx]
        step = i_step[idx].item() if hasattr(i_step[idx], "item") else i_step[idx]
        num_nodes = curr_locs.shape[0]
        
        u_idx = curr_edges[:, 0].astype(int)
        v_idx = curr_edges[:, 1].astype(int)
        edge_weights = np.linalg.norm(curr_locs[u_idx] - curr_locs[v_idx], axis=1)
        
        selected_indices = np.where(curr_mask == 1)[0]
        if len(selected_indices) > 0:
            sel_u = curr_edges[selected_indices, 0]
            sel_v = curr_edges[selected_indices, 1]
            data = np.ones(len(selected_indices))
            adj_matrix = csr_matrix((data, (sel_u, sel_v)), shape=(num_nodes, num_nodes))
        else:
            adj_matrix = csr_matrix((num_nodes, num_nodes)) 
        n_comps, labels = connected_components(adj_matrix, directed=False)

        term_groups = {} 
        for t in curr_terminals:
            t = int(t)
            l = labels[t]
            if l not in term_groups: term_groups[l] = []
            term_groups[l].append(t)
            
        group_coords_map = {}
        for l, t_indices in term_groups.items():
            group_coords_map[l] = curr_locs[t_indices] 
            
        max_connected_terminals = max(len(g) for g in term_groups.values()) if term_groups else 0
        total_terminals = len(curr_terminals)
        is_solved = (max_connected_terminals == total_terminals)

        # --- Evaluate Edge Function (Same as before) ---
        def evaluate_edge(e_idx):
            # ... (Copied from previous strict version logic) ...
            e_idx = int(e_idx)
            u, v = int(curr_edges[e_idx, 0]), int(curr_edges[e_idx, 1])
            w = edge_weights[e_idx]
            lab_u, lab_v = labels[u], labels[v]
            
            if lab_u == lab_v:
                return {"id": e_idx, "u": u, "v": v, "priority": -999, "desc": "Redundant", "type": "None"}

            has_term_u = lab_u in term_groups
            has_term_v = lab_v in term_groups
            
            if has_term_u and has_term_v:
                priority = 100 - w
                desc = f"**MERGE GROUPS**: Comp {lab_u} <-> Comp {lab_v} (Cost: {w:.2f})"
                gain_type = "Critical"
            elif has_term_u or has_term_v:
                # Heuristic Logic
                active_lab, new_node_idx = (lab_u, v) if has_term_u else (lab_v, u)
                min_dist_to_other = float('inf')
                for other_lab, other_coords in group_coords_map.items():
                    if other_lab == active_lab: continue
                    dists = np.linalg.norm(other_coords - curr_locs[new_node_idx], axis=1)
                    min_dist_to_other = min(min_dist_to_other, np.min(dists))
                h_score = 0.0 if min_dist_to_other == float('inf') else min_dist_to_other
                priority = 50 - (w + h_score)
                desc = f"EXTEND: towards nearest group (Edge: {w:.2f} + Dist: {h_score:.2f})"
                gain_type = "Expand"
            else:
                priority = 0 - w
                desc = f"Connects empty areas (Cost: {w:.2f})"
                gain_type = "Low"
            return {"id": e_idx, "u": u, "v": v, "priority": priority, "desc": desc, "type": gain_type}

        # --- Candidate Generation ---
        top_candidates = []
        if given_topk_acts is not None:
            indices_to_use = given_topk_acts[idx]
            for e_idx in indices_to_use:
                if e_idx == -1: continue 
                top_candidates.append(evaluate_edge(e_idx))
        else:
            candidates = []
            unchosen_indices = np.where(curr_mask == 0)[0]
            for e_idx in unchosen_indices:
                candidates.append(evaluate_edge(e_idx))
            valid_candidates = [c for c in candidates if c['type'] != "None"]
            if not valid_candidates: valid_candidates = candidates
            valid_candidates.sort(key=lambda x: x["priority"], reverse=True)
            top_candidates = valid_candidates[:top_k]
            
            indices = [c['id'] for c in top_candidates]
            if indices:
                valid_len = len(indices)
                padded_indices = indices + [-1] * (top_k - valid_len)
                action_candidates[idx] = torch.tensor(padded_indices, device=td.device)

        # --- Visualization ---
        img_b64 = None
        debug_path = None
        if idx == 0 and (step == 0 or step % 5 == 0):
             debug_path = f"./debug_images/stp/env0_step{step}.png"
            #  pass

        if image_obs or debug_path:
            img_b64 = render_stp_image(
                curr_locs, curr_edges, curr_terminals, curr_mask, top_candidates,
                debug_save_path=debug_path
            )

        # --- Text Prompt ---
        cand_str_list = []
        for rank, cand in enumerate(top_candidates):
            label = get_label(rank)
            cand_str_list.append(
                f"Option {label} [Edge {cand['id']}]: "
                f"{cand['desc']} | Connects Node {cand['u']} <-> {cand['v']}"
            )
        cand_section = "\n".join(cand_str_list)
        
        current_cost = np.sum(edge_weights[selected_indices]) if len(selected_indices) > 0 else 0.0
        status_line = "SOLVED!" if is_solved else "In Progress"
        
        obs_text = (
            f"### Task: Steiner Tree Problem (STP)\n"
            f"Goal: Connect all {total_terminals} Terminals (Red Squares) with minimum total edge weight.\n"
            f"Step: {step}\n"
            f"Status: {status_line}\n"
            f"- Connected Terminals: Max group has {max_connected_terminals} / {total_terminals} terminals.\n"
            f"- Current Total Weight: {current_cost:.2f}\n"
            f"- Disconnected Groups: {len(term_groups)}\n\n"
            f"### Top Recommended Edges (Heuristic Analysis):\n"
            f"Looking at the image, I have marked candidate edges in Green with labels (A, B...).\n"
            f"I have prioritized edges that merge groups or extend towards other terminals:\n"
            f"{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Edge ID (via Option Label) that merges Terminal groups or efficiently extends towards them. Return only the Option Label."
        )

        if image_obs and img_b64:
            obs_list.append({"text": obs_text, "image": img_b64})
        else:
            obs_list.append(obs_text)
        
    if given_topk_acts is None:
        td["topk_acts"] = action_candidates

    return obs_list