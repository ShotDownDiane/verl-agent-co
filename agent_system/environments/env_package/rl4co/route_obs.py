from typing import List, Optional
import os
import cv2
import base64
import torch
import numpy as np
from tensordict.tensordict import TensorDict
from scipy.spatial.distance import cdist
import math
import uuid

def apply_angular_masking(
    candidates: list,
    current_pos: np.ndarray,
    angle_threshold_deg: float = 10.0,
    always_keep_ids: set = None,
    coordinate_keys: tuple = ('x', 'y')
) -> list:
    """
    通用角度遮蔽函数 (适用于 CVRP 和 TSP)。
    
    逻辑：
    1. 计算所有候选点相对于 current_pos 的距离和角度。
    2. 按距离从小到大排序。
    3. 保留最近的点，遮蔽其后方同角度扇区内的点。
    4. 特殊点 (always_keep_ids) 总是保留且不参与遮挡计算（透明）。

    Args:
        candidates: 候选点字典列表，必须包含坐标信息。
        current_pos: 当前位置 [x, y]。
        angle_threshold_deg: 遮蔽扇区角度 (默认 20度)。
        always_keep_ids: 不受遮蔽影响的 ID 集合 (如 CVRP 的 Depot: {0})。
        coordinate_keys: 字典中坐标的键名，默认 ('x', 'y')。

    Returns:
        filtered_candidates: 筛选后的列表 (顺序可能根据距离重排)。
    """
    if not candidates:
        return []

    if always_keep_ids is None:
        always_keep_ids = set()

    # 1. 预计算几何信息
    threshold_rad = math.radians(angle_threshold_deg)
    enhanced_cands = []
    
    kx, ky = coordinate_keys

    for cand in candidates:
        # 获取坐标 (兼容不同命名)
        c_x, c_y = cand[kx], cand[ky]
        
        dx = c_x - current_pos[0]
        dy = c_y - current_pos[1]
        
        # 实时计算距离和角度 (确保独立性)
        dist = math.sqrt(dx**2 + dy**2)
        angle = math.atan2(dy, dx) # (-pi, pi)
        
        enhanced_cands.append({
            "original_data": cand,
            "dist": dist,
            "angle": angle,
            "id": cand.get('id')
        })

    # 2. 核心排序：距离优先
    # 这是遮蔽逻辑成立的前提：必须先看到近的，才能决定是否遮蔽远的
    enhanced_cands.sort(key=lambda x: x['dist'])

    # 3. 执行遮蔽
    final_list = []
    accepted_angles = [] # 存储已接受点的角度（仅限非特殊点）

    for item in enhanced_cands:
        cand_id = item['id']
        curr_angle = item['angle']
        
        # A. 特殊点处理 (透传)
        # 例如 CVRP 的 Depot，或者 TSP 的 Start Node
        # 它们既不会被遮蔽，也不会遮蔽别人 (透明)
        if cand_id in always_keep_ids:
            final_list.append(item['original_data'])
            continue

        # B. 遮蔽判定
        is_masked = False
        for acc_angle in accepted_angles:
            # 计算最小角度差 (处理圆周跨越)
            diff = abs(curr_angle - acc_angle)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            
            if diff < threshold_rad:
                is_masked = True
                break
        
        # C. 接受逻辑
        if not is_masked:
            final_list.append(item['original_data'])
            # 只有普通点才会被加入“遮挡源”列表
            accepted_angles.append(curr_angle)

    return final_list

def get_spatial_desc(dx, dy):
    """
    将向量转换为方位描述。
    假设标准笛卡尔坐标系：Y+ 为北，X+ 为东。
    """
    if abs(dx) < 1e-4 and abs(dy) < 1e-4:
        return "Self"
    
    angle = math.degrees(math.atan2(dy, dx))
    # 归一化到 [0, 360)
    if angle < 0: angle += 360
    
    # 简单的八向切分
    if 22.5 <= angle < 67.5: return "North-East (↗)"
    elif 67.5 <= angle < 112.5: return "North (↑)"
    elif 112.5 <= angle < 157.5: return "North-West (↖)"
    elif 157.5 <= angle < 202.5: return "West (←)"
    elif 202.5 <= angle < 247.5: return "South-West (↙)"
    elif 247.5 <= angle < 292.5: return "South (↓)"
    elif 292.5 <= angle < 337.5: return "South-East (↘)"
    else: return "East (→)"

def _to_numpy(x):
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

def _get_topk_str(td: TensorDict, i: int, actions: List[List[int]], return_topk_options: bool) -> str:
    """Helper to generate Top-K options string if applicable."""
    if not return_topk_options or actions is None or len(actions) == 0:
        return ""
    
    if "topk_acts" not in td.keys() or "topk_costs" not in td.keys():
        return ""
        
    topk_acts_list = td["topk_acts"].tolist()
    topk_costs_list = td["topk_costs"].tolist()
    
    options_str = "\nTop candidates based on distance:\n"
    opts_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
    
    b_acts = topk_acts_list[i]
    b_costs = topk_costs_list[i]
    
    valid_opts = []
    for idx, (act, cost) in enumerate(zip(b_acts, b_costs)):
        if cost == float('inf'):
            continue
        
        label = opts_labels[idx] if idx < len(opts_labels) else str(idx+1)
        valid_opts.append(f"{label}. Node {act} (Distance: {cost:.3f})")
    
    if not valid_opts:
        options_str += "No valid moves available."
    else:
        options_str += "; ".join(valid_opts)
        
    return options_str

def _get_common_metadata(td: TensorDict, i: int, actions: List[List[int]]) -> str:
    """Helper to generate common routing metadata (Start, Current, Trajectory)."""
    meta_parts: List[str] = []
    
    # Extract first_node and current_node
    first_node = None
    current_node = None
    if actions is not None and len(actions) > 0:
        if "first_node" in td.keys():
            fn = _to_numpy(td["first_node"][i])
            first_node = int(fn) if hasattr(fn, "__int__") else int(fn[0])
        elif "depot" in td.keys():
            # For CVRP/OP, start node is depot (0)
            first_node = 0
            
        if "current_node" in td.keys():
            cn = _to_numpy(td["current_node"][i])
            current_node = int(cn) if hasattr(cn, "__int__") else int(cn[0])
    
    if first_node is not None:
        meta_parts.append(f"Start node: {first_node};")
    else:
        meta_parts.append("Choose an arbitrary node as the starting node.")
        
    if current_node is not None:
        meta_parts.append(f"Current node: {current_node};")
        
    if actions is not None and len(actions) > 0:
        # Assuming actions is List[List[int]], we need to extract the i-th batch's trajectory
        # actions is [step1_batch, step2_batch, ...]
        # so we need to collect [step1_batch[i], step2_batch[i], ...]
        traj = []
        for step_acts in actions:
            if i < len(step_acts):
                 traj.append(step_acts[i])
        
        if traj:
            action_str = ",".join(str(a) for a in traj) 
            meta_parts.append(f"Trajectory: {action_str};")
            
    return " ".join(meta_parts) + " " if meta_parts else ""

def _get_locs_scaled(td: TensorDict, i: int):
    """Helper to extract and scale locations."""
    locs = td["locs"][i]
    if "locs_mask" in td.keys():
        mask = td["locs_mask"][i]
        if mask.numel() > 0:
            valid_n = int(mask.sum().item())
            locs = locs[:valid_n]
            
    locs_np = _to_numpy(locs)
    try:
        locs_scaled = (locs_np * 1000).astype(int)
    except Exception:
        locs_scaled = np.array(locs_np, dtype=int)
    return locs_scaled

def render_tsp_smart_dual_view(
    locs, visited_mask, current_node_idx, path_history, top_candidates, 
    img_height=336, debug_save_path=None
):
    """
    TSP 智能双视图渲染 (Clean Geometric Style - Refined Labels).
    
    Changes:
    - Layer Order Fix: Label Box is now drawn ON TOP of the Red Node Dot.
    - Size Reduction: Smaller font and padding to reduce clutter/overlap.
    """
    
    # --- 1. 配色方案 ---
    COLOR_BG = (255, 255, 255)
    
    # 节点颜色
    COLOR_CURRENT_FILL = (220, 100, 50)    # Royal Blue
    COLOR_START_FILL = (50, 200, 50)
    COLOR_UNVISITED = (34, 34, 200)        # Deep Red
    COLOR_VISITED = (200, 200, 200)        # Light Grey
    COLOR_START = (20, 20, 20)             # Black
    
    # 辅助
    COLOR_TEXT = (10, 10, 10)
    COLOR_ZOOM_BOX = (0, 0, 255)
    COLOR_BORDER = (180, 180, 180)

    # 画布初始化
    img_width = img_height * 2 
    combined_canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    left_roi = combined_canvas[:, :img_height] 
    right_roi = combined_canvas[:, img_height:]

    # --- 2. 坐标变换逻辑 ---
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

    # --- 3. 全局视图计算 ---
    g_min = np.min(locs, axis=0)
    g_max = np.max(locs, axis=0)
    g_center = (g_min + g_max) / 2.0
    g_span = np.max(g_max - g_min)
    
    global_transform, _ = get_transform(g_center, g_span, img_height, padding=60)

    # --- 4. 智能聚焦逻辑 ---
    curr_pos = locs[current_node_idx]
    if top_candidates:
        cand_coords = np.array([[c['x'], c['y']] for c in top_candidates])
        dists = np.linalg.norm(cand_coords - curr_pos, axis=1)
        max_dist = np.max(dists)
        zoom_span = max(max_dist * 2.5, g_span * 0.05)
        zoom_span = min(zoom_span, g_span * 0.5)
    else:
        zoom_span = g_span * 0.2

    zoom_transform, (z_real_min, z_real_max) = get_transform(curr_pos, zoom_span, img_height, padding=40)

    # --- 5. 绘图函数 ---
    def draw_scene(canvas, transform_fn, view_bounds=None, is_zoomed=False):
        
        def is_visible(pt):
            if view_bounds is None: return True
            x, y = pt
            vmin, vmax = view_bounds
            return (x >= vmin[0]-0.05) and (x <= vmax[0]+0.05) and (y >= vmin[1]-0.05) and (y <= vmax[1]+0.05)

        pts = transform_fn(locs)
        
        # === Layer 1: Gradient Path History (Momentum) ===
        if len(path_history) > 1:
            hist_to_draw = path_history if not is_zoomed else path_history[-15:]
            hist_pts = pts[hist_to_draw]
            
            num_segments = len(hist_pts) - 1
            for i in range(num_segments):
                pt_a = tuple(hist_pts[i])
                pt_b = tuple(hist_pts[i+1])
                ratio = i / max(num_segments, 1)
                gray_val = int(230 - (150 * ratio)) 
                color = (gray_val, gray_val, gray_val)
                thickness = 3 if is_zoomed else 2
                cv2.line(canvas, pt_a, pt_b, color, thickness, cv2.LINE_AA)

        # === Layer 2: Base Nodes (Non-Candidates) ===
        node_radius = 6 if is_zoomed else 4
        
        for i in range(len(locs)):
            if not is_visible(locs[i]): continue
            pt = tuple(pts[i])
            
            # 检查是否为 candidate 或 current
            is_candidate = False
            for c in top_candidates:
                if c['id'] == i: is_candidate = True; break
                
            if i == current_node_idx:
                continue
            elif is_candidate:
                continue # 留给 Top Layer 绘制
            elif visited_mask[i]:
                cv2.circle(canvas, pt, node_radius, COLOR_VISITED, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, pt, node_radius, COLOR_UNVISITED, -1, cv2.LINE_AA)

        # === Layer 3: Candidates (Dot -> Box -> Text) ===
        if len(path_history) > 0:
            start_idx = path_history[0]
            if is_visible(locs[start_idx]) and start_idx != current_node_idx:
                s_pt = tuple(pts[start_idx])
                # 画一个蓝色小方块，尺寸略小于或等于 Current Node
                s_box_size = 8 if is_zoomed else 6
                # 使用与 Current Node 相同的蓝色填充
                cv2.rectangle(canvas, (s_pt[0]-s_box_size, s_pt[1]-s_box_size), 
                              (s_pt[0]+s_box_size, s_pt[1]+s_box_size), COLOR_START_FILL, -1, cv2.LINE_AA)
                # 可选：加上白色边框以增加对比度
                cv2.rectangle(canvas, (s_pt[0]-s_box_size, s_pt[1]-s_box_size), 
                              (s_pt[0]+s_box_size, s_pt[1]+s_box_size), (255,255,255), 1, cv2.LINE_AA)
        
        # 参数调整：更小的字体，更紧凑的边距
        font_scale = 0.5 if is_zoomed else 0.4  # 原来是 0.7/0.5
        cand_label_box_pad = 6 if is_zoomed else 4 # 原来是 12/8
        label_thickness = 1 if is_zoomed else 1
        
        # 倒序遍历，这样 Rank A (index 0) 最后绘制，确保它在最上层不被遮挡
        candidate_list = list(enumerate(top_candidates))
        
        for rank, cand in reversed(candidate_list):
            cand_idx = cand['id']
            cand_pt = tuple(pts[cand_idx])
            label = chr(65 + rank) # A, B, C...
            
            # 3.1 Draw Red Dot FIRST (作为底层)
            # 即使被标签盖住，边缘稍微露出来一点也没关系，表示这里有个点
            cv2.circle(canvas, cand_pt, node_radius, COLOR_UNVISITED, -1, cv2.LINE_AA)
            
            # 3.2 Label Background (Box) ON TOP
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)
            
            box_tl = (cand_pt[0] - w//2 - cand_label_box_pad, cand_pt[1] - h//2 - cand_label_box_pad)
            box_br = (cand_pt[0] + w//2 + cand_label_box_pad, cand_pt[1] + h//2 + cand_label_box_pad)
            
            cv2.rectangle(canvas, box_tl, box_br, (255, 255, 255), -1, cv2.LINE_AA) # White BG
            cv2.rectangle(canvas, box_tl, box_br, (50, 50, 50), 1, cv2.LINE_AA)     # Thin Border
            
            # 3.3 Label Text
            # 居中对齐
            text_x = cand_pt[0] - w // 2
            text_y = cand_pt[1] + h // 2
            cv2.putText(canvas, label, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, label_thickness, cv2.LINE_AA)

        # === Layer 4: Current Node (Ultimate Top) ===
        curr_pt = tuple(pts[current_node_idx])
        curr_size = 10 if is_zoomed else 6
        cv2.rectangle(canvas, (curr_pt[0]-curr_size-2, curr_pt[1]-curr_size-2), 
                      (curr_pt[0]+curr_size+2, curr_pt[1]+curr_size+2), (255,255,255), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (curr_pt[0]-curr_size, curr_pt[1]-curr_size), 
                      (curr_pt[0]+curr_size, curr_pt[1]+curr_size), COLOR_CURRENT_FILL, -1, cv2.LINE_AA)

    # --- 6. 执行绘制 ---
    draw_scene(left_roi, global_transform, is_zoomed=False)
    
    # Focus Box
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
    
    # --- 7. 简化版图例 ---
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
        
        current_y = draw_item(current_y, "Current Agent", lambda x, y: cv2.rectangle(img, (x-6, y-6), (x+6, y+6), COLOR_CURRENT_FILL, -1))
        current_y = draw_item(current_y, "Start Node", lambda x, y: cv2.rectangle(img, (x-5, y-5), (x+5, y+5), COLOR_START_FILL, -1))
        current_y = draw_item(current_y, "Target Node", lambda x, y: cv2.circle(img, (x, y), 5, COLOR_UNVISITED, -1))
        
        # 更新图例以反映新的样式 (Box over Dot)
        def icon_cand_label(x, y):
            # 先画点
            cv2.circle(img, (x, y), 5, COLOR_UNVISITED, -1)
            # 再画框
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), (255,255,255), -1)
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), (50,50,50), 1)
            cv2.putText(img, "A", (x-4, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        current_y = draw_item(current_y, "Candidate", icon_cand_label)
        
    draw_legend(left_roi)
    
    # UI Border & Title
    cv2.rectangle(combined_canvas, (img_height, 0), (img_width-1, img_height-1), COLOR_BORDER, 4)
    cv2.putText(left_roi, "Global View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 2, cv2.LINE_AA)
    cv2.putText(right_roi, "Egocentric View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 2, cv2.LINE_AA)

    # --- 8. 输出 ---
    _, buffer = cv2.imencode('.png', combined_canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    img_rgb_np = cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2RGB)

    if debug_save_path is not None:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, combined_canvas)
    
    return b64_str, img_rgb_np

from scipy.spatial.distance import cdist, pdist, squareform
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components



def get_cluster_entry_points(curr_pos, unvisited_locs, max_bridges=5):
    """
    使用基于距离的连通分量分析，识别独立的簇，并找到每个簇的入口点。
    """
    n = len(unvisited_locs)
    if n < 2:
        return [], []
    
    # 1. 计算所有未访问点之间的距离矩阵
    # 注意：这里计算的是 unvisited 内部的结构
    dists_matrix = squareform(pdist(unvisited_locs))
    
    # 2. 动态确定"断裂"阈值
    # 逻辑：计算每个点到其最近邻居的距离
    # 簇内的点间距通常很小，簇间的间距很大
    # 我们取所有点"最近邻距离"的 (Mean + 2*Std) 作为连接阈值
    # 这意味着如果两个点距离超过了普通间距的很多倍，它们就断开了
    np.fill_diagonal(dists_matrix, np.inf)
    nearest_neighbor_dists = np.min(dists_matrix, axis=1)
    
    # 阈值设定：比较宽松，保证簇内连通，但也足够切断大的跳跃
    # 如果点非常稀疏，这个阈值会自动变大
    threshold = np.mean(nearest_neighbor_dists) + 2.5 * np.std(nearest_neighbor_dists)
    # 保底阈值，防止过于密集时阈值太小
    threshold = max(threshold, 0.05) 
    
    # 3. 构建邻接矩阵 & 求解连通分量
    adj_matrix = dists_matrix < threshold
    n_components, labels = connected_components(csr_matrix(adj_matrix), directed=False)
    
    # 4. 分析每个簇，找到入口点
    # 计算当前位置到所有未访问点的距离
    dists_to_curr = cdist(curr_pos.reshape(1, 2), unvisited_locs).flatten()
    
    cluster_entries = []
    
    # 找到当前所在的簇（即离我最近的点所在的簇）
    nearest_node_idx = np.argmin(dists_to_curr)
    current_cluster_label = labels[nearest_node_idx]
    
    for label_id in range(n_components):
        # 跳过当前所在的簇 (因为这部分由 KNN 负责)
        if label_id == current_cluster_label and n_components > 1:
            continue
            
        # 找到该簇的所有点
        member_indices = np.where(labels == label_id)[0]
        
        # 过滤掉噪点：如果一个簇太小（比如只有1-2个点），且不是唯一的簇，可能不值得作为一个 Strategic Jump
        # 但用户提到要连接"其他未连接点"，为了保险，只要是独立的簇我们都考虑
        if len(member_indices) == 0: continue
        
        # 找到该簇中离我最近的点 (Entry Point)
        dists_subset = dists_to_curr[member_indices]
        best_idx_in_subset = np.argmin(dists_subset)
        real_idx = member_indices[best_idx_in_subset]
        
        dist_val = dists_to_curr[real_idx]
        cluster_size = len(member_indices)
        
        cluster_entries.append({
            "rel_idx": real_idx,   # 在 unvisited_locs 中的索引
            "dist": dist_val,
            "size": cluster_size
        })
    
    # 5. 排序逻辑：优先推荐"大簇"的入口，或者"最近"的簇入口
    # 这里我们采用混合分：优先考虑距离，但如果簇很小则降权
    # 简单策略：按距离排序
    cluster_entries.sort(key=lambda x: x['dist'])
    
    # 返回前 N 个入口的索引
    return [item['rel_idx'] for item in cluster_entries[:max_bridges]]

# --- 主构建函数 ---
def build_obs_tsp(
    td, 
    env_num: int, 
    trajectory: list = None, 
    top_k: int = 24, 
    image_obs: str = "path", # or "base64"
    given_topk_acts = None
) -> list:
    """
    TSP Observation 构建函数 (Density-Based Cluster Aware + Spatial Semantics).
    """
    obs_list = []
    
    # 数据提取
    locs = _to_numpy(td["locs"])               
    current_node = _to_numpy(td["current_node"]) 
    visited = ~_to_numpy(td["action_mask"])    
    i_step = _to_numpy(td["i"])                
    
    if "topk_acts" not in td.keys():
        td["topk_acts"] = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)
    
    topk_acts_list = []
    
    # 策略参数
    num_bridges = 5 if top_k >= 10 else 1
    num_knn = top_k - num_bridges
    
    if given_topk_acts is not None:
        given_topk_acts = _to_numpy(given_topk_acts)

    for idx in range(env_num):
        curr_locs = locs[idx]
        curr_idx = int(current_node[idx])
        curr_visited = visited[idx]
        step = i_step[idx].item() if hasattr(i_step[idx], "item") else i_step[idx]
        curr_pos = curr_locs[curr_idx]
        
        # --- [MODIFIED] 轨迹处理与动量计算 ---
        path_history = []
        momentum_str = "None (Start)"
        
        if trajectory is not None and len(trajectory) > 0:
            for t_step in trajectory:
                val = t_step[idx]
                if hasattr(val, 'item'): val = val.item()
                path_history.append(int(val))
        
        # 补全当前点到历史
        if len(path_history) == 0 or path_history[-1] != curr_idx:
            path_history.append(curr_idx)
            
        # 计算动量 (Momentum): 上一步是如何到达当前点的？
        if len(path_history) >= 2:
            prev_idx = path_history[-2]
            prev_pos = curr_locs[prev_idx]
            m_dx = curr_pos[0] - prev_pos[0]
            m_dy = curr_pos[1] - prev_pos[1]
            m_dir = get_spatial_desc(m_dx, m_dy)
            # 动量描述：显示我是朝哪个方向移动才到了这里
            momentum_str = f"Heading {m_dir} from Node {prev_idx}"

        # 候选生成 (逻辑保持不变)
        candidates = []
        
        # [Branch A: SFT Injection]
        if given_topk_acts is not None:
            indices = given_topk_acts[idx]
            for cand_id in indices:
                if cand_id == -1: continue
                cand_id = int(cand_id)
                dist_val = np.linalg.norm(curr_locs[cand_id] - curr_pos)
                candidates.append({
                    "id": cand_id, "dist": dist_val, "strategy": "inject",
                    "x": curr_locs[cand_id][0], "y": curr_locs[cand_id][1],
                })
            topk_acts_list.append(indices)

        # [Branch B: 智能混合策略]
        else:
            # 1. 准备未访问数据
            unvisited_indices = np.where(curr_visited == 0)[0]
            unvisited_indices = unvisited_indices[unvisited_indices != curr_idx]
            
            if len(unvisited_indices) <= top_k:
                # 剩余点少于 K，全选
                final_indices = unvisited_indices
                strategies = {uid: "knn" for uid in final_indices}
            else:
                unvisited_locs = curr_locs[unvisited_indices]
                
                # --- 策略 A: KNN (最近邻) ---
                dists_to_curr = cdist(curr_pos.reshape(1, 2), unvisited_locs).flatten()
                knn_sorted_args = np.argsort(dists_to_curr)
                knn_local_indices = knn_sorted_args[:num_knn]
                knn_real_indices = unvisited_indices[knn_local_indices]
                
                # --- 策略 B: Cluster Bridge (独立簇入口) ---
                # 使用改进的基于密度的聚类寻找入口
                bridge_local_indices = get_cluster_entry_points(curr_pos, unvisited_locs, max_bridges=num_bridges)
                bridge_real_indices = unvisited_indices[bridge_local_indices]
                
                # --- 合并 ---
                final_set = set(knn_real_indices) | set(bridge_real_indices)
                final_indices = list(final_set)
                
                strategies = {}
                for uid in final_indices:
                    # 如果一个点既是 KNN 又是 Bridge，优先标记为 Bridge (因为它具有战略意义)
                    # 或者反过来，为了防止 KNN 被误标。
                    # 通常如果 Bridge 出现在 KNN 里，说明簇很近，那就是 KNN。
                    # Bridge 的真正价值在于那些“不在 KNN 列表里的远方入口”。
                    if uid in bridge_real_indices and uid not in knn_real_indices:
                        strategies[uid] = "bridge"
                    else:
                        strategies[uid] = "knn"

            # 排序 & Padding
            final_dists = []
            for uid in final_indices:
                final_dists.append(np.linalg.norm(curr_locs[uid] - curr_pos))
            
            sorted_bundled = sorted(zip(final_indices, final_dists), key=lambda x: x[1])
            sorted_indices = [x[0] for x in sorted_bundled]
            
            sorted_indices = sorted_indices[:top_k]

            
            for uid in sorted_indices:
                candidates.append({
                    "id": int(uid),
                    "dist": np.linalg.norm(curr_locs[uid] - curr_pos),
                    "strategy": strategies.get(uid, "knn"),
                    "x": curr_locs[uid][0],
                    "y": curr_locs[uid][1],
                })
            keep_set={}
            candidates = apply_angular_masking(
                candidates=candidates,
                current_pos=curr_pos,       # numpy array [x, y]
                angle_threshold_deg=20.0,   # 你的设定
                always_keep_ids=keep_set,
                coordinate_keys=('x', 'y')
            )

            valid_len = len(sorted_indices)
            padded = np.array(sorted_indices + [-1]*(top_k - valid_len))
            topk_acts_list.append(padded)


        # 4. 可视化
        img_b64 = None
        image_save_path = None
        # if given_topk_acts is not None:
        # debug_path = f"./debug_images/tsp/env{idx}_step{step:03d}.png"
        
        if image_obs == "path":
            uid = uuid.uuid4()
            image_save_dir = f"/root/autodl-tmp/image/tsp/"
            os.makedirs(image_save_dir, exist_ok=True)
            image_save_path = f"{image_save_dir}env{idx}_step{len(path_history):03d}_{uid}.png"
        
        img_b64, image_rgb_np = render_tsp_smart_dual_view(
            locs=curr_locs, 
            visited_mask=(curr_visited==1), 
            current_node_idx=curr_idx, 
            path_history=path_history, 
            top_candidates=candidates, 
            debug_save_path=image_save_path
        )

        # 5.Prompt 生成：加入相对空间语义
        cand_str_list = []
        for rank, cand in enumerate(candidates):
            label = chr(65 + rank)
            
            # 计算向量
            dx = cand['x'] - curr_pos[0]
            dy = cand['y'] - curr_pos[1]
            bearing = get_spatial_desc(dx, dy)
            
            # 格式化数字，避免过长
            dist_disp = f"{cand['dist']*100:.1f}" # 假设坐标0-1，乘100更易读
            vec_disp = f"[{dx:+.2f}, {dy:+.2f}]"
            
            strat_mark = ""
            if cand.get('strategy') == 'bridge':
                strat_mark = " **[New Cluster Entry]**" 
            
            # 构建富含信息的描述行
            # Option A [Node 22]: Dist: 1.4 | Vec: [+0.01, -0.04] (South)
            cand_str_list.append(
                f"Option {label} [Node {cand['id']}]: "
                f"Dist: {dist_disp} | Vec: {vec_disp} ({bearing}){strat_mark}"
            )
            
        cand_section = "\n".join(cand_str_list)
        remaining = curr_locs.shape[0] - np.sum(curr_visited)

        obs_text = (
            f"### Task: Traveling Salesperson Problem (TSP)\n"
            f"Step: {step}\n"
            f"Status: Current Node {curr_idx}, Unvisited {remaining}\n"
            f"Momentum: {momentum_str}\n"  # 新增动量信息
            f"History: {path_history[-10:]}\n\n"
            f"### Candidate Options (Spatial & Cluster Analysis):\n"
            f"Relative Vectors [dx, dy] indicate position relative to Current Node (0,0).\n"
            f"- Standard: Local neighbors.\n"
            f"- **[New Cluster Entry]**: Jump to distant cluster.\n"
            f"\n{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Option Label (A, B...) to visit next."
        )
        
        status_str = (
            f"Step: {step}\n"
            f"Status: Current Node {curr_idx}, Unvisited {remaining}\n"
            f"Momentum: {momentum_str}\n"  # 新增动量信息
            f"History: {path_history[-10:]}\n"
        )
        candidates_str = (
            f"Relative Vectors [dx, dy] indicate position relative to Current Node (0,0).\n"
            f"- Standard: Local neighbors.\n"
            f"- **[New Cluster Entry]**: Jump to distant cluster.\n"
            f"\n{cand_section}\n"
        )

        if image_obs == "base64":
            obs_list.append({"text": obs_text, "image": img_b64, "obs": status_str, "candidates": candidates_str})
        elif image_obs == "path":
            obs_list.append({"text": obs_text, "image": image_save_path, "obs": status_str, "candidates": candidates_str})
        else:
            obs_list.append({"text": obs_text, "image": image_rgb_np, "obs": status_str, "candidates": candidates_str})

    if given_topk_acts is None and len(topk_acts_list) > 0:
        try:
             td["topk_acts"] = torch.tensor(np.array(topk_acts_list), device=td.device)
        except: pass
        
    return obs_list

def render_cvrp_image(
    locs, 
    demands, 
    visited_mask, 
    current_node_idx, 
    path_history, 
    used_capacity, 
    vehicle_capacity, 
    top_candidates, 
    img_size=448, 
    debug_save_path=None
):
    """
    CVRP 单视图渲染 (Single Scientific View).
    
    Layout:
    - Main: Global Map with Depot, Customers, and Agent Path.
    - Top: Capacity Progress Bar.
    - Overlay: Top-K Candidate Edges (Green) with Labels.
    """
    
    # --- 1. 配色方案 (Scientific) ---
    COLOR_BG = (255, 255, 255)
    
    # 节点
    COLOR_DEPOT = (20, 20, 20)             # Black Square
    COLOR_AGENT = (220, 100, 50)           # Royal Blue (Current)
    COLOR_UNVISITED = (34, 34, 200)        # Deep Red
    COLOR_VISITED = (220, 220, 220)        # Light Grey
    
    # 线条
    COLOR_PATH = (180, 180, 180)           # Grey History
    COLOR_CANDIDATE = (50, 180, 50)        # Forest Green
    
    # 辅助
    COLOR_TEXT = (20, 20, 20)
    COLOR_CAP_BG = (240, 240, 240)
    COLOR_CAP_FILL = (100, 100, 255)       # Blueish fill
    COLOR_CAP_ALERT = (50, 50, 220)        # Reddish if >90% full

    # 画布初始化
    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # --- 2. 坐标变换 (全局) ---
    g_min = np.min(locs, axis=0)
    g_max = np.max(locs, axis=0)
    g_center = (g_min + g_max) / 2.0
    g_span = np.max(g_max - g_min)
    
    # padding 稍微大一点，给 Capacity Bar 和 Legend 留空间
    padding = 80
    available_size = img_size - 2 * padding
    scale = available_size / max(g_span, 1e-6)
    canvas_center = img_size / 2.0
    
    def to_xy(coords):
        coords = np.array(coords)
        centered = coords - g_center
        scaled = centered * scale
        final = scaled.copy()
        final[..., 0] += canvas_center
        final[..., 1] = canvas_center - final[..., 1] 
        return final.astype(int)

    # 预计算所有点坐标
    pts = to_xy(locs)

    # --- 3. 绘图: Layer 1 - History Path (最底层) ---
    if len(path_history) > 1:
        # 筛选：只展示当前从 Depot 出发后的轨迹 (Current Subtour)
        last_depot_idx = 0
        if top_candidates != []:
            for i in range(len(path_history) - 1, -1, -1):
                if path_history[i] == 0:
                    last_depot_idx = i
                    break
        
        current_subtour = path_history[last_depot_idx:]
        
        if len(current_subtour) > 1:
            hist_pts = pts[current_subtour]
            cv2.polylines(canvas, [hist_pts], isClosed=False, color=COLOR_PATH, thickness=2, lineType=cv2.LINE_AA)

    # --- 4. 绘图: Layer 2 - Nodes (中间层) ---
    node_radius = 5
    depot_size = 10
    
    for i in range(len(locs)):
        pt = tuple(pts[i])
        
        if i == 0:
            # Depot (Square)
            cv2.rectangle(canvas, (pt[0]-depot_size, pt[1]-depot_size), 
                          (pt[0]+depot_size, pt[1]+depot_size), COLOR_DEPOT, -1, cv2.LINE_AA)
            # Label 'D'
            cv2.putText(canvas, "D", (pt[0]-4, pt[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
            
        elif i == current_node_idx:
            # Current Agent (Blue Square) - 稍后在 Top Layer 画
            continue
            
        else:
            # Customers
            if visited_mask[i]: # Visited
                # cv2.circle(canvas, pt, node_radius, COLOR_VISITED, -1, cv2.LINE_AA)
                pass
            else: # Unvisited
                cv2.circle(canvas, pt, node_radius, COLOR_UNVISITED, -1, cv2.LINE_AA)

    # --- 5. 绘图: Layer 3 - Current Node & Candidates (最顶层) ---
    
    # 5.1 Current Agent
    curr_pt = tuple(pts[current_node_idx])
    curr_size = 8
    # White Halo
    cv2.rectangle(canvas, (curr_pt[0]-curr_size-2, curr_pt[1]-curr_size-2), 
                  (curr_pt[0]+curr_size+2, curr_pt[1]+curr_size+2), (255,255,255), -1, cv2.LINE_AA)
    # Blue Fill
    cv2.rectangle(canvas, (curr_pt[0]-curr_size, curr_pt[1]-curr_size), 
                  (curr_pt[0]+curr_size, curr_pt[1]+curr_size), COLOR_AGENT, -1, cv2.LINE_AA)

    # 5.2 Candidates (Green Lines & Labels)
    cand_line_width = 2
    cand_label_box = 12
    font_scale = 0.6
    
    # Reverse 遍历：确保 Option A (Rank 0) 最后绘制，压在 B, C 上
    candidate_list = list(enumerate(top_candidates))
    for rank, cand in reversed(candidate_list):
        cand_id = cand['id']
        cand_pt = tuple(pts[cand_id])
        
        # 绿线
        # cv2.line(canvas, curr_pt, cand_pt, COLOR_CANDIDATE, cand_line_width, cv2.LINE_AA)
        
        # 标签 (在目标点位置)
        label = get_label(rank)
        
        # 白底黑框 (Box)
        cv2.rectangle(canvas, (cand_pt[0]-cand_label_box, cand_pt[1]-cand_label_box), 
                      (cand_pt[0]+cand_label_box, cand_pt[1]+cand_label_box), (255,255,255), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (cand_pt[0]-cand_label_box, cand_pt[1]-cand_label_box), 
                      (cand_pt[0]+cand_label_box, cand_pt[1]+cand_label_box), (20,20,20), 1, cv2.LINE_AA)
        
        # 文字
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        text_x = cand_pt[0] - w // 2
        text_y = cand_pt[1] + h // 2
        cv2.putText(canvas, label, (text_x, text_y), 
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, 1, cv2.LINE_AA)

    # --- 6. Capacity Bar (顶部信息栏) ---
    bar_w = int(img_size * 0.4)
    bar_h = 20
    bar_x = (img_size - bar_w) - 10
    bar_y = 40
    
    # 背景
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_CAP_BG, -1)
    
    # 填充
    if vehicle_capacity > 0:
        fill_ratio = min(1.0, used_capacity / vehicle_capacity)
    else:
        fill_ratio = 0.0
        
    fill_w = int(bar_w * fill_ratio)
    fill_color = COLOR_CAP_ALERT if fill_ratio > 0.9 else COLOR_CAP_FILL
    
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)
    # 边框
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (100,100,100), 1)
    
    # 文字
    cap_text = f"Vehicle Load: {used_capacity:.1f} / {vehicle_capacity:.1f}"
    cv2.putText(canvas, cap_text, (bar_x, bar_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 1, cv2.LINE_AA)

    # --- 7. 图例 (Legend) ---
    def draw_legend(img):
        start_x, start_y = 20, img_size - 25
        # 横向排列图例，因为单视图底部空间宽裕
        # Item 1: Depot
        cv2.rectangle(img, (start_x, start_y-10), (start_x+10, start_y), COLOR_DEPOT, -1)
        cv2.putText(img, "Depot", (start_x+15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60,60,60), 1, cv2.LINE_AA)
        
        # Item 2: Unvisited
        offset = 100
        cv2.circle(img, (start_x+offset, start_y-5), 5, COLOR_UNVISITED, -1)
        cv2.putText(img, "Unvisited", (start_x+offset+10, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60,60,60), 1, cv2.LINE_AA)

        # Item 3: Agent
        offset += 120
        cv2.rectangle(img, (start_x+offset, start_y-10), (start_x+offset+10, start_y), COLOR_AGENT, -1)
        cv2.putText(img, "Vehicle", (start_x+offset+15, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60,60,60), 1, cv2.LINE_AA)
        
        # Item 4: Candidate
        offset += 110
        cv2.line(img, (start_x+offset, start_y-5), (start_x+offset+20, start_y-5), COLOR_CANDIDATE, 2)
        cv2.putText(img, "Candidate", (start_x+offset+25, start_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (60,60,60), 1, cv2.LINE_AA)

    draw_legend(canvas)
    
    # 标题
    cv2.putText(canvas, "CVRP Global Status", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2, cv2.LINE_AA)

    # --- 8. 输出 ---
    _, buffer = cv2.imencode('.png', canvas)
    image_rgb_np = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    
    b64_str = base64.b64encode(buffer).decode('utf-8')

    if debug_save_path is not None:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, canvas)
    
    return b64_str, image_rgb_np

def _get_direction_label(dx, dy):
    """Auxiliary: Convert vector to 8-direction cardinal label."""
    angle = np.arctan2(dy, dx) * 180 / np.pi
    if -22.5 <= angle < 22.5: return "East (\u2192)"
    elif 22.5 <= angle < 67.5: return "North-East (\u2197)"
    elif 67.5 <= angle < 112.5: return "North (\u2191)"
    elif 112.5 <= angle < 157.5: return "North-West (\u2196)"
    elif 157.5 <= angle <= 180 or -180 <= angle < -157.5: return "West (\u2190)"
    elif -157.5 <= angle < -112.5: return "South-West (\u2199)"
    elif -112.5 <= angle < -67.5: return "South (\u2193)"
    else: return "South-East (\u2198)"

def build_obs_cvrp(
    td, 
    env_num: int, 
    trajectory: list = None, 
    top_k: int = 24,
    given_topk_acts = None,
    image_obs: str = "rgb", # or "base64" or "path"
) -> list:
    """
    CVRP Observation Builder (Updated with Dual View & Relative Geometry).
    """
    obs_list = []
    
    # --- 数据转换 ---
    locs = _to_numpy(td["locs"])               
    demands = _to_numpy(td["demand"])            
    current_node = _to_numpy(td["current_node"]) 
    used_capacity = _to_numpy(td["used_capacity"]) 
    vehicle_capacity = _to_numpy(td["vehicle_capacity"]) 
    
    if "action_mask" in td.keys():
        visited = _to_numpy(~td["visited"])   
    else:
        visited = np.zeros((env_num, locs.shape[1]))

    if given_topk_acts is not None:
        given_topk_acts = _to_numpy(given_topk_acts)

    topk_acts_list = []

    # 策略参数：Hybrid Strategy
    if top_k >= 10: num_far = 4 
    elif top_k >= 5: num_far = 1
    else: num_far = 0
    num_knn = top_k - num_far

    for idx in range(env_num):
        # --- 1. 状态提取 ---
        curr_locs = locs[idx]          
        curr_demands = demands[idx]    
        curr_idx = int(current_node[idx])
        curr_visited = ~visited[idx]    
        
        curr_used = float(used_capacity[idx])
        curr_cap = float(vehicle_capacity[idx])
        remaining_cap = curr_cap - curr_used
        
        # 轨迹处理 & Momentum 计算
        path_history = []
        if trajectory is not None and len(trajectory) > 0:
            for t_step in trajectory:
                val = t_step[idx]
                if hasattr(val, 'item'): val = val.item()
                path_history.append(int(val))
        if len(path_history) == 0 or path_history[-1] != curr_idx:
            path_history.append(curr_idx)
            
        # 计算动量 (Momentum)
        momentum_str = "Stationary (Start)"
        if len(path_history) >= 2:
            prev_idx = path_history[-2]
            prev_pos = curr_locs[prev_idx]
            curr_pos_vec = curr_locs[curr_idx]
            m_vec = curr_pos_vec - prev_pos
            m_dx, m_dy = m_vec[0], m_vec[1]
            if np.linalg.norm(m_vec) > 1e-5:
                m_dir = _get_direction_label(m_dx, m_dy)
                momentum_str = f"Heading {m_dir} from Node {prev_idx}"

        # 预处理 Demand
        num_nodes = curr_locs.shape[0]
        full_demands = np.zeros(num_nodes)
        if len(curr_demands) == num_nodes - 1:
            full_demands[1:] = curr_demands
        elif len(curr_demands) == num_nodes:
            full_demands = curr_demands
        else:
            limit = min(len(curr_demands), num_nodes)
            full_demands[:limit] = curr_demands[:limit]

        # --- 2. 候选生成 (Advanced Multi-Strategy) ---
        candidates = []
        curr_pos = curr_locs[curr_idx]

        if given_topk_acts is not None:
             # SFT Logic (unchanged)
             final_indices = given_topk_acts[idx]
             # Ensure they are valid integers
             final_indices = [int(x) for x in final_indices if x != -1]
        else:
            # 基础距离计算
            dists = cdist(curr_pos.reshape(1, 2), curr_locs, metric='euclidean').flatten()
            
            # 可行性 Mask
            is_unvisited = (curr_visited == 0)
            is_unvisited[0] = 0 # Depot handled separately
            
            # Capacity Constraints
            # strict_feasible: unvisited AND demand <= remaining
            can_fit = full_demands <= (remaining_cap + 1e-5)
            strict_feasible_mask = is_unvisited & can_fit
            # strict_feasible_mask = td["action_mask"][idx]
            
            feasible_indices = np.where(strict_feasible_mask)[0]
            
            # 如果没有可行客户，且 Depot 也不在候选（通常意味着结束），但在 build_obs 阶段
            # 我们至少要保证 Depot 可选 (如果当前不在 Depot)
            if len(feasible_indices) == 0:
                final_indices = [] # Will add depot later
            else:
                # --- Strategy A: KNN (Distance + Angular Masking) ---
                # Take top portion of K for KNN
                k_knn = max(1, int(top_k * 0.5)) 
                
                # Sort feasible by distance
                sorted_by_dist = sorted(feasible_indices, key=lambda x: dists[x])
                
                # [Refinement] Apply Angular Masking to reduce redundancy
                # We initially pick more candidates (3x) and then filter out those that are "shadowed"
                knn_candidates_raw = []
                for c_idx in sorted_by_dist[:k_knn * 3]:
                    knn_candidates_raw.append({
                        "id": c_idx,
                        "x": curr_locs[c_idx][0],
                        "y": curr_locs[c_idx][1]
                    })
                
                knn_filtered_dicts = apply_angular_masking(
                    candidates=knn_candidates_raw,
                    current_pos=curr_pos,
                    angle_threshold_deg=20.0,
                    always_keep_ids={0}, 
                    coordinate_keys=('x', 'y')
                )
                knn_candidates = [d['id'] for d in knn_filtered_dicts][:k_knn]
                
                # --- Strategy B: Capacity Fit (Best Fit) ---
                # Prioritize nodes that fill the remaining capacity tightly
                # Sort by (remaining - demand) ascending, i.e., demand descending
                k_cap = max(0, int(top_k * 0.2))
                if k_cap > 0:
                    sorted_by_demand = sorted(feasible_indices, key=lambda x: full_demands[x], reverse=True)
                    cap_candidates = sorted_by_demand[:k_cap]
                else:
                    cap_candidates = []

                # --- Strategy C: Momentum (Directional) ---
                # Prioritize nodes in front of us
                k_mom = max(0, int(top_k * 0.2))
                mom_candidates = []
                if k_mom > 0 and len(path_history) >= 2:
                    # Current vector
                    prev_pos_v = curr_locs[path_history[-2]]
                    move_vec = curr_pos - prev_pos_v
                    if np.linalg.norm(move_vec) > 1e-5:
                        # Normalize move vector
                        move_vec = move_vec / np.linalg.norm(move_vec)
                        
                        # Calculate cosine similarity for all feasible
                        # vec_to_cand = cand_pos - curr_pos
                        cand_vecs = curr_locs[feasible_indices] - curr_pos
                        norms = np.linalg.norm(cand_vecs, axis=1)
                        norms[norms < 1e-6] = 1.0 # Avoid div by zero
                        
                        # Dot product
                        dots = np.sum(move_vec * cand_vecs, axis=1)
                        cos_sims = dots / norms
                        
                        # Get indices with highest cosine
                        mom_local_indices = np.argsort(-cos_sims)[:k_mom]
                        mom_candidates = [feasible_indices[i] for i in mom_local_indices]

                # --- Strategy D: Furthest (Global) ---
                # Use remaining slots for furthest to avoid local optima
                # We simply take the furthest from the feasible set
                sorted_by_dist_desc = sorted(feasible_indices, key=lambda x: dists[x], reverse=True)
                far_candidates = sorted_by_dist_desc[:num_far]

                # --- Merge & Dedup ---
                # Order matters: KNN first, then Cap, then Mom, then Far
                merged = []
                seen = set()
                
                for cand_list in [knn_candidates, cap_candidates, mom_candidates, far_candidates]:
                    for c in cand_list:
                        if c not in seen:
                            merged.append(c)
                            seen.add(c)
                
                # Fill remaining with KNN if not enough
                if len(merged) < top_k:
                    for c in sorted_by_dist:
                        if c not in seen:
                            merged.append(c)
                            seen.add(c)
                            if len(merged) >= top_k:
                                break
                                
                final_indices = merged[:top_k]

            # --- Critical: Always Ensure Depot (0) is an option if feasible ---
            # Depot is feasible if we are not at depot.
            # In CVRP, return to depot is always allowed (demand 0 <= remaining).
            if curr_idx != 0:
                if 0 not in final_indices:
                    # Insert Depot. 
                    # If full, replace the last candidate (lowest priority)
                    if len(final_indices) >= top_k:
                        final_indices[-1] = 0
                    else:
                        final_indices.append(0)
            
            # Move Depot to end or beginning? 
            # Usually users like seeing Depot as a distinct option, order doesn't matter for Transformer but matters for user readability.
            # Let's keep it where it is or append.

            for cand_idx in final_indices:
                cand_idx = int(cand_idx)
                strat = "knn"
                if 'far_candidates' in locals() and cand_idx in far_candidates: strat = "furthest"
                elif 'cap_candidates' in locals() and cand_idx in cap_candidates: strat = "capacity_fit"
                elif 'mom_candidates' in locals() and cand_idx in mom_candidates: strat = "momentum"
                
                # 计算相对向量
                cand_pos = curr_locs[cand_idx]
                rel_vec = cand_pos - curr_pos
                dx, dy = rel_vec[0], rel_vec[1]
                
                candidates.append({
                    "id": cand_idx,
                    "dist": dists[cand_idx],
                    "dx": dx, "dy": dy, # Added
                    "demand": float(full_demands[cand_idx]),
                    "is_depot": (cand_idx == 0),
                    "feasible": True,
                    "strategy": strat,
                    "x": cand_pos[0], "y": cand_pos[1] # For renderer
                })
            
            # 定义哪些 ID 需要保留 (CVRP 中 Depot id=0 必须保留)
            keep_set = {0} 
            
            # 执行过滤
            # candidates = apply_angular_masking(
            #     candidates=candidates,
            #     current_pos=curr_pos,       # numpy array [x, y]
            #     angle_threshold_deg=20.0,   # 你的设定
            #     always_keep_ids=keep_set,
            #     coordinate_keys=('x', 'y')
            # )
            final_indices = [cand['id'] for cand in candidates]
            valid_len = len(final_indices)
            padded_indices = np.array(final_indices + [-1]*(top_k - valid_len))
            topk_acts_list.append(padded_indices)

        # --- 3. 生成 Prompt (Updated format) ---
        cand_str_list = []
        for rank, cand in enumerate(candidates):
            label = chr(65 + rank)
            
            # Type Label
            if cand['is_depot']:
                node_type = "**DEPOT (Refill)**" 
            else:
                node_type = f"Node {cand['id']}"
            
            # Demand & Strategy info
            demand_info = f" | Dem: {cand['demand']:.2f}" if not cand['is_depot'] else ""
            
            strat_mark = ""
            if cand.get('strategy') == 'furthest':
                strat_mark = " **[Far Cluster]**" 
            elif cand['is_depot']:
                strat_mark = " **[Return]**"

            # Geometry Info
            dir_label = _get_direction_label(cand['dx'], cand['dy'])
            vec_str = f"[{cand['dx']:+.2f}, {cand['dy']:+.2f}]"
            
            strat_info = ""
            if cand.get('strategy') == 'capacity_fit':
                strat_info = " **[Efficient]**"
            elif cand.get('strategy') == 'momentum':
                strat_info = " **[Forward]**"
            
            cand_str_list.append(
                f"Option {label} [{node_type}]: "
                f"Dist: {cand['dist']:.2f} | Vec: {vec_str} ({dir_label}){demand_info}{strat_mark}{strat_info}"
            )
        cand_section = "\n".join(cand_str_list)
        
        # 统计
        unvisited_mask = (curr_visited == 0)
        unvisited_mask[0] = 0 
        unvisited_customers = np.sum(unvisited_mask)
        
        obs_text = (
            f"### Task: Capacitated Vehicle Routing Problem (CVRP)\n"
            f"Step: {len(path_history)}\n"
            f"Status: Current Node {curr_idx}, Unvisited {unvisited_customers}\n"
            f"Load: {curr_used:.2f} / {curr_cap:.2f} (Rem: {remaining_cap:.2f})\n"
            f"Momentum: {momentum_str}\n"
            f"History: {path_history[-10:]}\n\n"
            f"### Candidate Options (Multi-Strategy: KNN, Capacity, Momentum, Far):\n"
            f"Relative Vectors [dx, dy] indicate position relative to Current Node (0,0).\n"
            f"- Standard: Feasible local neighbors.\n"
            f"- **[Efficient]**: High demand nodes that fit remaining capacity well.\n"
            f"- **[Forward]**: Nodes aligned with current direction.\n"
            f"- **[Far Cluster]**: Distant nodes with high demand (Global planning).\n"
            f"- **[Return]**: Return to Depot to refill capacity.\n"
            f"\n{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Option Label (A, B...) to visit next.\n"
        )
        status_str = (
            f"Step: {len(path_history)}\n"
            f"Status: Current Node {curr_idx}, Unvisited {unvisited_customers}\n"
            f"Load: {curr_used:.2f} / {curr_cap:.2f} (Rem: {remaining_cap:.2f})\n"
            f"Momentum: {momentum_str}\n"
            f"History: {path_history[-10:]}\n"
        )
        candidates_str = (
            f"Relative Vectors [dx, dy] indicate position relative to Current Node (0,0).\n"
            f"- Standard: Feasible local neighbors.\n"
            f"- **[Efficient]**: High demand nodes that fit remaining capacity well.\n"
            f"- **[Forward]**: Nodes aligned with current direction.\n"
            f"- **[Far Cluster]**: Distant nodes with high demand (Global planning).\n"
            f"- **[Return]**: Return to Depot to refill capacity.\n"
            f"\n{cand_section}\n"
        )
        

        # --- 4. 可视化 (Updated) ---
        img_b64 = None
        image_save_path = None
        if image_obs == "path":
            uid = uuid.uuid4()
            image_save_dir = f"/root/autodl-tmp/image/cvrp/"
            os.makedirs(image_save_dir, exist_ok=True)
            image_save_path = f"{image_save_dir}env{idx}_step{len(path_history):03d}_{uid}.png"
        
        # if image_obs == "base64":
        img_b64, image_rgb_np = render_cvrp_image(
            locs=curr_locs,
            demands=full_demands, 
            visited_mask=(curr_visited==1),
            current_node_idx=curr_idx,
            path_history=path_history,
            used_capacity=curr_used,
            vehicle_capacity=curr_cap,
            top_candidates=candidates, 
            debug_save_path=image_save_path
        )

        if image_obs == "base64":
            obs_list.append({"text": obs_text, "image": img_b64, "obs": status_str, "candidates": candidates_str})
        elif image_obs == "path":
            obs_list.append({"text": obs_text, "image": image_save_path, "obs": status_str, "candidates": candidates_str})
        else:
            obs_list.append({"text": obs_text, "image": image_rgb_np, "obs": status_str, "candidates": candidates_str})
            
    # 更新 TD
    if given_topk_acts is None and len(topk_acts_list) > 0:
        try:
             td["topk_acts"] = torch.tensor(np.array(topk_acts_list), device=td.device)
        except:
             pass 

    return obs_list

def build_obs_op(td: TensorDict, env_num: int, trajectory: List[List[int]] = None, return_topk_options: bool = False, top_k: int = 5) -> List[str]:
    batch_size = td.batch_size[0] if td.batch_size else 1
    obs_list: List[str] = []

    for i in range(batch_size):
        # 1. Base Info (with prizes and max length)
        locs_scaled = _get_locs_scaled(td, i)
        
        prize = td.get("prize", None)
        p_np = _to_numpy(prize[i]) if prize is not None else None
        
        max_len_tensor = td.get("max_length", td.get("max_route_length", None))
        max_route_length = None
        if max_len_tensor is not None:
            try:
                max_route_length = float(_to_numpy(max_len_tensor[i]).item())
            except:
                pass

        lines = []
        for node_idx, (x, y) in enumerate(locs_scaled.tolist()):
            prize_val = int(p_np[node_idx]) if (p_np is not None and node_idx < len(p_np)) else 0
            lines.append(f"Node {node_idx}, coordinates: [{x}, {y}], prize: {prize_val};")
        max_len_str = f" Max route length: {max_route_length}." if max_route_length is not None else ""
        base_info = " ".join(lines) + max_len_str + "\n"
        
        # 2. Metadata
        meta_prefix = _get_common_metadata(td, i, trajectory)
        
        # 3. Top-K Options
        topk_str = _get_topk_str(td, i, trajectory, return_topk_options)
        
        obs_str = base_info + meta_prefix + topk_str
        obs_list.append(obs_str)
        
    return obs_list


def uuid_name():
    return str(uuid.uuid4())[:8]
    
def build_obs_tdtsp(
    td, 
    env_num: int, 
    trajectory: list = None, 
    top_k: int = 24, 
    image_obs: str = "path",
    given_topk_acts = None
) -> list:
    """
    Observation builder for TDTSPMatrixEnv.
    Feature: Spatial Semantics + Temporal Congestion Awareness.
    """
    
    obs_list = []
    
    # --- 1. Data Extraction ---
    locs = _to_numpy(td["locs"])               # [B, N, 2]
    current_node = _to_numpy(td["current_node"]) # [B]
    current_time = _to_numpy(td["current_time"]) # [B]
    visited = _to_numpy(td.get("visited", td["action_mask"])) # [B, N]
    action_mask = _to_numpy(td["action_mask"]) # [B, N]
    
    # TDTSP specific
    # matrix shape usually: [B, N, N, T] or [N, N, T] if shared
    matrix = td["travel_time_matrix"] 
    duration = td["time_step_duration"] # Scalar or [B]
    
    # SFT Injection preparation
    if "topk_acts" not in td.keys():
        td["topk_acts"] = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)
    
    topk_acts_list = []
    if given_topk_acts is not None:
        given_topk_acts = _to_numpy(given_topk_acts)

    # --- 2. Environment Loop ---
    for i in range(env_num):
        # 2.1 Basic State Info
        curr_locs = locs[i]
        curr_idx = int(current_node[i])
        curr_pos = curr_locs[curr_idx]
        time_val = float(current_time[i]) if hasattr(current_time[i], 'item') else float(current_time[i])
        
        # Calculate Time Step Index for Matrix Lookup
        if hasattr(duration, 'dim') and duration.dim() > 0:
            curr_duration = float(duration[i])
        else:
            curr_duration = float(duration)
            
        time_step_idx = int(time_val // curr_duration)
        # Handle matrix bounds (clamping)
        max_s = matrix.shape[-1] - 1
        s = min(time_step_idx, max_s)

        # 2.2 Trajectory & Momentum Logic
        path_history = []
        momentum_str = "None (Start)"
        
        if trajectory is not None and len(trajectory) > 0:
            for t_step in trajectory:
                val = t_step[i]
                if hasattr(val, 'item'): val = val.item()
                path_history.append(int(val))
        
        # Add current if missing (robustness)
        if len(path_history) == 0 or path_history[-1] != curr_idx:
            path_history.append(curr_idx)

        # Calculate Momentum Vector
        if len(path_history) >= 2:
            prev_idx = path_history[-2]
            prev_pos = curr_locs[prev_idx]
            m_dx = curr_pos[0] - prev_pos[0]
            m_dy = curr_pos[1] - prev_pos[1]
            m_dir = get_spatial_desc(m_dx, m_dy)
            momentum_str = f"Heading {m_dir} from Node {prev_idx}"

        # 2.3 Candidate Generation
        candidates = []
        
        # Identify unvisited
        # action_mask is 1 for feasible/unvisited, 0 for visited (typically)
        # Ensure we are using the correct logic: 1=valid/unvisited
        unvisited_indices = np.where(action_mask[i] == 1)[0]
        # Remove self if present
        unvisited_indices = unvisited_indices[unvisited_indices != curr_idx]
        
        # Get Time Costs from Matrix
        if matrix.dim() == 4:
            # [B, N, N, T]
            tt_slice = matrix[i, curr_idx, :, s]
        else:
            # [N, N, T]
            tt_slice = matrix[curr_idx, :, s]
        
        # If using PyTorch tensor for matrix, converting to numpy for processing
        if hasattr(tt_slice, 'cpu'):
            tt_slice = tt_slice.cpu().numpy()
            
        unvisited_tt = tt_slice[unvisited_indices]

        # --- Branch A: SFT / Teacher Injection ---
        if given_topk_acts is not None:
            indices = given_topk_acts[i]
            # SFT logic: just verify they are valid and calculate their stats
            for cand_id in indices:
                cand_id = int(cand_id)
                if cand_id == -1: continue
                
                # Re-calculate stats for the forced candidate
                # Note: SFT acts might be visited or invalid, handle gracefully
                try:
                    c_pos = curr_locs[cand_id]
                    dist_val = np.linalg.norm(c_pos - curr_pos)
                    # We need to look up time manually if it wasn't in unvisited list
                    time_cost = float(tt_slice[cand_id])
                    
                    candidates.append({
                        'id': cand_id, 'dist': dist_val, 'tt': time_cost,
                        'x': c_pos[0], 'y': c_pos[1], 'strategy': 'inject'
                    })
                except: pass
            topk_acts_list.append(indices)
            
        # --- Branch B: Greedy (Time-Based) Top-K ---
        else:
            temp_candidates = []
            for j, real_idx in enumerate(unvisited_indices):
                real_idx = int(real_idx)
                time_cost = float(unvisited_tt[j])
                dist_val = np.linalg.norm(curr_locs[real_idx] - curr_pos)
                
                temp_candidates.append({
                    'id': real_idx,
                    'dist': dist_val,
                    'tt': time_cost,
                    'x': curr_locs[real_idx][0],
                    'y': curr_locs[real_idx][1],
                    'strategy': 'greedy'
                })
            
            # Sort primarily by Time Cost (Greedy) for TDTSP
            temp_candidates.sort(key=lambda x: x['tt'])
            candidates = temp_candidates[:top_k]
            
            # Save for TD
            sorted_indices = [c['id'] for c in candidates]
            valid_len = len(sorted_indices)
            padded = np.array(sorted_indices + [-1]*(top_k - valid_len))
            topk_acts_list.append(padded)

        # 2.4 Prompt Construction (Text Generation)
        
        # Calculate statistics for relative descriptions
        if candidates:
            # Pace = Time / Dist. Higher = Slower/Congested.
            # Adding small epsilon to dist to avoid division by zero
            paces = [c['tt'] / (c['dist'] + 1e-5) for c in candidates]
            avg_pace = np.mean(paces)
        else:
            avg_pace = 1.0

        cand_str_list = []
        for rank, cand in enumerate(candidates):
            label = chr(65 + rank) if rank < 26 else f"Opt{rank}"
            
            # Spatial Vector
            dx = cand['x'] - curr_pos[0]
            dy = cand['y'] - curr_pos[1]
            bearing = get_spatial_desc(dx, dy)
            
            # Congestion Logic
            current_pace = cand['tt'] / (cand['dist'] + 1e-5)
            traffic_tag = ""
            
            # Thresholds can be tuned based on your data distribution
            if current_pace < avg_pace * 0.8:
                traffic_tag = "(Fast Route)"
            elif current_pace > avg_pace * 1.2:
                traffic_tag = "(Congested)"
            else:
                traffic_tag = "(Normal)"
            
            # Formatting
            # Dist: Static physics distance
            # Time: Real-world dynamic cost
            cand_str_list.append(
                f"Option {label} [Node {cand['id']}]: "
                f"Dist: {cand['dist']*100:.1f} | "
                f"Time: {cand['tt']:.1f} {traffic_tag} | "
                f"Vec: [{dx:+.2f}, {dy:+.2f}] ({bearing})"
            )
            
        cand_section = "\n".join(cand_str_list)
        remaining_count = len(unvisited_indices) if given_topk_acts is None else "N/A"

        # Construct Final Text
        obs_text = (
            f"### Task: Time-Dependent TSP (TDTSP)\n"
            f"Step: {len(path_history)}\n"
            f"Status: Current Node {curr_idx}, Unvisited {remaining_count}\n"
            f"Time: {time_val:.1f} (Index {s})\n"
            f"Momentum: {momentum_str}\n"
            f"History: {path_history[-10:]}\n\n"
            f"### Visual Legend (Image Reference):\n"
            f"- **Green Border**: Fast Route (Traffic is smooth).\n"
            f"- **Red Border**: Congested (Traffic is heavy/slow).\n"
            f"- **Grey Border**: Normal Traffic.\n\n"
            f"### Candidate Options (Traffic Aware):\n"
            f"Columns: [Static Dist] | [Dynamic Time (Congestion Status)] | [Relative Vector]\n"
            f"Minimize Total Time. Watch for Congested routes (High Time/Dist ratio).\n"
            f"\n{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Option Label (A, B...) to visit next."
        )

        status_str = (
            f"Step: {len(path_history)}\n"
            f"Status: Node {curr_idx}, Unvisited {remaining_count}\n"
            f"Time: {time_val:.1f}\n"
            f"Momentum: {momentum_str}\n"
        )
        
        # 2.5 Image Rendering
        image_save_path = None
        if image_obs == "path":
            uid = uuid.uuid4()
            image_save_dir = f"/root/autodl-tmp/image/tdtsp/"
            os.makedirs(image_save_dir, exist_ok=True)
            image_save_path = f"{image_save_dir}env{i}_step{len(path_history):03d}_{uid}.png"
            
        img_b64, image_rgb_np = render_tdtsp_smart_dual_view(
            locs=curr_locs,
            visited_mask=(visited[i]==1), # Ensure boolean mask
            current_node_idx=curr_idx,
            path_history=path_history,
            top_candidates=candidates,
            current_time=time_val,
            debug_save_path=image_save_path
        )
        
        # 2.6 Return Packaging
        obs_item = {
            "text": obs_text,
            "obs": status_str,
            "candidates": cand_section
        }
        
        if image_obs == "base64":
            obs_item["image"] = img_b64
        elif image_obs == "path":
            obs_item["image"] = image_save_path
        else:
            obs_item["image"] = image_rgb_np
            
        obs_list.append(obs_item)

    # Update TD with calculated top-k acts for loss calculation (if needed)
    if given_topk_acts is None and len(topk_acts_list) > 0:
        try:
             td["topk_acts"] = torch.tensor(np.array(topk_acts_list), device=td.device)
        except: pass
        
    return obs_list

# --- 1. 物理斥力分散算法 (通用 Helper) ---
def disperse_locations_global(locs, repulsion_radius=0.035, iterations=50):
    """
    对所有点进行物理斥力扩散，解决视觉重叠问题。
    
    Args:
        locs: 原始坐标 [N, 2]
        repulsion_radius: 斥力半径 (0-1空间下，0.035 约等于 3.5% 的地图跨度)
        iterations: 迭代次数
    Returns:
        new_locs: 分散后的坐标，保持拓扑结构但互不遮挡
    """
    new_locs = locs.copy().astype(np.float32)
    num_points = len(new_locs)
    
    if num_points < 2: return new_locs
    
    # 动态调整半径：如果点太多，适当减小半径防止炸开
    if num_points > 50:
        repulsion_radius *= 0.8
    
    for _ in range(iterations):
        # 计算所有点对之间的距离矩阵
        # diff[i, j] = pos[i] - pos[j]
        diff = new_locs[:, None, :] - new_locs[None, :, :] # [N, N, 2]
        dist = np.linalg.norm(diff, axis=-1) # [N, N]
        
        # 避免自身排斥和除零
        np.fill_diagonal(dist, np.inf)
        
        # 找到冲突点 (距离 < 半径)
        mask = dist < repulsion_radius
        if not np.any(mask): break # 如果没有冲突，提前结束
            
        # 计算斥力大小：距离越近斥力越大 (线性衰减)
        force_mag = (repulsion_radius - dist) * 0.5
        force_mag[~mask] = 0
        
        # 计算方向向量
        direction = diff / (dist[..., None] + 1e-9)
        
        # 合力 = Sum(方向 * 大小)
        total_force = np.sum(direction * force_mag[..., None], axis=1)
        
        # 更新位置 (阻尼系数 0.5 保证稳定)
        new_locs += total_force * 0.5
        
    return new_locs

def render_tdtsp_smart_dual_view(
    locs, visited_mask, current_node_idx, path_history, top_candidates, 
    current_time, img_height=336, debug_save_path=None
):
    """
    TDTSP 智能双视图渲染 (Clean Semantic Style - No Numbers).
    
    Changes:
    - Removed numerical time cost (e.g. "15s") from zoom view.
    - Added semantic text labels ("Fast", "Slow") below candidate boxes.
    - Relies on color coding (Green/Red) for quick visual cues.
    """
    locs = disperse_locations_global(locs)
    # --- 1. 配色方案 (BGR) ---
    COLOR_BG = (255, 255, 255)
    
    # 节点颜色
    COLOR_CURRENT_FILL = (220, 100, 50)    # Royal Blue
    COLOR_START_FILL = (50, 200, 50)       # Green
    COLOR_UNVISITED = (34, 34, 200)        # Deep Red
    COLOR_VISITED = (220, 220, 220)        # Light Grey
    
    # 交通状况颜色 (Border Colors)
    COLOR_TRAFFIC_FAST = (50, 200, 50)     # Green
    COLOR_TRAFFIC_NORMAL = (80, 80, 80)    # Dark Grey
    COLOR_TRAFFIC_SLOW = (0, 0, 255)       # Red
    
    # 辅助
    COLOR_TEXT = (10, 10, 10)
    COLOR_ZOOM_BOX = (0, 0, 255)
    COLOR_BORDER = (180, 180, 180)

    # 画布初始化
    img_width = img_height * 2 
    combined_canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    left_roi = combined_canvas[:, :img_height] 
    right_roi = combined_canvas[:, img_height:]

    # --- 2. 坐标变换逻辑 ---
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

    # --- 3. 全局视图计算 ---
    g_min = np.min(locs, axis=0)
    g_max = np.max(locs, axis=0)
    g_center = (g_min + g_max) / 2.0
    g_span = np.max(g_max - g_min)
    global_transform, _ = get_transform(g_center, g_span, img_height, padding=60)

    # --- 4. 智能聚焦逻辑 ---
    curr_pos = locs[current_node_idx]
    if top_candidates:
        cand_coords = np.array([[c['x'], c['y']] for c in top_candidates])
        dists = np.linalg.norm(cand_coords - curr_pos, axis=1)
        max_dist = np.max(dists)
        zoom_span = max(max_dist * 2.5, g_span * 0.05)
        zoom_span = min(zoom_span, g_span * 0.5)
    else:
        zoom_span = g_span * 0.2
    zoom_transform, (z_real_min, z_real_max) = get_transform(curr_pos, zoom_span, img_height, padding=40)

    # --- 5. 绘图函数 ---
    def draw_scene(canvas, transform_fn, view_bounds=None, is_zoomed=False):
        pts = transform_fn(locs)
        
        # === Layer 1: Gradient Path History ===
        if len(path_history) > 1:
            hist_to_draw = path_history if not is_zoomed else path_history[-15:]
            hist_pts = pts[hist_to_draw]
            num_segments = len(hist_pts) - 1
            for i in range(num_segments):
                pt_a = tuple(hist_pts[i])
                pt_b = tuple(hist_pts[i+1])
                ratio = i / max(num_segments, 1)
                gray_val = int(230 - (150 * ratio)) 
                color = (gray_val, gray_val, gray_val)
                thickness = 3 if is_zoomed else 2
                cv2.line(canvas, pt_a, pt_b, color, thickness, cv2.LINE_AA)

        # === Layer 2: Base Nodes ===
        node_radius = 6 if is_zoomed else 4
        for i in range(len(locs)):
            pt = tuple(pts[i])
            # Skip Current & Candidates
            is_candidate = False
            for c in top_candidates:
                if c['id'] == i: is_candidate = True; break
            
            if i == current_node_idx or is_candidate:
                continue
            
            if visited_mask[i]:
                cv2.circle(canvas, pt, node_radius, COLOR_VISITED, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, pt, node_radius, COLOR_UNVISITED, -1, cv2.LINE_AA)

        # === Layer 3: Candidates (Traffic Aware - Semantic) ===
        font_scale = 0.5 if is_zoomed else 0.4
        label_thickness = 1
        cand_label_box_pad = 6 if is_zoomed else 4
        
        # Calculate Pace for Color Logic
        all_paces = []
        if top_candidates:
            for c in top_candidates:
                dist = np.linalg.norm(locs[c['id']] - curr_pos)
                pace = c['tt'] / (dist + 1e-6)
                all_paces.append(pace)
            avg_pace = np.mean(all_paces) if all_paces else 1.0

        candidate_list = list(enumerate(top_candidates))
        for rank, cand in reversed(candidate_list):
            cand_idx = cand['id']
            cand_pt = tuple(pts[cand_idx])
            label = chr(65 + rank)
            
            # 3.1 Draw Dot
            cv2.circle(canvas, cand_pt, node_radius, COLOR_UNVISITED, -1, cv2.LINE_AA)
            
            # 3.2 Determine Traffic Color & Semantic Label
            dist = np.linalg.norm(locs[cand_idx] - curr_pos)
            pace = cand['tt'] / (dist + 1e-6)
            
            semantic_label = ""
            if pace < avg_pace * 0.8:
                border_color = COLOR_TRAFFIC_FAST
                border_width = 2
                semantic_label = "Fast"
            elif pace > avg_pace * 1.2:
                border_color = COLOR_TRAFFIC_SLOW
                border_width = 2
                semantic_label = "Slow"
            else:
                border_color = COLOR_TRAFFIC_NORMAL
                border_width = 2
                semantic_label = "Normal" # Usually hidden to avoid clutter
            
            # 3.3 Label Box
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)
            box_tl = (cand_pt[0] - w//2 - cand_label_box_pad, cand_pt[1] - h//2 - cand_label_box_pad)
            box_br = (cand_pt[0] + w//2 + cand_label_box_pad, cand_pt[1] + h//2 + cand_label_box_pad)
            
            cv2.rectangle(canvas, box_tl, box_br, (255, 255, 255), -1, cv2.LINE_AA) 
            cv2.rectangle(canvas, box_tl, box_br, border_color, border_width, cv2.LINE_AA)
            
            # 3.4 Label Text
            text_x = cand_pt[0] - w // 2
            text_y = cand_pt[1] + h // 2
            cv2.putText(canvas, label, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, label_thickness, cv2.LINE_AA)
            
            # 3.5 [Zoom View] Show Semantic Label (No Numbers)
            # Only display "Fast" or "Slow", hide "Normal" to reduce clutter
            if is_zoomed and semantic_label in ["Fast", "Slow"]:
                (cw, ch), _ = cv2.getTextSize(semantic_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cx = cand_pt[0] - cw // 2
                cy = box_br[1] + ch + 2 # Slightly below box
                cv2.putText(canvas, semantic_label, (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, border_color, 1, cv2.LINE_AA)

        # === Layer 4: Current Node ===
        curr_pt = tuple(pts[current_node_idx])
        curr_size = 10 if is_zoomed else 6
        cv2.rectangle(canvas, (curr_pt[0]-curr_size-2, curr_pt[1]-curr_size-2), 
                      (curr_pt[0]+curr_size+2, curr_pt[1]+curr_size+2), (255,255,255), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (curr_pt[0]-curr_size, curr_pt[1]-curr_size), 
                      (curr_pt[0]+curr_size, curr_pt[1]+curr_size), COLOR_CURRENT_FILL, -1, cv2.LINE_AA)

    # --- 6. Execute Drawing ---
    draw_scene(left_roi, global_transform, is_zoomed=False)
    
    # Focus Box
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
    
    # --- 7. Legend ---
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
        current_y = draw_item(current_y, "Current Node", lambda x, y: cv2.rectangle(img, (x-6, y-6), (x+6, y+6), COLOR_CURRENT_FILL, -1))
        
        # Traffic Legend
        def icon_fast(x, y):
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), (255,255,255), -1)
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), COLOR_TRAFFIC_FAST, 2)
        current_y = draw_item(current_y, "Fast Route", icon_fast)
        
        def icon_slow(x, y):
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), (255,255,255), -1)
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), COLOR_TRAFFIC_SLOW, 2)
        current_y = draw_item(current_y, "Congested / Slow", icon_slow)
        
    draw_legend(left_roi)
    
    # Titles & Time
    cv2.rectangle(combined_canvas, (img_height, 0), (img_width-1, img_height-1), COLOR_BORDER, 4)
    cv2.putText(left_roi, "Global Map", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2, cv2.LINE_AA)
    
    cv2.putText(right_roi, "Egocentric View", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2, cv2.LINE_AA)
    cv2.putText(right_roi, f"Time: {current_time:.1f}s", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

    # --- 8. Output ---
    _, buffer = cv2.imencode('.png', combined_canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    img_rgb_np = cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2RGB)

    if debug_save_path is not None:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, combined_canvas)
    
    return b64_str, img_rgb_np

# 假设 helper 函数已定义
# from utils import _to_numpy, get_spatial_desc, render_tdtsptw_smart_dual_view
def build_obs_tdtsp_tw(
    td, 
    env_num: int, 
    trajectory: list = None, 
    top_k: int = 24, 
    image_obs: str = "path",
    given_topk_acts = None
) -> list:
    """
    Observation builder for TDTSPTW (Time-Dependent + Time Windows).
    Features: Soft Constraint Handling (Valid > Late), Urgency analysis, Wait Cost visibility.
    """
    obs_list = []
    
    # --- 1. Data Extraction ---
    locs = _to_numpy(td["locs"])
    current_node = _to_numpy(td["current_node"]) 
    current_time = _to_numpy(td["current_time"]) 
    visited = _to_numpy(td.get("visited", torch.zeros_like(td["action_mask"]))) 
    action_mask = _to_numpy(td["action_mask"]) 
    
    # TDTSPTW Specifics
    time_windows = _to_numpy(td["time_windows"])
    matrix = td["travel_time_matrix"]
    duration = td["time_step_duration"]
    
    # SFT Injection Setup
    if "topk_acts" not in td.keys():
        td["topk_acts"] = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)
    
    topk_acts_list = []
    if given_topk_acts is not None:
        given_topk_acts = _to_numpy(given_topk_acts)

    # --- 2. Environment Loop ---
    for i in range(env_num):
        # 2.1 Basic State & Momentum
        curr_locs = locs[i]
        curr_idx = int(current_node[i])
        curr_pos = curr_locs[curr_idx]
        time_val = float(current_time[i]) if hasattr(current_time[i], 'item') else float(current_time[i])
        
        # Matrix Lookup
        if hasattr(duration, 'dim') and duration.dim() > 0:
            curr_duration = float(duration[i])
        else:
            curr_duration = float(duration)
        time_step_idx = int(time_val // curr_duration)
        max_s = matrix.shape[-1] - 1
        s = min(time_step_idx, max_s)
        
        # Momentum
        path_history = []
        momentum_str = "None (Start)"
        if trajectory is not None and len(trajectory) > 0:
            for t_step in trajectory:
                val = t_step[i]
                if hasattr(val, 'item'): val = val.item()
                path_history.append(int(val))
        
        if len(path_history) == 0 or path_history[-1] != curr_idx:
            path_history.append(curr_idx)
            
        if len(path_history) >= 2:
            prev_idx = path_history[-2]
            prev_pos = curr_locs[prev_idx]
            m_dx = curr_pos[0] - prev_pos[0]
            m_dy = curr_pos[1] - prev_pos[1]
            m_dir = get_spatial_desc(m_dx, m_dy)
            momentum_str = f"Heading {m_dir} from Node {prev_idx}"

        # 2.2 Candidate Analysis
        candidates = []
        
        # Get Unvisited
        # 注意：这里我们只依赖 'unvisited' 逻辑，不再依赖 mask 里的 'valid' 逻辑
        # 因为在全死的情况下，我们需要捡回那些被 mask 掉的点
        # 通常 mask = unvisited & valid。如果全 invalid，mask 可能全 0。
        # 为了鲁棒性，我们直接找 visited == 0 的点。
        unvisited_indices = np.where(visited[i] == 0)[0] 
        unvisited_indices = unvisited_indices[unvisited_indices != curr_idx]
        
        # Extract Travel Times
        if matrix.dim() == 4:
            tt_slice = matrix[i, curr_idx, :, s]
        else:
            tt_slice = matrix[curr_idx, :, s]
        if hasattr(tt_slice, 'cpu'): tt_slice = tt_slice.cpu().numpy()
        
        # 这里需要注意索引对齐，unvisited_tt 需要对应 unvisited_indices
        unvisited_tt = tt_slice[unvisited_indices]

        # Branch A: SFT Injection
        if given_topk_acts is not None:
            indices = given_topk_acts[i]
            target_indices = [int(x) for x in indices if x != -1]
            strategy_tag = "inject"
        else:
            target_indices = unvisited_indices
            strategy_tag = "greedy"

        # Calculate Detailed Metrics for ALL potential candidates (Soft Filter)
        temp_candidates = []
        for j, idx_val in enumerate(target_indices):
            uid = int(idx_val)
            
            # Physics
            dist_val = np.linalg.norm(curr_locs[uid] - curr_pos)
            
            # Time & Window
            try:
                # 如果是 Greedy 模式，直接用切片；如果是 SFT，可能需要查表
                if strategy_tag == "greedy":
                    tt = float(unvisited_tt[j])
                else:
                    tt = float(tt_slice[uid])
            except:
                tt = 9999.0 

            tw = time_windows[i, uid] # [start, end]
            tw_start, tw_end = float(tw[0]), float(tw[1])
            
            arrival_time = time_val + tt
            wait_time = max(0.0, tw_start - arrival_time)
            
            start_time = max(arrival_time, tw_start)
            is_late = start_time > tw_end
            slack = tw_end - start_time
            
            # [MODIFIED] Always append, do not filter out 'is_late' here.
            # We filter/sort later.
            temp_candidates.append({
                'id': uid,
                'dist': dist_val,
                'tt': tt,
                'wait': wait_time,
                'eta': arrival_time,
                'slack': slack,
                'tw_end': tw_end,
                'is_late': is_late,
                'x': curr_locs[uid][0],
                'y': curr_locs[uid][1],
                'strategy': strategy_tag
            })

        # --- 2.3 Sorting & Selection (Hybrid Strategy) ---
        if given_topk_acts is None:
            # 排序逻辑：
            # 1. 核心目标：在保证不造成新的 Late (is_late=False) 的前提下，优先满足紧迫需求 (Small Slack)。
            # 2. 次要目标：合并考虑效率 (Efficiency)，减少路程和等待。
            # 3. 紧迫度采用分段式 Bonus (奖励) 机制，从 Efficiency 中扣除：
            #    - Slack >= 600 (宽裕): Bonus = 0. 按照 Efficiency 排序。
            #    - 300 <= Slack < 600 (关注): Bonus 线性增长。
            #    - Slack < 300 (非常紧迫): Bonus 极大 (Offset + Slope)，确保 Score 为负值，强制排在前面。

            def calculate_score(c):
                # 提取基础数据
                travel_time = c['tt']
                wait_time = c['wait']
                slack = c['slack']
                
                # 定义阈值 (单位: 秒)
                # PANIC_THRESHOLD: 低于这个值，说明必须马上出发，否则来不及
                # 设定逻辑：大约等于平均 travel_time * 1.5，或者直接设为 10-15 分钟
                PANIC_THRESHOLD = 1200.0   # < 20 min: 恐慌区 (生死存亡)
                URGENT_THRESHOLD = 2400.0 # < 40 min: 紧急区 (兼顾效率与紧迫)
                
                # --- 层级 1: 恐慌区 (Panic Zone) ---
                # 逻辑：只要 Slack 极小，完全不看距离远近，谁 Slack 最小谁排最前 (EDD规则)
                if slack < PANIC_THRESHOLD:
                    # 使用一个巨大的负数 Offset，确保它们永远排在 Urgent/Safe 节点之前
                    # 这里的 Score 只由 slack 决定。Slack 越小，Score 越负 (越小)
                    # 注意：这里移除了 travel_time，防止因为距离远而放弃救援
                    return -1000000.0 + slack 

                # --- 层级 2: 紧急区 (Urgent Zone) ---
                # 逻辑：虽然急，但还有喘息机会。此时需要平衡“距离”和“紧迫度”。
                # 我们希望：距离越短越好，Slack 越大越好(意味着越不急)
                # Score = Distance + Penalty(Slack)
                elif slack < URGENT_THRESHOLD:
                    # 这是一个指数型惩罚，随着 slack 变小，惩罚急剧增加
                    # 比线性奖励更平滑，且在接近 PANIC 区时会有极强的吸力
                    urgency_weight = 1000.0 * (1.0 - (slack / URGENT_THRESHOLD))
                    return (travel_time + wait_time) - urgency_weight - 50000.0 # Offset 确保排在 Safe 之前

                # --- 层级 3: 安全区 (Safe Zone) ---
                # 逻辑：时间很充裕，完全按照效率（距离/时间）决策
                else:
                    # 纯粹的 TSP 贪婪逻辑：找最近的
                    # 可以加上微量的 slack 惩罚，防止一直把某个人拖到最后
                    return (travel_time + wait_time) - (slack * 0.01)

            # 排序逻辑保持不变
            candidates = sorted(temp_candidates, key=lambda x: (
                x['is_late'],           # 1. 死的放最后
                calculate_score(x)      # 2. Score 越小越好
            ))

            
            # 取 Top-K
            candidates = candidates[:top_k]
            
            # Save Acts
            sorted_indices = [c['id'] for c in candidates]
            valid_len = len(sorted_indices)
            padded = np.array(sorted_indices + [-1]*(top_k - valid_len))
            topk_acts_list.append(padded)
        else:
            candidates = temp_candidates

        # --- 2.4 Text Generation (Rich Semantics with LATE Handling) ---
        cand_str_list = []
        for rank, cand in enumerate(candidates):
            label = chr(65 + rank) if rank < 26 else f"Opt{rank}"
            
            # Spatial Vector
            dx = cand['x'] - curr_pos[0]
            dy = cand['y'] - curr_pos[1]
            bearing = get_spatial_desc(dx, dy)
            
            # Status Logic
            status_tags = []
            
            if cand['is_late']:
                # [NEW] Explicit Late Tag
                status_tags.append("**[LATE / PENALTY]**")
            else:
                # Normal Logic
                if cand['slack'] < 300: 
                    status_tags.append("**[URGENT]**")
                
                if cand['wait'] > cand['tt']: 
                    status_tags.append("(Long Wait)")
                elif cand['wait'] > 0:
                    status_tags.append("(Wait)")
                else:
                    status_tags.append("(Ready)")
                
            status_str = " ".join(status_tags)
            
            # Formatting
            cand_str_list.append(
                f"Option {label} [Node {cand['id']}]: "
                f"Dist: {cand['dist']*100:.1f} | "
                f"Cost: Travel {cand['tt']:.1f} + Wait {cand['wait']:.1f} | "
                f"Slack: {cand['slack']:.1f} {status_str} | "
                f"Vec: [{dx:+.2f}, {dy:+.2f}] ({bearing})"
            )
            
        cand_section = "\n".join(cand_str_list)
        remaining = len(unvisited_indices)

        obs_text = (
            f"### Task: TDTSP with Time Windows (TDTSPTW)\n"
            f"Step: {len(path_history)}\n"
            f"Status: Current Node {curr_idx}, Unvisited {remaining}\n"
            f"Time: {time_val:.1f}\n"
            f"Momentum: {momentum_str}\n"
            f"History: {path_history[-10:]}\n\n"
            f"### Visual Legend (Image Reference):\n"
            f"- **Green Border**: Fast Route / Ready.\n"
            f"- **Red Border**: Congested / **[LATE]** / **[URGENT]**.\n"
            f"- **Orange Border**: Waiting Required.\n\n"
            f"### Candidate Options:\n"
            f"Prioritize Valid nodes. Avoid **[LATE]** nodes unless no other choice.\n"
            f"\n{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Option Label (A, B...) to visit next."
        )

        status_str = (
            f"Step: {len(path_history)}\n"
            f"Status: Node {curr_idx}, Unvisited {remaining}\n"
            f"Time: {time_val:.1f}\n"
            f"Momentum: {momentum_str}\n"
        )
        
        # --- 2.5 Visualizer ---
        image_save_path = None
        if image_obs == "path":
            uid = uuid.uuid4()
            image_save_dir = f"/root/autodl-tmp/image/tdtsptw/"
            os.makedirs(image_save_dir, exist_ok=True)
            image_save_path = f"{image_save_dir}env{i}_step{len(path_history):03d}_{uid}.png"
        
        # render 函数内部已经处理了 is_late=True 显示红色边框的逻辑
        # 只要传入的 candidates 里包含 is_late 字段即可
        img_b64, image_rgb_np = render_tdtsptw_smart_dual_view(
            locs=curr_locs,
            visited_mask=(visited[i]==1),
            current_node_idx=curr_idx,
            path_history=path_history,
            top_candidates=candidates, 
            current_time=time_val,
            time_windows=time_windows[i],
            debug_save_path=image_save_path
        )
        
        # --- 2.6 Return ---
        obs_item = {
            "text": obs_text, 
            "obs": status_str, 
            "candidates": cand_section
        }
        
        if image_obs == "base64":
            obs_item["image"] = img_b64
        elif image_obs == "path":
            obs_item["image"] = image_save_path
        else:
            obs_item["image"] = image_rgb_np
            
        obs_list.append(obs_item)

    if given_topk_acts is None and len(topk_acts_list) > 0:
        try:
             td["topk_acts"] = torch.tensor(np.array(topk_acts_list), device=td.device)
        except: pass
        
    return obs_list


def render_tdtsptw_smart_dual_view(
    locs, visited_mask, current_node_idx, path_history, top_candidates, 
    current_time, time_windows, img_height=336, debug_save_path=None
):
    """
    TDTSPTW 智能双视图渲染 (Clean Semantic Style - No Numbers).
    
    Changes:
    - Removed numerical TW text from base nodes.
    - Removed numerical Slack/Wait values from candidate labels.
    - Labels now show semantic tags only: "URGENT", "Wait", "Ready".
    """
    locs = disperse_locations_global(locs)
    # --- 1. 配色方案 (BGR) ---
    COLOR_BG = (255, 255, 255)
    
    # 节点基础色
    COLOR_CURRENT_FILL = (220, 100, 50)    # Royal Blue
    COLOR_UNVISITED = (34, 34, 200)        # Deep Red
    COLOR_VISITED = (220, 220, 220)        # Light Grey
    
    # 状态语义色 (Border Colors) - Aligned with Text Obs
    COLOR_TW_URGENT = (0, 0, 255)          # Red (Low Slack / Late)
    COLOR_TW_WAIT = (0, 165, 255)          # Orange (Wait)
    COLOR_TW_READY = (50, 200, 50)         # Green (Safe/Ready)
    
    # 辅助色
    COLOR_TEXT = (10, 10, 10)
    COLOR_ZOOM_BOX = (0, 0, 255)
    COLOR_BORDER = (180, 180, 180)

    # 画布初始化
    img_width = img_height * 2 
    combined_canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    left_roi = combined_canvas[:, :img_height] 
    right_roi = combined_canvas[:, img_height:]

    # --- 2. 坐标变换逻辑 ---
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

    # --- 3. 全局视图计算 ---
    g_min = np.min(locs, axis=0)
    g_max = np.max(locs, axis=0)
    g_center = (g_min + g_max) / 2.0
    g_span = np.max(g_max - g_min)
    global_transform, _ = get_transform(g_center, g_span, img_height, padding=60)

    # --- 4. 智能聚焦逻辑 ---
    curr_pos = locs[current_node_idx]
    if top_candidates:
        cand_coords = np.array([[c['x'], c['y']] for c in top_candidates])
        dists = np.linalg.norm(cand_coords - curr_pos, axis=1)
        max_dist = np.max(dists)
        zoom_span = max(max_dist * 2.5, g_span * 0.05)
        zoom_span = min(zoom_span, g_span * 0.5)
    else:
        zoom_span = g_span * 0.2
    zoom_transform, (z_real_min, z_real_max) = get_transform(curr_pos, zoom_span, img_height, padding=40)

    # --- 5. 绘图函数 ---
    def draw_scene(canvas, transform_fn, view_bounds=None, is_zoomed=False):
        pts = transform_fn(locs)
        
        # === Layer 1: Gradient Path History ===
        if len(path_history) > 1:
            hist_to_draw = path_history if not is_zoomed else path_history[-15:]
            hist_pts = pts[hist_to_draw]
            num_segments = len(hist_pts) - 1
            for i in range(num_segments):
                pt_a = tuple(hist_pts[i])
                pt_b = tuple(hist_pts[i+1])
                ratio = i / max(num_segments, 1)
                gray_val = int(230 - (150 * ratio)) 
                color = (gray_val, gray_val, gray_val)
                thickness = 3 if is_zoomed else 2
                cv2.line(canvas, pt_a, pt_b, color, thickness, cv2.LINE_AA)

        # === Layer 2: Base Nodes (No Numbers) ===
        node_radius = 6 if is_zoomed else 4
        for i in range(len(locs)):
            pt = tuple(pts[i])
            # Skip Current & Candidates
            is_candidate = False
            for c in top_candidates:
                if c['id'] == i: is_candidate = True; break
            
            if i == current_node_idx or is_candidate:
                continue
            
            if visited_mask[i]:
                cv2.circle(canvas, pt, node_radius, COLOR_VISITED, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, pt, node_radius, COLOR_UNVISITED, -1, cv2.LINE_AA)
                # [MODIFIED] Removed numerical TW text display

        # === Layer 3: Candidates (Semantic Colors) ===
        font_scale = 0.5 if is_zoomed else 0.4
        label_thickness = 1
        cand_label_box_pad = 6 if is_zoomed else 4
        
        candidate_list = list(enumerate(top_candidates))
        for rank, cand in reversed(candidate_list):
            cand_idx = cand['id']
            cand_pt = tuple(pts[cand_idx])
            label = chr(65 + rank)
            
            # 3.1 Draw Dot
            cv2.circle(canvas, cand_pt, node_radius, COLOR_UNVISITED, -1, cv2.LINE_AA)
            
            # 3.2 Determine Status & Color
            slack = cand.get('slack', 9999)
            wait = cand.get('wait', 0)
            is_late = cand.get('is_late', False)
            
            semantic_label = "Ready"
            if is_late or slack < 15.0:  # URGENT / LATE
                border_color = COLOR_TW_URGENT
                border_width = 2
                semantic_label = "URGENT" # Semantic Label
            elif wait > 0:               # WAIT
                border_color = COLOR_TW_WAIT
                border_width = 2
                semantic_label = "Wait"   # Semantic Label
            else:                        # READY
                border_color = COLOR_TW_READY
                border_width = 2
                semantic_label = "Ready"
            
            # 3.3 Label Box
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)
            box_tl = (cand_pt[0] - w//2 - cand_label_box_pad, cand_pt[1] - h//2 - cand_label_box_pad)
            box_br = (cand_pt[0] + w//2 + cand_label_box_pad, cand_pt[1] + h//2 + cand_label_box_pad)
            
            cv2.rectangle(canvas, box_tl, box_br, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.rectangle(canvas, box_tl, box_br, border_color, border_width, cv2.LINE_AA)
            
            # 3.4 Label Text
            text_x = cand_pt[0] - w // 2
            text_y = cand_pt[1] + h // 2
            cv2.putText(canvas, label, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, label_thickness, cv2.LINE_AA)
            
            # 3.5 [Zoom View] Show Semantic Label (No Numbers)
            # Only show text if it's NOT just "Ready" (to keep it clean)
            if is_zoomed and semantic_label != "Ready":
                (cw, ch), _ = cv2.getTextSize(semantic_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cx = cand_pt[0] - cw // 2
                cy = box_br[1] + ch + 2
                cv2.putText(canvas, semantic_label, (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, border_color, 1, cv2.LINE_AA)

        # === Layer 4: Current Node ===
        curr_pt = tuple(pts[current_node_idx])
        curr_size = 10 if is_zoomed else 6
        cv2.rectangle(canvas, (curr_pt[0]-curr_size-2, curr_pt[1]-curr_size-2), 
                      (curr_pt[0]+curr_size+2, curr_pt[1]+curr_size+2), (255,255,255), -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (curr_pt[0]-curr_size, curr_pt[1]-curr_size), 
                      (curr_pt[0]+curr_size, curr_pt[1]+curr_size), COLOR_CURRENT_FILL, -1, cv2.LINE_AA)

    # --- 6. Execute Drawing ---
    draw_scene(left_roi, global_transform, is_zoomed=False)
    
    # Focus Box
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
    
    # --- 7. Legend ---
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
        current_y = draw_item(current_y, "Current Node", lambda x, y: cv2.rectangle(img, (x-6, y-6), (x+6, y+6), COLOR_CURRENT_FILL, -1))
        
        # TW Legend
        def icon_urgent(x, y):
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), (255,255,255), -1)
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), COLOR_TW_URGENT, 2)
        current_y = draw_item(current_y, "Urgent (Low Slack)", icon_urgent)
        
        def icon_wait(x, y):
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), (255,255,255), -1)
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), COLOR_TW_WAIT, 2)
        current_y = draw_item(current_y, "Waiting Required", icon_wait)

        def icon_ready(x, y):
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), (255,255,255), -1)
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), COLOR_TW_READY, 2)
        current_y = draw_item(current_y, "Ready / Safe", icon_ready)
        
    draw_legend(left_roi)
    
    # Titles & Time
    cv2.rectangle(combined_canvas, (img_height, 0), (img_width-1, img_height-1), COLOR_BORDER, 4)
    cv2.putText(left_roi, "Global Map", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2, cv2.LINE_AA)
    
    cv2.putText(right_roi, "Egocentric View", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2, cv2.LINE_AA)
    cv2.putText(right_roi, f"Time: {current_time:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

    # --- 8. Output ---
    _, buffer = cv2.imencode('.png', combined_canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    img_rgb_np = cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2RGB)
    
    if debug_save_path:
        os.makedirs(os.path.dirname(debug_save_path) if os.path.dirname(debug_save_path) else ".", exist_ok=True)
        cv2.imwrite(debug_save_path, combined_canvas)
    
    return b64_str, img_rgb_np

def build_obs_tdvrp(
    td, 
    env_num: int, 
    trajectory: list = None, 
    top_k: int = 24, 
    image_obs: str = "path",
    penalty_value: float = 0.0,
    given_topk_acts = None
) -> list:
    """
    Observation builder for TDVRPEnv (Cost + Time Window + Strategic Depot).
    Highlights: Cost breakdown (Fixed vs Labor), Urgency (Slack), and Spatial Momentum.
    """
    obs_list = []
    
    # --- 1. Data Extraction ---
    locs = _to_numpy(td["locs"])
    current_node = _to_numpy(td["current_node"])
    current_time = _to_numpy(td["current_time"])
    visited = _to_numpy(td.get("visited", td["action_mask"]))
    action_mask = _to_numpy(td["action_mask"])
    
    # TDVRP Specifics
    time_windows = _to_numpy(td["time_windows"])
    matrix = td["travel_time_matrix"]
    duration = td["time_step_duration"]
    
    # Cost Constants (From your definition)
    service_time = 180.0 
    FIXED_COST = 200.0
    PER_HOUR_COST = 20.0
    
    # SFT / Teacher Forcing Setup
    if "topk_acts" not in td.keys():
        td["topk_acts"] = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)
    
    topk_acts_list = []
    if given_topk_acts is not None:
        given_topk_acts = _to_numpy(given_topk_acts)

    # --- 2. Environment Loop ---
    for i in range(env_num):
        # 2.1 Basic State
        curr_locs = locs[i]
        curr_idx = int(current_node[i])
        curr_pos = curr_locs[curr_idx]
        time_val = float(current_time[i]) if hasattr(current_time[i], 'item') else float(current_time[i])
        
        # Matrix Lookup Index
        if hasattr(duration, 'dim') and duration.dim() > 0:
            curr_duration = float(duration[i])
        else:
            curr_duration = float(duration)
        time_step_idx = int(time_val // curr_duration)
        max_s = matrix.shape[-1] - 1
        s = min(time_step_idx, max_s)
        
        # 2.2 Momentum Logic (History Analysis)
        path_history = []
        momentum_str = "None (At Depot)"
        if trajectory is not None and len(trajectory) > 0:
            for t_step in trajectory:
                val = t_step[i]
                if hasattr(val, 'item'): val = val.item()
                path_history.append(int(val))
        
        if len(path_history) >= 2 and curr_idx != 0:
            prev_idx = path_history[-2]
            prev_pos = curr_locs[prev_idx]
            m_dx = curr_pos[0] - prev_pos[0]
            m_dy = curr_pos[1] - prev_pos[1]
            m_dir = get_spatial_desc(m_dx, m_dy)
            momentum_str = f"Heading {m_dir} from Node {prev_idx}"
        elif curr_idx == 0 and len(path_history) > 1:
            momentum_str = "Returned to Depot (Reset)"

        # 2.3 Candidate Generation
        candidates = []
        
        # Identify Targets: Unvisited Customers AND Depot (if valid)
        unvisited_indices = np.where(action_mask[i] == 1)[0]
        
        # VRP Logic: Always consider Depot (0) if we are not currently there
        # This allows "Early Return" strategy even if mask technically handles it
        if curr_idx != 0:
            if 0 not in unvisited_indices:
                # Only add if not strictly forbidden by mask (some envs ban return until visited all)
                # But usually for VRP "End Route" is always an option. 
                # Let's trust the mask mostly, but ensure we don't crash if 0 is added.
                pass
        
        # Matrix Slice
        if matrix.dim() == 4:
            tt_slice = matrix[i, curr_idx, :, s]
        else:
            tt_slice = matrix[curr_idx, :, s]
        if hasattr(tt_slice, 'cpu'): tt_slice = tt_slice.cpu().numpy()

        # Branching for SFT
        if given_topk_acts is not None:
            target_indices = [int(x) for x in given_topk_acts[i] if x != -1]
            strategy_tag = "inject"
        else:
            target_indices = unvisited_indices
            strategy_tag = "greedy"

        # 2.4 Metric Calculation
        temp_candidates = []
        for uid in target_indices:
            uid = int(uid)
            if uid == curr_idx: continue
            
            # Physics
            dist_val = np.linalg.norm(curr_locs[uid] - curr_pos)
            
            # Time & Logic
            try:
                tt = float(tt_slice[uid])
            except:
                tt = 9999.0
            
            tw = time_windows[i, uid] # [start, end]
            tw_start, tw_end = float(tw[0]), float(tw[1])
            
            eta = time_val + tt
            is_late = eta > tw_end # Hard Constraint
            
            wait_time = max(0.0, tw_start - eta)
            ready_time = max(eta, tw_start)
            
            # Cost Calculation (Preserving your logic)
            # Departure includes service time ONLY for customers (uid > 0)
            departure_time = ready_time + (service_time if uid > 0 else 0.0)
            
            # New Trip Logic: Leaving Depot (0 -> Customer) triggers Fixed Cost
            is_new_trip = (curr_idx == 0) and (uid > 0)
            
            fixed_cost_val = FIXED_COST if is_new_trip else 0.0
            # Labor Cost: Pay for Travel + Wait + Service
            labor_cost_val = (departure_time - time_val) / 3600.0 * PER_HOUR_COST
            total_cost = fixed_cost_val + labor_cost_val
            
            slack = tw_end - eta
            if penalty_value != 0 or not is_late: # Filter out infeasible options immediately
                if is_late:
                    print(f"Late for Node {uid}: ETA {eta:.2f} > TW End {tw_end:.2f}")
                    
                temp_candidates.append({
                    'id': uid,
                    'dist': dist_val,
                    'tt': tt,
                    'cost': total_cost,
                    'fixed': fixed_cost_val,
                    'wait': wait_time,
                    'eta': eta,
                    'is_late': is_late,
                    'slack': slack,
                    'is_depot': (uid == 0),
                    'x': curr_locs[uid][0],
                    'y': curr_locs[uid][1],
                    'strategy': strategy_tag
                })

        # 2.5 Sorting
        if given_topk_acts is None:
            # Sort Strategy:
            # 0. 提取 Depot 和 客户
            depot_cand = None
            customer_candidates = []
            
            for c in temp_candidates:
                if c['is_depot']:
                    depot_cand = c
                else:
                    customer_candidates.append(c)
            
            # 1. 准备两个排序列表
            # 列表 A: 经济优先 (只看钱) - 用于省成本
            # 优先看是否迟到(0/1)，再看成本
            list_economy = sorted(customer_candidates, key=lambda x: (x.get('is_late', False), x['cost']))
            
            # 列表 B: 紧迫优先 (只看时间) - 用于防超时
            # 优先看是否迟到，再看截止时间(tw_end)或松弛度(slack)
            # 注意：这里我们希望尽早去那些快过期的点
            list_urgency = sorted(customer_candidates, key=lambda x: (x.get('is_late', False), x['slack']))
            
            # 2. 混合采样 (Interleaved Sampling)
            final_candidates = []
            seen_ids = set()
            
            # 如果我们不在 Depot，预留 1 个位置给 Depot
            # 如果已经在 Depot，depot_cand 可能不在 temp_candidates 里(被过滤了)，或者不需要显示
            limit = top_k
            if depot_cand and curr_idx != 0:
                limit = top_k - 1
            
            # 双指针交替提取
            idx_eco = 0
            idx_urg = 0
            
            while len(final_candidates) < limit:
                added_any = False
                
                # [Round 1] 取一个最便宜的
                if idx_eco < len(list_economy):
                    cand = list_economy[idx_eco]
                    if cand['id'] not in seen_ids:
                        final_candidates.append(cand)
                        seen_ids.add(cand['id'])
                        added_any = True
                    idx_eco += 1
                
                if len(final_candidates) >= limit: break
                
                # [Round 2] 取一个最紧急的
                if idx_urg < len(list_urgency):
                    cand = list_urgency[idx_urg]
                    if cand['id'] not in seen_ids:
                        final_candidates.append(cand)
                        seen_ids.add(cand['id'])
                        added_any = True
                    idx_urg += 1
                
                # 如果两边都遍历完了，退出
                if not added_any and idx_eco >= len(list_economy) and idx_urg >= len(list_urgency):
                    break
            
            # 3. 强制加入 Depot (作为最后一个选项，或保底)
            if depot_cand and curr_idx != 0:
                # 只有当它还没被加进去的时候 (通常 Depot 不在 customer list 里，肯定没加)
                if depot_cand['id'] not in seen_ids:
                    final_candidates.append(depot_cand)

            # 4. 最终展示排序 (Presentation Sorting)
            # 虽然选点是用混合策略，但展示给模型看的时候，建议按统一逻辑排序，方便阅读
            # 策略：Depot 垫底，其余按 Cost 排序 (这是最直观的)
            # 这样模型会看到：一堆按价格排序的客户（其中夹杂着很贵但很急的），最后是一个回库选项。
            
            def presentation_sort(x):
                # 1. Depot 放最后 (rank=2)
                if x['is_depot']: return (2, 0)
                # 2. 迟到的放倒数第二 (rank=1)
                if x.get('is_late', False): return (1, x['cost'])
                # 3. 正常的放前面 (rank=0)，按 Cost 升序
                return (0, x['cost'])
                
            final_candidates.sort(key=presentation_sort)
            
            candidates = final_candidates
            
            # Save Acts for Loss Calculation
            sorted_indices = [c['id'] for c in candidates]
            valid_len = len(sorted_indices)
            padded = np.array(sorted_indices + [-1]*(top_k - valid_len))
            topk_acts_list.append(padded)
        else:
            candidates = temp_candidates

        # 2.6 Text Generation
        cand_str_list = []
        for rank, cand in enumerate(candidates):
            label = chr(65 + rank) if rank < 26 else f"Opt{rank}"
            
            dx = cand['x'] - curr_pos[0]
            dy = cand['y'] - curr_pos[1]
            bearing = get_spatial_desc(dx, dy)
            
            # Status Tags
            tags = []
            if cand['slack'] < 300.0: # e.g. less than 5 mins slack
                tags.append("**[URGENT]**")
            
            if cand['wait'] > 600.0: # Long wait
                tags.append("(Long Wait)")
            elif cand['wait'] > 0:
                tags.append(f"(Wait {cand['wait']:.0f}s)")
            
            tag_str = " ".join(tags)
            
            # Cost Formatting
            cost_str = f"${cand['cost']:.2f}"
            if cand['fixed'] > 0:
                cost_str += " (New Trip)"
            
            # Row Generation
            if cand['is_depot']:
                row_str = (
                    f"Option {label} **[RETURN TO BASE]**: "
                    f"Cost: {cost_str} | "
                    f"Action: Terminate route. Return to Node 0."
                )
            else:
                row_str = (
                    f"Option {label} [Node {cand['id']}]: "
                    f"Cost: {cost_str} | "
                    f"Slack: {cand['slack']:.0f}s {tag_str} | "
                    f"Vec: [{dx:+.2f}, {dy:+.2f}] ({bearing})"
                )
            cand_str_list.append(row_str)
            
        cand_section = "\n".join(cand_str_list)
        
        # Unvisited Count (excluding depot logic)
        remaining_nodes = len(unvisited_indices)
        if 0 in unvisited_indices: remaining_nodes -= 1

        obs_text = (
            f"### Task: TDVRP (Minimize Cost with Time Windows)\n"
            f"Step: {len(path_history)}\n"
            f"Status: Current Node {curr_idx}, Unvisited Customers {remaining_nodes}\n"
            f"Time: {time_val:.1f}s\n"
            f"Momentum: {momentum_str}\n"
            f"Costs: Fixed ${FIXED_COST} (New Trip) | Labor ${PER_HOUR_COST}/hr\n\n"
            f"### Candidate Options (Financial & Urgency Analysis):\n"
            f"Goal: Visit all customers. Minimize Total Cost ($). Watch for [URGENT] slack.\n"
            f"\n{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select Option (A, B...) to visit next. Use [RETURN TO BASE] only if necessary."
        )

        status_str = (
            f"Step: {len(path_history)}\n"
            f"Status: Node {curr_idx}, Left {remaining_nodes}\n"
            f"Time: {time_val:.1f}s\n"
            f"Momentum: {momentum_str}\n"
        )

        # 2.7 Visualization
        image_save_path = None
        if image_obs == "path":
            uid = uuid.uuid4()
            image_save_dir = f"/root/autodl-tmp/image/tdvrp/"
            os.makedirs(image_save_dir, exist_ok=True)
            image_save_path = f"{image_save_dir}env{i}_step{len(path_history):03d}_{uid}.png"
            
        img_b64, image_rgb_np = render_tdvrp_smart_dual_view(
            locs=curr_locs,
            visited_mask=(visited[i]==1),
            current_node_idx=curr_idx,
            path_history=path_history,
            top_candidates=candidates,
            current_time=time_val,
            time_windows=time_windows[i],
            debug_save_path=image_save_path
        )
        
        obs_item = {
            "text": obs_text, 
            "obs": status_str, 
            "candidates": cand_section
        }
        
        if image_obs == "base64":
            obs_item["image"] = img_b64
        elif image_obs == "path":
            obs_item["image"] = image_save_path
        else:
            obs_item["image"] = image_rgb_np
            
        obs_list.append(obs_item)

    if given_topk_acts is None and len(topk_acts_list) > 0:
        try:
             td["topk_acts"] = torch.tensor(np.array(topk_acts_list), device=td.device)
        except: pass
        
    return obs_list
        
def render_tdvrp_smart_dual_view(
    locs, visited_mask, current_node_idx, path_history, top_candidates, 
    current_time, time_windows, img_height=336, debug_save_path=None
):
    """
    TDVRPTW 智能双视图渲染 (Clean Semantic Style - No Numbers).
    
    Key Changes:
    - Removed numerical Time Windows from base nodes in zoom view.
    - Removed numerical Cost/Slack values from candidate labels.
    - Candidate labels now only show semantic text (e.g., "URGENT", "Wait") or nothing if status is normal ("Ready").
    - Relies entirely on color coding for state visualization.
    """
    locs = disperse_locations_global(locs)
    # --- 1. 配色方案 (BGR) ---
    COLOR_BG = (255, 255, 255)
    
    # 节点基础色
    COLOR_DEPOT = (40, 40, 40)             # Black/Dark Grey (Base)
    COLOR_CURRENT_FILL = (220, 100, 50)    # Royal Blue
    COLOR_UNVISITED = (34, 34, 200)        # Deep Red
    COLOR_VISITED = (220, 220, 220)        # Light Grey
    
    # 状态语义色 (Border Colors) - Aligned with Text Obs legend
    COLOR_VRP_RETURN = (255, 0, 0)         # Blue (Return to Base)
    COLOR_VRP_URGENT = (0, 0, 255)         # Red (Low Slack)
    COLOR_VRP_WAIT = (0, 165, 255)         # Orange (Wait)
    COLOR_VRP_READY = (50, 200, 50)        # Green (Ready)
    
    # 历史路径颜色池 (Muted Pastels)
    ROUTE_COLORS = [
        (180, 180, 180), (150, 150, 220), (150, 220, 150), 
        (220, 180, 150), (200, 150, 200),
    ]

    # 辅助
    COLOR_TEXT = (10, 10, 10)
    COLOR_ZOOM_BOX = (0, 0, 255)
    COLOR_BORDER = (180, 180, 180)

    # 画布初始化
    img_width = img_height * 2 
    combined_canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    left_roi = combined_canvas[:, :img_height] 
    right_roi = combined_canvas[:, img_height:]

    # --- 2. 坐标变换逻辑 ---
    def get_transform(center, span, output_size, padding=40):
        scale = (output_size - 2 * padding) / max(span, 1e-6)
        canvas_center = output_size / 2.0
        def transform_fn(coords):
            coords = np.array(coords)
            centered = coords - center
            scaled = centered * scale
            final = scaled.copy()
            final[..., 0] += canvas_center
            final[..., 1] = canvas_center - final[..., 1] 
            return final.astype(int)
        return transform_fn, (center - span/2, center + span/2)

    # --- 3. 全局视图计算 ---
    g_min, g_max = np.min(locs, axis=0), np.max(locs, axis=0)
    g_center, g_span = (g_min + g_max) / 2.0, np.max(g_max - g_min)
    global_transform, _ = get_transform(g_center, g_span, img_height, padding=60)

    # --- 4. 智能聚焦逻辑 ---
    curr_pos = locs[current_node_idx]
    if top_candidates:
        cand_coords = np.array([[c['x'], c['y']] for c in top_candidates])
        dists = np.linalg.norm(cand_coords - curr_pos, axis=1)
        max_dist = np.max(dists)
        zoom_span = max(max_dist * 2.5, g_span * 0.05)
        zoom_span = min(zoom_span, g_span * 0.5)
    else:
        zoom_span = g_span * 0.2
    zoom_transform, (z_real_min, z_real_max) = get_transform(curr_pos, zoom_span, img_height, padding=40)

    # --- 5. 绘图函数 ---
    def draw_scene(canvas, transform_fn, view_bounds=None, is_zoomed=False):
        pts = transform_fn(locs)
        
        # === Layer 1: Multi-Route History ===
        if len(path_history) > 1:
            routes = []
            current_route_seg = []
            for node_idx in path_history:
                current_route_seg.append(node_idx)
                if node_idx == 0 and len(current_route_seg) > 1:
                    routes.append(current_route_seg)
                    current_route_seg = [0]
            if len(current_route_seg) > 1:
                routes.append(current_route_seg)
            
            for r_idx, route in enumerate(routes):
                r_color = ROUTE_COLORS[r_idx % len(ROUTE_COLORS)]
                hist_pts = pts[route]
                for j in range(len(hist_pts)-1):
                    thickness = 3 if is_zoomed else 2
                    cv2.line(canvas, tuple(hist_pts[j]), tuple(hist_pts[j+1]), r_color, thickness, cv2.LINE_AA)

        # === Layer 2: Base Nodes (Clean, No Numbers) ===
        node_radius = 6 if is_zoomed else 4
        for i in range(len(locs)):
            pt = tuple(pts[i])
            is_candidate = False
            for c in top_candidates:
                if c['id'] == i: is_candidate = True; break
            
            if i == current_node_idx or is_candidate: continue
            
            if i == 0: # Depot
                d_size = node_radius + 2
                cv2.rectangle(canvas, (pt[0]-d_size, pt[1]-d_size), (pt[0]+d_size, pt[1]+d_size), COLOR_DEPOT, -1, cv2.LINE_AA)
            elif visited_mask[i]:
                cv2.circle(canvas, pt, node_radius, COLOR_VISITED, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, pt, node_radius, COLOR_UNVISITED, -1, cv2.LINE_AA)
                # [MODIFIED] Removed numerical TW display in zoom view for cleanliness

        # === Layer 3: Candidates (Semantic Colors & Labels only) ===
        font_scale = 0.5 if is_zoomed else 0.4
        label_thickness = 1
        cand_label_box_pad = 6 if is_zoomed else 4
        
        candidate_list = list(enumerate(top_candidates))
        for rank, cand in reversed(candidate_list):
            cand_idx = cand['id']
            cand_pt = tuple(pts[cand_idx])
            label = chr(65 + rank)
            
            # 3.1 Draw Node Body
            if cand_idx == 0:
                d_size = node_radius + 2
                cv2.rectangle(canvas, (cand_pt[0]-d_size, cand_pt[1]-d_size), 
                              (cand_pt[0]+d_size, cand_pt[1]+d_size), COLOR_DEPOT, -1, cv2.LINE_AA)
            else:
                cv2.circle(canvas, cand_pt, node_radius, COLOR_UNVISITED, -1, cv2.LINE_AA)
            
            # 3.2 Determine Status & Color (Pure Semantics)
            slack = cand.get('slack', 9999)
            wait = cand.get('wait', 0)
            
            semantic_label = "" # Default empty for Ready state
            if cand['is_depot']:
                border_color = COLOR_VRP_RETURN
                border_width = 2
                semantic_label = "End Route"
            elif slack < 300.0:  # URGENT
                border_color = COLOR_VRP_URGENT
                border_width = 2
                semantic_label = "URGENT" # Replaced cost with semantic label
            elif wait > 0:       # WAIT
                border_color = COLOR_VRP_WAIT
                border_width = 2
                semantic_label = "Wait"   # Replaced cost with semantic label
            else:                # READY
                border_color = COLOR_VRP_READY
                border_width = 2
                semantic_label = "Ready"  # Will not be drawn below box
            
            # 3.3 Label Box
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, label_thickness)
            box_tl = (cand_pt[0] - w//2 - cand_label_box_pad, cand_pt[1] - h//2 - cand_label_box_pad)
            box_br = (cand_pt[0] + w//2 + cand_label_box_pad, cand_pt[1] + h//2 + cand_label_box_pad)
            cv2.rectangle(canvas, box_tl, box_br, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.rectangle(canvas, box_tl, box_br, border_color, border_width, cv2.LINE_AA)
            
            # 3.4 Label Text (A, B, C...)
            text_x = cand_pt[0] - w // 2
            text_y = cand_pt[1] + h // 2
            cv2.putText(canvas, label, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, label_thickness, cv2.LINE_AA)
            
            # 3.5 [Zoom View] Show Semantic Label (Only if not just "Ready")
            # [MODIFIED] Only draw text if it adds semantic value beyond the green color.
            if is_zoomed and semantic_label != "Ready":
                (cw, ch), _ = cv2.getTextSize(semantic_label, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
                cx = cand_pt[0] - cw // 2
                cy = box_br[1] + ch + 2
                cv2.putText(canvas, semantic_label, (cx, cy), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, border_color, 1, cv2.LINE_AA)

        # === Layer 4: Current Agent ===
        curr_pt = tuple(pts[current_node_idx])
        curr_size = 10 if is_zoomed else 6
        if current_node_idx == 0:
            cv2.rectangle(canvas, (curr_pt[0]-curr_size-2, curr_pt[1]-curr_size-2), 
                      (curr_pt[0]+curr_size+2, curr_pt[1]+curr_size+2), (255,255,255), -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (curr_pt[0]-curr_size, curr_pt[1]-curr_size), 
                      (curr_pt[0]+curr_size, curr_pt[1]+curr_size), COLOR_DEPOT, -1, cv2.LINE_AA)
        else:
            cv2.rectangle(canvas, (curr_pt[0]-curr_size-2, curr_pt[1]-curr_size-2), 
                      (curr_pt[0]+curr_size+2, curr_pt[1]+curr_size+2), (255,255,255), -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (curr_pt[0]-curr_size, curr_pt[1]-curr_size), 
                      (curr_pt[0]+curr_size, curr_pt[1]+curr_size), COLOR_CURRENT_FILL, -1, cv2.LINE_AA)

    # --- 6. Execute Drawing ---
    draw_scene(left_roi, global_transform, is_zoomed=False)
    if top_candidates:
        # ... (Focus box logic same as before) ...
        box_p1 = global_transform(g_center - g_span * 0.15); box_p2 = global_transform(g_center + g_span * 0.15) # Simplified for brevity, use actual zoom calculation
        # Re-calculate real zoom bounds for drawing box
        z_span_half = zoom_span / 2.0
        z_real_min = curr_pos - z_span_half
        z_real_max = curr_pos + z_span_half
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

    draw_scene(right_roi, zoom_transform, view_bounds=(curr_pos-zoom_span/2, curr_pos+zoom_span/2), is_zoomed=True)
    
    # --- 7. Legend (Semantic Only) ---
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
        def icon_depot(x, y): cv2.rectangle(img, (x-6, y-6), (x+6, y+6), COLOR_DEPOT, -1)
        current_y = draw_item(current_y, "Depot (Base)", icon_depot)
        def icon_return(x, y):
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), (255,255,255), -1)
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), COLOR_VRP_RETURN, 2)
        current_y = draw_item(current_y, "Return to Base", icon_return)
        def icon_urgent(x, y):
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), (255,255,255), -1)
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), COLOR_VRP_URGENT, 2)
        current_y = draw_item(current_y, "Urgent (Low Slack)", icon_urgent)
        def icon_wait(x, y):
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), (255,255,255), -1)
            cv2.rectangle(img, (x-7, y-7), (x+7, y+7), COLOR_VRP_WAIT, 2)
        current_y = draw_item(current_y, "Waiting Needed", icon_wait)
    draw_legend(left_roi)
    
    # Titles & Time (Time is okay, it's global context)
    cv2.rectangle(combined_canvas, (img_height, 0), (img_width-1, img_height-1), COLOR_BORDER, 4)
    cv2.putText(left_roi, "Global Map", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2, cv2.LINE_AA)
    cv2.putText(right_roi, "Egocentric View", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80,80,80), 2, cv2.LINE_AA)
    cv2.putText(right_roi, f"Time: {current_time:.1f}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)

    # --- 8. Output ---
    _, buffer = cv2.imencode('.png', combined_canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    img_rgb_np = cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2RGB)
    if debug_save_path:
        os.makedirs(os.path.dirname(debug_save_path) if os.path.dirname(debug_save_path) else ".", exist_ok=True)
        cv2.imwrite(debug_save_path, combined_canvas)
    return b64_str, img_rgb_np

def render_lrp_image(
    locs, 
    demands,           # [N] Customer demands (0 for depots)
    visited_mask,      # [N] 1=Visited, 0=Unvisited
    open_depots_mask,  # [N] 1=Open, 0=Closed (Only relevant for first num_depots indices)
    current_node_idx, 
    path_history, 
    current_load,      # Scalar: Current load of the vehicle
    vehicle_capacity,  # Scalar: Max capacity of vehicle
    depot_usages,      # [N] Usage of each depot (0 for customers)
    depot_capacities,  # [N] or Scalar
    top_candidates, 
    num_depots,
    img_height=336, 
    debug_save_path=None
):
    """
    Optimized LRP Dual-View Renderer (Global + Egocentric Zoom).
    Features: Multi-route coloring, Semantic Candidate Labels, Smart Zoom.
    """
    # 尝试分散坐标防止重叠 (如果有这个函数的话)
    try:
        locs = disperse_locations_global(locs)
    except: pass
    
    # --- 1. Color Palette (BGR) ---
    COLOR_BG = (255, 255, 255)
    
    # Node Colors
    COLOR_DEPOT_OPEN   = (50, 50, 200)      # Red (Active)
    COLOR_DEPOT_CLOSED = (180, 180, 180)    # Light Gray (Inactive)
    COLOR_CUST_UNVISIT = (200, 120, 50)     # Blue-ish
    COLOR_CUST_VISITED = (230, 230, 230)    # Very Light Gray
    
    # Highlight Colors
    COLOR_CURRENT_NODE = (50, 200, 50)      # Green (Current Position)
    COLOR_TEXT         = (20, 20, 20)
    COLOR_ZOOM_BOX     = (0, 0, 255)        # Red Box on Global Map
    COLOR_BORDER       = (100, 100, 100)
    
    # Load Bar Colors
    COLOR_BAR_BG       = (200, 200, 200)
    COLOR_BAR_FILL     = (50, 150, 250)     # Orange/Yellow for load

    # --- 2. Canvas Setup ---
    img_width = img_height * 2 
    combined_canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    left_roi = combined_canvas[:, :img_height] 
    right_roi = combined_canvas[:, img_height:]

    # --- 3. Coordinate Transformation Logic ---
    def get_transform(center, span, output_size, padding=40):
        half_span = span / 2.0
        scale = (output_size - 2 * padding) / max(span, 1e-6)
        canvas_center = output_size / 2.0
        
        def transform_fn(coords):
            coords = np.array(coords)
            centered = coords - center
            scaled = centered * scale
            final = scaled.copy()
            final[..., 0] += canvas_center
            final[..., 1] = canvas_center - final[..., 1] # Flip Y for image coords
            return final.astype(int)
        return transform_fn, (center - half_span, center + half_span)

    # Global Transform
    g_min = np.min(locs, axis=0)
    g_max = np.max(locs, axis=0)
    g_center = (g_min + g_max) / 2.0
    g_span = np.max(g_max - g_min)
    global_transform, _ = get_transform(g_center, g_span, img_height, padding=50)

    # Zoom Transform (Smart Focus)
    curr_pos = locs[current_node_idx]
    if top_candidates:
        cand_coords = np.array([[c['x'], c['y']] for c in top_candidates])
        dists = np.linalg.norm(cand_coords - curr_pos, axis=1)
        max_dist = np.max(dists)
        zoom_span = max(max_dist * 2.5, g_span * 0.15) # Ensure strictly local view
        zoom_span = min(zoom_span, g_span * 0.6)       # Don't zoom out too much
    else:
        zoom_span = g_span * 0.25
        
    zoom_transform, (z_real_min, z_real_max) = get_transform(curr_pos, zoom_span, img_height, padding=40)

    # --- 4. Drawing Function ---
    def draw_scene(canvas, transform_fn, is_zoomed=False):
        pts = transform_fn(locs)
        
        # A. Draw Trajectory Lines (Current Route Only - Gray)
        if len(path_history) > 1:
            # Find the start of the current route (last depot visited)
            start_idx = 0
            for i, node_idx in enumerate(path_history):
                if node_idx < num_depots:
                    start_idx = i
            
            # Extract current route segment
            current_route = path_history[start_idx:]
            
            if len(current_route) > 1:
                route_pts = pts[current_route]
                # Use standard gray color
                cv2.polylines(canvas, [route_pts], False, (150, 150, 150), 2, cv2.LINE_AA)

        # B. Draw Nodes (Base Layer)
        base_r = 5 if is_zoomed else 3
        
        for i in range(len(locs)):
            # Skip current node and candidates (drawn later on top)
            is_cand = False
            for c in top_candidates:
                if c['id'] == i: is_cand = True; break
            if i == current_node_idx or is_cand:
                continue

            pt = tuple(pts[i])
            
            if i < num_depots:
                # === DEPOT ===
                color = COLOR_DEPOT_OPEN if open_depots_mask[i] else COLOR_DEPOT_CLOSED
                tl = (pt[0] - base_r, pt[1] - base_r)
                br = (pt[0] + base_r, pt[1] + base_r)
                if open_depots_mask[i]:
                    cv2.rectangle(canvas, tl, br, color, -1, cv2.LINE_AA) # Filled
                else:
                    cv2.rectangle(canvas, tl, br, color, 1, cv2.LINE_AA)  # Hollow
            else:
                # === CUSTOMER ===
                if visited_mask[i]:
                    continue # Hide visited customers
                else:
                    # Dynamic size based on demand
                    dem_r = base_r
                    if demands is not None:
                        ratio = demands[i] / (vehicle_capacity + 1e-6)
                        dem_r = int(base_r + (4 * ratio)) if is_zoomed else base_r
                    cv2.circle(canvas, pt, dem_r, COLOR_CUST_UNVISIT, -1, cv2.LINE_AA)

        # C. Draw Candidates (Highlighted with Labels)
        font_scale = 0.5 if is_zoomed else 0.4
        for rank, cand in reversed(list(enumerate(top_candidates))):
            idx = cand['id']
            pt = tuple(pts[idx])
            label = chr(65 + rank) if rank < 26 else str(rank)
            
            # Determine visual style based on node type
            if idx < num_depots:
                border_col = COLOR_DEPOT_OPEN
                fill_col = COLOR_DEPOT_OPEN
                shape = 'rect'
            else:
                border_col = COLOR_CUST_UNVISIT
                fill_col = COLOR_CUST_UNVISIT
                shape = 'circle'
                
            # Draw Body
            sz = base_r + 2
            if shape == 'rect':
                cv2.rectangle(canvas, (pt[0]-sz, pt[1]-sz), (pt[0]+sz, pt[1]+sz), fill_col, -1)
                cv2.rectangle(canvas, (pt[0]-sz, pt[1]-sz), (pt[0]+sz, pt[1]+sz), (0,0,0), 1)
            else:
                cv2.circle(canvas, pt, sz, fill_col, -1)
                cv2.circle(canvas, pt, sz, (0,0,0), 1)
            
            # Label Box (White background with colored border)
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            pad = 4
            box_tl = (pt[0] - w//2 - pad, pt[1] - h//2 - pad)
            box_br = (pt[0] + w//2 + pad, pt[1] + h//2 + pad)
            
            cv2.rectangle(canvas, box_tl, box_br, (255,255,255), -1)
            cv2.rectangle(canvas, box_tl, box_br, border_col, 1)
            cv2.putText(canvas, label, (pt[0]-w//2, pt[1]+h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, 1, cv2.LINE_AA)

        # D. Draw Current Vehicle (Agent)
        curr_pt = tuple(pts[current_node_idx])
        agent_size = 7 if is_zoomed else 4
        
        # Agent Body
        cv2.rectangle(canvas, (curr_pt[0]-agent_size, curr_pt[1]-agent_size),
                      (curr_pt[0]+agent_size, curr_pt[1]+agent_size), COLOR_CURRENT_NODE, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (curr_pt[0]-agent_size, curr_pt[1]-agent_size),
                      (curr_pt[0]+agent_size, curr_pt[1]+agent_size), (255,255,255), 1, cv2.LINE_AA)
        
        # Load Bar (Only in Zoom View)
        if is_zoomed:
            bar_w = 24
            bar_h = 4
            bar_x = curr_pt[0] - bar_w // 2
            bar_y = curr_pt[1] - agent_size - 8
            
            # Background
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), COLOR_BAR_BG, -1)
            # Fill
            load_ratio = max(0.0, min(1.0, current_load / (vehicle_capacity + 1e-6)))
            fill_w = int(bar_w * load_ratio)
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+fill_w, bar_y+bar_h), COLOR_BAR_FILL, -1)
            # Border
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (50,50,50), 1)

    # --- 5. Render Both Views ---
    # Global View
    draw_scene(left_roi, global_transform, is_zoomed=False)
    
    # Zoom Box visualization
    if top_candidates:
        box_p1 = global_transform(z_real_min)
        box_p2 = global_transform(z_real_max)
        x1, y1 = min(box_p1[0], box_p2[0]), min(box_p1[1], box_p2[1])
        x2, y2 = max(box_p1[0], box_p2[0]), max(box_p1[1], box_p2[1])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_height-1, x2), min(img_height-1, y2)
        
        if x2 > x1 and y2 > y1:
            cv2.rectangle(left_roi, (x1, y1), (x2, y2), COLOR_ZOOM_BOX, 2)
            cv2.line(combined_canvas, (x2, y1), (img_height, 0), COLOR_ZOOM_BOX, 1, cv2.LINE_AA)
            cv2.line(combined_canvas, (x2, y2), (img_height, img_height), COLOR_ZOOM_BOX, 1, cv2.LINE_AA)

    # Local View
    draw_scene(right_roi, zoom_transform, is_zoomed=True)

    # --- 6. Legend & Info ---
    # Titles
    cv2.putText(left_roi, "LRP Global Map", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,80), 2, cv2.LINE_AA)
    cv2.putText(right_roi, "Egocentric View", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80,80,80), 2, cv2.LINE_AA)
    
    # Load Info
    load_pct = (current_load / (vehicle_capacity + 1e-6)) * 100
    load_str = f"Load: {load_pct:.0f}%"
    cv2.putText(right_roi, load_str, (img_height - 110, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    
    # Simple Legend
    leg_x, leg_y = 15, img_height - 15
    def draw_leg(img, txt, col, y):
        cv2.circle(img, (leg_x, y), 4, col, -1)
        cv2.putText(img, txt, (leg_x+12, y+4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60,60,60), 1, cv2.LINE_AA)
        return y - 20
        
    cy = leg_y
    cy = draw_leg(left_roi, "Customer", COLOR_CUST_UNVISIT, cy)
    cy = draw_leg(left_roi, "Depot (Open)", COLOR_DEPOT_OPEN, cy)
    cy = draw_leg(left_roi, "Current Agent", COLOR_CURRENT_NODE, cy)
    
    # Border
    cv2.line(combined_canvas, (img_height, 0), (img_height, img_height), COLOR_BORDER, 2)
    cv2.rectangle(combined_canvas, (0, 0), (img_width-1, img_height-1), COLOR_BORDER, 2)

    # --- 7. Output ---
    success, buffer = cv2.imencode('.png', combined_canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    img_rgb_np = cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2RGB)
    
    if debug_save_path:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, combined_canvas)

    return b64_str, img_rgb_np

def build_obs_lrp(td, env_num, trajectory=None, top_k=24, given_topk_acts=None, image_obs="rgb") -> list:
    obs_list = []
    
    # --- 1. Data Extraction & Pre-processing (更为鲁棒的转换) ---
    def to_np(x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

    locs = to_np(td["locs"])
    current_node = to_np(td["current_node"])
    current_load = to_np(td["current_load"])
    current_depot = to_np(td["current_depot"])
    visited = to_np(td["visited"])
    mask = to_np(td["action_mask"])
    i_step = to_np(td["i"])
    
    # LRP Specifics
    open_depots = to_np(td["open_depots"]) # [Batch, Num_Depots] usually binary
    depot_usage = to_np(td["depot_usage"]) # [Batch, Num_Depots]
    
    # Demands (Critical for decision making)
    # 假设 demand 存在 td 中，如果不存在则全为 0 (兼容性)
    demands = to_np(td.get("demand", torch.zeros_like(td["locs"][..., 0]))) 
    
    # Capacities
    # 处理标量或向量形式的 Capacity
    veh_cap_raw = td.get("vehicle_capacity", torch.tensor(1.0))
    depot_cap_raw = td.get("depot_capacity", torch.tensor(1.0))
    
    veh_cap = to_np(veh_cap_raw)
    depot_cap = to_np(depot_cap_raw)

    # Num depots determination
    if "num_depots" in td.keys():
        num_depots_val = int(td["num_depots"][0].item())
    else:
        num_depots_val = open_depots.shape[1]

    # Initialize topk placeholder if needed
    if "topk_acts" not in td.keys():
        td["topk_acts"] = torch.full((env_num, top_k), -1, dtype=torch.long, device=td.device)
    topk_acts_list = []

    # --- 2. Per-Environment Loop ---
    for idx in range(env_num):
        # Scalar extraction helper
        get_scalar = lambda x: x[idx].item() if hasattr(x[idx], "item") else x[idx]
        
        # State extraction
        curr_depot_idx = int(current_depot[idx])
        curr_locs = locs[idx]     # [N, 2]
        curr_idx = int(current_node[idx])
        curr_visited = visited[idx]
        curr_mask = mask[idx]
        step = get_scalar(i_step)
        
        cur_veh_load = get_scalar(current_load)
        cur_veh_cap = veh_cap[idx] if veh_cap.ndim > 0 else veh_cap
        
        # Depot Info
        cur_depot_usage = depot_usage[idx]
        cur_depot_cap = depot_cap[idx] if depot_cap.ndim > 0 else depot_cap
        cur_open_depots = open_depots[idx]
        
        curr_pos = curr_locs[curr_idx]
        
        # --- 3. Candidate Selection (Cost-Aware Sorting) ---
        candidates = []
        valid_indices = np.where(curr_mask)[0]
        
        path_history = []

        if trajectory:
            for t_step in trajectory:
                val = t_step[idx]
                if hasattr(val, 'item'): 
                    val = val.item()
                    path_history.append(int(val))
                if not path_history or path_history[-1] != curr_idx:
                    path_history.append(curr_idx)

        # === 1. 获取成本系数 (从 td 中提取，假设存在，不存在则设为默认) ===
        # 注意：这里需要确保维度匹配，通常 cost 是标量
        cost_dist_coef = td.get("cost_dist", torch.tensor(1.0)).item() if "cost_dist" in td else 1.0
        cost_depot_open = td.get("cost_depot", torch.tensor(100.0)).item() if "cost_depot" in td else 100.0
        # vehicle_cost 通常用于决定是否启用新车，但在节点选择排序中，如果是从 depot 出发，所有 customer 都会触发它
        # 所以对于 customer 之间的排序，vehicle_cost 影响不大，但对于 depot 之间的排序，depot_cost 影响巨大。
        
        # === 2. 计算基础距离 ===
        diff = curr_locs[valid_indices] - curr_pos
        dists = np.linalg.norm(diff, axis=1)
        
        # === 3. 计算“真实金钱成本”分数 (Greedy Score) ===
        # Score 越低越好。我们将所有成本统一量纲（通常对齐到 Cost）
        scores = dists * cost_dist_coef
        
        # 遍历 valid_indices 进行特殊惩罚加成
        # 为了利用 numpy 的速度，我们先转换成 boolean mask 处理
        is_depot_mask = valid_indices < num_depots_val
        
        # --- 处理 Depot (回程逻辑) ---
        # 找出那些是 Depot 且 Currently Closed 的索引
        # valid_indices[i] 是全局索引
        # open_depots[idx] 是 [Num_Depots] 的 bool 数组
        
        # 这里需要小心处理 numpy 索引
        candidate_is_depot = valid_indices < num_depots_val
        
        # 如果是 Depot，我们需要检查它是否 Open
        # 创建一个惩罚数组
        penalties = np.zeros_like(scores)
        
        for i, node_idx in enumerate(valid_indices):
            if node_idx < num_depots_val: # Is Depot
                # 检查 Depot 状态
                is_open = open_depots[idx][node_idx]
                if not is_open:
                    # 如果去这个 Depot 需要开启它，加上开启成本！
                    # 这是 Greedy 能得分高的关键：它会权衡 (绕路费 vs 开站费)
                    penalties[i] += cost_depot_open
                    
                # [进阶] 检查 Depot 容量 (如果满载了去也没用，虽然 mask 应该处理了)
                # 如果你想让 Greedy 更聪明，可以给剩余容量很少的 Depot 加一点微小的惩罚
                # 但主要矛盾是 Open/Close
            else: # Is Customer
                # Is Customer
                pass 
                # 通常去 Customer 不需要额外 penalty，除非你考虑 Demand 很大导致后续容易死
                # 简单的 Greedy 不需要考虑那么远
        
        final_scores = scores + penalties
        
        # === 4. 排序 ===
        # 根据 Final Score (总成本) 排序，而不是距离
        sorted_arg = np.argsort(final_scores)
        sorted_indices = valid_indices[sorted_arg][:top_k]
        
        for i, node_idx in enumerate(sorted_indices):
            node_idx = int(node_idx)
            is_depot = node_idx < num_depots_val
            dist = np.linalg.norm(curr_locs[node_idx] - curr_pos)
            
            # 原始索引在 sorted_arg 中的位置
            original_idx_in_valid = sorted_arg[i]
            estimated_cost = final_scores[original_idx_in_valid]
            
            cand_info = {
                "id": node_idx,
                "type": "Depot" if is_depot else "Cust",
                "dist": dist,
                "cost": estimated_cost, # 记录计算出的预估成本
                "x": curr_locs[node_idx][0],
                "y": curr_locs[node_idx][1],
            }
            
            # LRP Specific Info
            if is_depot:
                d_cap = depot_cap[idx] if hasattr(depot_cap, 'ndim') and depot_cap.ndim > 0 else depot_cap
                d_cap_val = d_cap[node_idx] if not np.isscalar(d_cap) else d_cap
                d_use = depot_usage[idx][node_idx]
                
                cand_info["rem_cap"] = d_cap_val - d_use
                cand_info["is_open"] = bool(open_depots[idx][node_idx])
                
                # 在 Info 里标记这个 cost 的来源，帮助 Agent 理解
                if not cand_info["is_open"]:
                    cand_info["tag"] = f"CLOSED (+{cost_depot_open:.0f} Cost)"
                else:
                    cand_info["tag"] = "OPEN (Cheap)"
            else:
                cand_info["demand"] = demands[idx][node_idx]
                cand_info["tag"] = ""
            
            candidates.append(cand_info)
            
        # Update TopK Acts
        valid_len = len(sorted_indices)
        padded = np.array(list(sorted_indices) + [-1]*(top_k - valid_len))
        topk_acts_list.append(padded)
        
        # --- 4. Render Image (External Function) ---
        img_b64 = None
        image_save_path = None
        if image_obs == "path":
            import uuid
            uid = uuid.uuid4()
            image_save_dir = f"/root/autodl-tmp/image/lrp/"
            os.makedirs(image_save_dir, exist_ok=True)
            image_save_path = f"{image_save_dir}env{idx}_step{len(path_history):03d}_{uid}.png"

        img_b64, image_rgb_np = render_lrp_image(
            locs=curr_locs,
            demands=None, # Not strictly needed for vis
            visited_mask=curr_visited,
            open_depots_mask=open_depots[idx],
            current_node_idx=curr_idx,
            path_history=path_history,
            current_load=current_load[idx] if hasattr(current_load[idx], 'item') else current_load[idx],
            vehicle_capacity=veh_cap[idx] if hasattr(veh_cap, 'ndim') and veh_cap.ndim > 0 else veh_cap,
            depot_usages=depot_usage[idx],
            depot_capacities=depot_cap[idx] if hasattr(depot_cap, 'ndim') and depot_cap.ndim > 0 else depot_cap,
            top_candidates=candidates,
            num_depots=num_depots_val,
            debug_save_path=image_save_path
        )

        # --- 5. Text Observation Construction (Enhanced) ---
        
        # Helper strings
        node_type_str = "Depot" if curr_idx < num_depots_val else "Customer"
        rem_veh_cap = cur_veh_cap - cur_veh_load
        
        # Construct Candidate String List
        cand_str_list = []
        for rank, cand in enumerate(candidates):
            label = chr(65 + rank) if rank < 26 else f"Opt{rank}"
            
            base_info = f"Option {label} [{cand['type']} {cand['id']}]: Dist: {cand['dist']:.2f}"
            
            if cand['type'] == "Cust":
                # Customer: 显示需求和是否能装下
                fit_str = " (Fits)" if cand['demand'] <= rem_veh_cap else " (!Overload!)"
                extra_info = f" | Dem: {cand['demand']:.2f}{fit_str}"
            else:
                # Depot: 显示状态和剩余容量
                status = "OPEN" if cand['is_open'] else "CLOSED"
                extra_info = f" | {status} | DepotRemCap: {cand['rem_cap']:.2f}"
                
            cand_str_list.append(base_info + extra_info)
            
        cand_section = "\n".join(cand_str_list)
        
        # Global Prompt
        obs_text = (
            f"### Task: Location Routing Problem (LRP)\n"
            f"Step: {step}\n"
            f"Current Location: Node {curr_idx} ({node_type_str})\n"
            f"Vehicle Status: Load {cur_veh_load:.2f} / Cap {cur_veh_cap:.2f} (Rem: {rem_veh_cap:.2f})\n"
            f"Recent Path: {path_history[-5:]}\n\n"
            f"### Candidates (Nearest Valid):\n{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the next node to visit. Consider:\n"
            f"1. Minimize distance.\n"
            f"2. Ensure vehicle has enough capacity for customer demand.\n"
            f"3. Balance depot usage (don't exceed depot capacity).\n"
            f"4. Prefer already OPEN depots to avoid opening costs."
        )
        
        status_str = f"Step:{step}|Current depot:{curr_depot_idx}|Current Node:{curr_idx}|Vehicle Load:{cur_veh_load:.2f}/{cur_veh_cap:.1f}"
        
        # Pack observation
        obs_dict = {
            "text": obs_text,
            "obs": status_str,
            "candidates": cand_section
        }
        
        # Handle Image formats
        if image_obs == "base64":
            obs_dict["image"] = img_b64 # assumes calculated above
        elif image_obs == "path":
            obs_dict["image"] = image_save_path
        else:
            obs_dict["image"] = image_rgb_np

        obs_list.append(obs_dict)

    # Update TensorDict
    if given_topk_acts is None and len(topk_acts_list) > 0:
        try:
             td["topk_acts"] = torch.tensor(np.array(topk_acts_list), device=td.device)
        except: pass
        
    return obs_list