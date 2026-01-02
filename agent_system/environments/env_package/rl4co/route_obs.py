from typing import List, Optional
import os
import cv2
import base64
import torch
import numpy as np
from tensordict.tensordict import TensorDict
from scipy.spatial.distance import cdist

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
    img_height=600, debug_save_path=None
):
    """
    TSP 智能双视图渲染 (Scientific Style).
    
    Layout:
    - Left: Global View (Path History + Overall Distribution)
    - Right: Smart Zoom (Centered on Current Node + Candidates)
    """
    
    # --- 1. 配色方案 (Scientific) ---
    COLOR_BG = (255, 255, 255)
    
    # 节点颜色
    COLOR_CURRENT_FILL = (220, 100, 50)    # Royal Blue (Current Agent)
    COLOR_UNVISITED = (34, 34, 200)        # Deep Red (Targets)
    COLOR_VISITED = (220, 220, 220)        # Light Grey (History)
    COLOR_START = (20, 20, 20)             # Black (Start Node)
    
    # 线条颜色
    COLOR_PATH_HISTORY = (180, 180, 180)   # Grey Lines
    COLOR_CANDIDATE_EDGE = (50, 180, 50)   # Forest Green
    
    # 辅助
    COLOR_TEXT = (10, 10, 10)
    COLOR_ZOOM_BOX = (0, 0, 255)           # Red Viewfinder
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

    # --- 3. 全局视图 (Global View) ---
    g_min = np.min(locs, axis=0)
    g_max = np.max(locs, axis=0)
    g_center = (g_min + g_max) / 2.0
    g_span = np.max(g_max - g_min)
    
    global_transform, _ = get_transform(g_center, g_span, img_height, padding=60)

    # --- 4. 智能聚焦逻辑 (Smart Zoom) ---
    # 逻辑：以当前点为中心，半径覆盖最远的候选点
    curr_pos = locs[current_node_idx]
    
    if top_candidates:
        cand_coords = np.array([[c['x'], c['y']] for c in top_candidates])
        # 计算当前点到所有候选点的距离
        dists = np.linalg.norm(cand_coords - curr_pos, axis=1)
        # 取最大距离作为半径，并增加 25% 的余量
        max_dist = np.max(dists)
        zoom_span = max(max_dist * 2.5, g_span * 0.05) # 至少显示 5% 的区域
        
        # 限制最大缩放 (不要超过全图的 50%)
        zoom_span = min(zoom_span, g_span * 0.5)
    else:
        # 如果没有候选(比如结束了)，默认显示一小块
        zoom_span = g_span * 0.2

    zoom_transform, (z_real_min, z_real_max) = get_transform(curr_pos, zoom_span, img_height, padding=40)

    # --- 5. 绘图函数 ---
    def draw_scene(canvas, transform_fn, view_bounds=None, is_zoomed=False):
        
        def is_visible(pt):
            if view_bounds is None: return True
            x, y = pt
            vmin, vmax = view_bounds
            # 宽松边界
            return (x >= vmin[0]-0.05) and (x <= vmax[0]+0.05) and (y >= vmin[1]-0.05) and (y <= vmax[1]+0.05)

        pts = transform_fn(locs)
        
        # === Layer 1: Path History (Bottom) ===
        if len(path_history) > 1:
            # 只有在 Global 视图或者 Zoom 视图包含相关线段时才画
            # 为了简单高效，Global 画全量，Zoom 画局部
            
            if not is_zoomed:
                # 全局图：画完整路径
                hist_pts = pts[path_history]
                cv2.polylines(canvas, [hist_pts], isClosed=False, color=COLOR_PATH_HISTORY, thickness=2, lineType=cv2.LINE_AA)
            else:
                # 局部图：只画当前点之前的几步，避免满屏乱线
                # 取最近 10 步
                recent_hist = path_history[-10:]
                if len(recent_hist) > 1:
                    hist_pts = pts[recent_hist]
                    # 需要裁切吗？OpenCV polylines 自动处理裁切，直接画即可
                    cv2.polylines(canvas, [hist_pts], isClosed=False, color=COLOR_PATH_HISTORY, thickness=3, lineType=cv2.LINE_AA)

        # === Layer 2: Nodes (Middle) ===
        # 根据缩放调整大小
        node_radius = 6 if is_zoomed else 4
        curr_size = 10 if is_zoomed else 6
        
        for i in range(len(locs)):
            if not is_visible(locs[i]): continue
            
            pt = tuple(pts[i])
            
            if i == current_node_idx:
                # Current Node (Blue Square) - 稍后在 Top Layer 画，防止被遮挡，这里先跳过或画底色
                continue 
            elif visited_mask[i]:
                # Visited (Grey)
                cv2.circle(canvas, pt, node_radius, COLOR_VISITED, -1, cv2.LINE_AA)
            else:
                # Unvisited (Red)
                cv2.circle(canvas, pt, node_radius, COLOR_UNVISITED, -1, cv2.LINE_AA)
        
        # Start Node 标记 (如果是起点)
        if len(path_history) > 0:
            start_idx = path_history[0]
            if is_visible(locs[start_idx]) and start_idx != current_node_idx:
                pt = tuple(pts[start_idx])
                cv2.rectangle(canvas, (pt[0]-4, pt[1]-4), (pt[0]+4, pt[1]+4), COLOR_START, -1, cv2.LINE_AA)

        # === Layer 3: Current Node & Candidates (Top) ===
        
        # 3.1 Draw Current Node (Blue Square)
        curr_pt = tuple(pts[current_node_idx])
        # White Halo
        cv2.rectangle(canvas, (curr_pt[0]-curr_size-2, curr_pt[1]-curr_size-2), 
                      (curr_pt[0]+curr_size+2, curr_pt[1]+curr_size+2), (255,255,255), -1, cv2.LINE_AA)
        # Blue Fill
        cv2.rectangle(canvas, (curr_pt[0]-curr_size, curr_pt[1]-curr_size), 
                      (curr_pt[0]+curr_size, curr_pt[1]+curr_size), COLOR_CURRENT_FILL, -1, cv2.LINE_AA)

        # 3.2 Draw Candidates (Green Lines & Labels)
        cand_line_width = 3 if is_zoomed else 2
        cand_label_box = 14 if is_zoomed else 10
        font_scale = 0.7 if is_zoomed else 0.5
        
        # 反向遍历，Rank A 在最上层
        candidate_list = list(enumerate(top_candidates))
        for rank, cand in reversed(candidate_list):
            cand_idx = cand['id']
            cand_pt = tuple(pts[cand_idx])
            
            # 画绿线 (Current -> Candidate)
            cv2.line(canvas, curr_pt, cand_pt, COLOR_CANDIDATE_EDGE, cand_line_width, cv2.LINE_AA)
            
            # 画标签 (在目标点旁边，稍微偏移一点以免遮挡点本身)
            # 对于 TSP，把标签直接盖在目标点上是最清晰的（因为我们知道那里有个点）
            # 或者像 STP 一样画在连线中点？
            # TSP 中点通常很密，画在目标点上更好。
            
            label_center = cand_pt
            label = get_label(rank)
            
            # 白底黑框
            cv2.rectangle(canvas, (label_center[0]-cand_label_box, label_center[1]-cand_label_box), 
                          (label_center[0]+cand_label_box, label_center[1]+cand_label_box), (255,255,255), -1, cv2.LINE_AA)
            cv2.rectangle(canvas, (label_center[0]-cand_label_box, label_center[1]-cand_label_box), 
                          (label_center[0]+cand_label_box, label_center[1]+cand_label_box), (20,20,20), 1, cv2.LINE_AA)
            
            # 文字居中
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)
            text_x = label_center[0] - w // 2
            text_y = label_center[1] + h // 2
            cv2.putText(canvas, label, (text_x, text_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, 2, cv2.LINE_AA)

    # --- 6. 执行绘制 ---
    draw_scene(left_roi, global_transform, is_zoomed=False)
    
    # 绘制红框 (Focus Area)
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
    
    # --- 7. 图例 (Legend) ---
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
        
        # Current
        def icon_curr(x, y):
            cv2.rectangle(img, (x-6, y-6), (x+6, y+6), COLOR_CURRENT_FILL, -1, cv2.LINE_AA)
        current_y = draw_item(current_y, "Current Location", icon_curr)
        
        # Unvisited
        current_y = draw_item(current_y, "Unvisited Node", lambda x, y: cv2.circle(img, (x, y), 5, COLOR_UNVISITED, -1, cv2.LINE_AA))
        # Visited
        current_y = draw_item(current_y, "Visited Node", lambda x, y: cv2.circle(img, (x, y), 4, COLOR_VISITED, -1, cv2.LINE_AA))
        # Candidate
        def icon_cand(x, y):
            cv2.line(img, (x-10, y), (x+10, y), COLOR_CANDIDATE_EDGE, 2, cv2.LINE_AA)
            cv2.rectangle(img, (x-3, y-3), (x+3, y+3), (255,255,255), -1)
            cv2.rectangle(img, (x-3, y-3), (x+3, y+3), (20,20,20), 1)
        current_y = draw_item(current_y, "Next Candidate", icon_cand)
        # Focus
        current_y = draw_item(current_y, "Smart Zoom Area", lambda x, y: cv2.rectangle(img, (x-8, y-6), (x+8, y+6), COLOR_ZOOM_BOX, 2, cv2.LINE_AA))

    draw_legend(left_roi)
    
    # UI Border & Title
    cv2.rectangle(combined_canvas, (img_height, 0), (img_width-1, img_height-1), COLOR_BORDER, 4)
    cv2.putText(left_roi, "Global Tour (TSP)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2, cv2.LINE_AA)
    cv2.putText(right_roi, "Egocentric View", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50,50,50), 2, cv2.LINE_AA)

    # --- 8. 输出 ---
    _, buffer = cv2.imencode('.png', combined_canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')

    if debug_save_path is not None:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, combined_canvas)
    
    return b64_str


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
    image_obs: bool = False,
    given_topk_acts = None
) -> list:
    """
    TSP Observation 构建函数 (Density-Based Cluster Aware).
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
        
        # 轨迹处理
        path_history = []
        if trajectory is not None and len(trajectory) > 0:
            for t_step in trajectory:
                val = t_step[idx]
                if hasattr(val, 'item'): val = val.item()
                path_history.append(int(val))
        if len(path_history) == 0 or path_history[-1] != curr_idx:
            path_history.append(curr_idx)

        # 候选生成
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
                    "x": curr_locs[cand_id][0],
                    "y": curr_locs[cand_id][1],
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
            valid_len = len(sorted_indices)
            padded = np.array(sorted_indices + [-1]*(top_k - valid_len))
            topk_acts_list.append(padded)
            
            for uid in sorted_indices:
                candidates.append({
                    "id": int(uid),
                    "dist": np.linalg.norm(curr_locs[uid] - curr_pos),
                    "strategy": strategies.get(uid, "knn"),
                    "x": curr_locs[uid][0],
                    "y": curr_locs[uid][1],
                })

        # 4. 可视化
        img_b64 = None
        debug_path = None
        if given_topk_acts is not None:
             debug_path = f"./debug_images/tsp/env{idx}_step{step:03d}.png"
             
        
        if image_obs or debug_path:
            img_b64 = render_tsp_smart_dual_view(
                locs=curr_locs, 
                visited_mask=(curr_visited==1), 
                current_node_idx=curr_idx, 
                path_history=path_history, 
                top_candidates=candidates, 
                debug_save_path=debug_path
            )

        # 5. Prompt 生成
        cand_str_list = []
        for rank, cand in enumerate(candidates):
            label = chr(65 + rank)
            dist_disp = f"{cand['dist']*100:.2f}"
            
            strat_mark = ""
            if cand.get('strategy') == 'bridge':
                # 明确标记为新簇入口
                strat_mark = " **[New Cluster Entry]**" 
            
            cand_str_list.append(
                f"Option {label} [Node {cand['id']}]: "
                f"Dist: {dist_disp}{strat_mark}"
            )
        cand_section = "\n".join(cand_str_list)
        
        remaining = curr_locs.shape[0] - np.sum(curr_visited)

        obs_text = (
            f"### Task: Traveling Salesperson Problem (TSP)\n"
            f"Step: {step}\n"
            f"Status: Current Node {curr_idx}, Unvisited {remaining}\n"
            f"History: {path_history[-10:]}\n\n"
            f"### Candidate Options (Cluster-Aware Strategy):\n"
            f"Candidates are selected based on distance and cluster analysis:\n"
            f"- Standard: Closest neighbors (cleaning up current local area).\n"
            f"- **[New Cluster Entry]**: The nearest entry point to a SEPARATE, distinct group of nodes.\n"
            f"\n{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Option Label (A, B...) to visit next. \n"
            f"Strategy Logic:\n"
            f"1. **Clear Local**: Prioritize close neighbors to finish the current cluster.\n"
            f"2. **Switch Cluster**: If the current cluster is finished (no close neighbors left), select a **[New Cluster Entry]** to jump to the next group."
        )
        
        if image_obs and img_b64:
            obs_list.append({"text": obs_text, "image": img_b64})
        else:
            obs_list.append(obs_text)

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
    img_size=800, 
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
        hist_pts = pts[path_history]
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
                cv2.circle(canvas, pt, node_radius, COLOR_VISITED, -1, cv2.LINE_AA)
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
        cv2.line(canvas, curr_pt, cand_pt, COLOR_CANDIDATE, cand_line_width, cv2.LINE_AA)
        
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
    bar_w = int(img_size * 0.6)
    bar_h = 20
    bar_x = (img_size - bar_w) // 2
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
    b64_str = base64.b64encode(buffer).decode('utf-8')

    if debug_save_path is not None:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, canvas)
    
    return b64_str
def build_obs_cvrp(
    td, 
    env_num: int, 
    trajectory: list = None, 
    top_k: int = 24,  # 默认扩大候选集，方便容纳远点
    given_topk_acts = None,
    image_obs: bool = False
) -> list:
    """
    CVRP Observation 构建函数 (Hybrid Strategy Version).
    策略：混合候选集生成
    - Part A: KNN (Feasible Nearest Neighbors) -> 负责局部贪婪
    - Part B: Furthest (Feasible Furthest Nodes) -> 负责全局规划 (消除孤岛)
    """
    obs_list = []
    
    # --- 数据转换 ---
    locs = _to_numpy(td["locs"])                 
    demands = _to_numpy(td["demand"])            
    current_node = _to_numpy(td["current_node"]) 
    used_capacity = _to_numpy(td["used_capacity"]) 
    vehicle_capacity = _to_numpy(td["vehicle_capacity"]) 
    
    if "action_mask" in td.keys():
        visited = _to_numpy(td["action_mask"])   
    else:
        visited = np.zeros((env_num, locs.shape[1]))

    if given_topk_acts is not None:
        given_topk_acts = _to_numpy(given_topk_acts)

    # 初始化 topk_acts 容器
    topk_acts_list = []

    # --- 策略参数配置 ---
    # 设定远点的数量。例如 top_k=24 时，4个是远点，20个是KNN。
    # 如果 top_k 很小（如5），则至少保留1个远点。
    if top_k >= 10:
        num_far = 4 
    elif top_k >= 5:
        num_far = 1
    else:
        num_far = 0 # 极少情况
    
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
        
        # 轨迹处理
        path_history = []
        if trajectory is not None and len(trajectory) > 0:
            for t_step in trajectory:
                val = t_step[idx]
                if hasattr(val, 'item'): val = val.item()
                path_history.append(int(val))
        if len(path_history) == 0 or path_history[-1] != curr_idx:
            path_history.append(curr_idx)

        # 预处理 Demand (对齐维度)
        num_nodes = curr_locs.shape[0]
        full_demands = np.zeros(num_nodes)
        if len(curr_demands) == num_nodes - 1:
            full_demands[1:] = curr_demands
        elif len(curr_demands) == num_nodes:
            full_demands = curr_demands
        else:
            limit = min(len(curr_demands), num_nodes)
            full_demands[:limit] = curr_demands[:limit]

        # --- 2. 候选生成逻辑 ---
        candidates = []
        curr_pos = curr_locs[curr_idx]

        # 【分支 A: 注入模式 (SFT)】
        if given_topk_acts is not None:
            topk_indices = given_topk_acts[idx]
            for cand_idx in topk_indices:
                if cand_idx == -1: continue
                cand_idx = int(cand_idx)
                dist = np.linalg.norm(curr_locs[cand_idx] - curr_pos)
                candidates.append({
                    "id": cand_idx,
                    "dist": dist,
                    "demand": float(full_demands[cand_idx]),
                    "is_depot": (cand_idx == 0),
                    "feasible": (cand_idx == 0) or (full_demands[cand_idx] <= remaining_cap + 1e-5),
                    "strategy": "knn" if dist < 0.5 else "furthest"
                })
            topk_acts_list.append(topk_indices)

        # 【分支 B: 混合策略模式 (Hybrid: KNN + Furthest)】
        else:
            # B1. 计算基础距离
            dists = cdist(curr_pos.reshape(1, 2), curr_locs, metric='euclidean').flatten()
            
            # B2. 构建 Feasible Mask
            # 这里的 Mask 逻辑是：1 代表不可选 (Visited, Self, or Overload)
            mask = (curr_visited == 1) 
            overload = full_demands > (remaining_cap + 1e-5)
            mask = mask | overload
            
            # 特殊处理
            mask[0] = 0        # Depot 总是可选 (Refill)
            mask[curr_idx] = 1 # 自己不可选
            
            # 将 mask 掉的点距离设为 inf
            dists_masked = dists.copy()
            dists_masked[mask == 1] = np.inf
            
            # B3. 选择 KNN (最近邻)
            # 先对所有可行点按距离从小到大排序
            all_sorted_indices = np.argsort(dists_masked)
            
            # 过滤掉 inf 的点 (即真正可行的点)
            feasible_indices = [i for i in all_sorted_indices if dists_masked[i] != np.inf]
            
            # 选取前 num_knn 个作为“近点”
            knn_selection = feasible_indices[:num_knn]
            
            # B4. 选择 Far Points (远点)
            far_selection = []
            if num_far > 0:
                # 剩余的可行点 (排除掉已经被选为 KNN 的)
                remaining_indices = feasible_indices[num_knn:]
                
                if len(remaining_indices) > 0:
                    # 按照距离从大到小排序 (找最远的)
                    # 注意：remaining_indices 已经是按距离从小到大排好序的了
                    # 所以我们直接取最后几个，就是最远的
                    # 倒序切片取最后 num_far 个
                    furthest_candidates = remaining_indices[-num_far:][::-1] # [最远, 第二远...]
                    
                    # 用户需求：“远点可以按照 demand 排序”
                    # 现在的 furthest_candidates 是按距离排的。
                    # 如果想在“最远的这几个里”优先展示大需求的，可以在这里重排。
                    # 或者，如果你的意思是“在所有远处的点里选需求大的”，逻辑会更复杂。
                    # 这里实现：选出距离最远的 num_far 个，然后按 Demand 降序排列它们。
                    
                    furthest_candidates = sorted(
                        furthest_candidates, 
                        key=lambda x: full_demands[x], 
                        reverse=True
                    )
                    far_selection = furthest_candidates

            # B5. 合并列表
            # 顺序：KNN (按距离) + Far (按Demand/距离)
            final_indices = knn_selection + far_selection
            
            # 填充 padding (如果可行点不足 top_k)
            valid_len = len(final_indices)
            padded_indices = np.array(final_indices + [-1]*(top_k - valid_len))
            topk_acts_list.append(padded_indices)

            # 生成 Candidates 数据结构
            for cand_idx in final_indices:
                cand_idx = int(cand_idx)
                
                # 标记策略来源
                strat = "knn"
                if cand_idx in far_selection:
                    strat = "furthest"
                
                candidates.append({
                    "id": cand_idx,
                    "dist": dists[cand_idx],
                    "demand": float(full_demands[cand_idx]),
                    "is_depot": (cand_idx == 0),
                    "feasible": True, # 既然选进来了肯定 feasible
                    "strategy": strat
                })

        # --- 3. 生成 Prompt ---
        cand_str_list = []
        for rank, cand in enumerate(candidates):
            label = chr(65 + rank) # A, B, C...
            
            node_type = "**DEPOT (Refill)**" if cand['is_depot'] else f"Customer {cand['id']}"
            demand_info = f", Demand: {cand['demand']:.2f}" if not cand['is_depot'] else ""
            
            # 策略高亮
            strat_mark = ""
            if cand.get('strategy') == 'furthest':
                strat_mark = " **[Far & Heavy]**" # 提示这是远距离的大需求点
            
            # 距离显示优化
            dist_display = f"{cand['dist']*100:.1f}"

            cand_str_list.append(
                f"Option {label} [{node_type}]: "
                f"Dist: {dist_display}{demand_info}{strat_mark}"
            )
        cand_section = "\n".join(cand_str_list)
        
        # 统计剩余信息
        unvisited_mask = (curr_visited == 0)
        unvisited_mask[0] = 0 
        unvisited_customers = np.sum(unvisited_mask)
        
        obs_text = (
            f"### Task: Capacitated Vehicle Routing Problem (CVRP)\n"
            f"Step: {len(path_history)}\n\n"
            f"### Status:\n"
            f"- Location: Node {curr_idx}\n"
            f"- Load: {curr_used:.2f} / {curr_cap:.2f} (Left: {remaining_cap:.2f})\n"
            f"- Pending Customers: {unvisited_customers}\n"
            f"- Recent Path: {path_history[-10:]}\n\n"
            f"### Candidate Analysis (Hybrid Search):\n"
            f"I have identified {len(candidates)} feasible destinations.\n"
            f"- Most options are the closest neighbors (KNN).\n"
            f"- Options marked **[Far & Heavy]** are distant nodes with high demand (Strategic targets).\n"
            f"\n{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Option Label (A, B...) to visit next. \n"
            f"Consider visiting distant clusters ([Far & Heavy]) early if you have high capacity, to avoid inefficient long trips later."
        )
        
        # --- 4. 可视化 (Single View) ---
        img_b64 = None
        debug_path = None
        # 仅在 Env 0 调试输出
        if idx == 0 and (len(path_history) == 1 or len(path_history) % 5 == 0):
             # debug_path = f"./debug_images/cvrp/env{idx}_step{len(path_history):03d}.png"
             pass

        if image_obs or debug_path:
             img_b64 = render_cvrp_image(
                locs=curr_locs,
                demands=full_demands, 
                visited_mask=(curr_visited==1),
                current_node_idx=curr_idx,
                path_history=path_history,
                used_capacity=curr_used,
                vehicle_capacity=curr_cap,
                top_candidates=candidates, # 渲染器会把远点也画出来
                debug_save_path=debug_path
            )

        if image_obs and img_b64:
            obs_list.append({"text": obs_text, "image": img_b64})
        else:
            obs_list.append(obs_text)
            
    # 更新 TD
    if given_topk_acts is None and len(topk_acts_list) > 0:
        # 转换为 Tensor，注意处理非等长情况（虽上面已 padding）
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
