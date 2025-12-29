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

def render_tsp_image(locs, visited_mask, current_node_idx, path_history, top_candidates, 
                          img_size=600, view_radius=0.25, debug_save_path=None):
    """
    生成以当前节点为中心的 TSP 缩放视图 (Egocentric View)。
    
    Args:
        current_node_idx: 当前节点的索引。
        view_radius (float): 视野半径（归一化坐标系下）。
                             例如 0.25 表示显示当前点周围 +/- 0.25 的区域。
                             数值越小，放得越大。
    """
    # 1. 初始化画布
    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    
    # 获取当前中心点的真实坐标
    curr_x, curr_y = locs[current_node_idx]

    # --- [核心] 建立动态坐标映射系统 ---
    # 视图窗口的边界（归一化坐标）
    x_min, x_max = curr_x - view_radius, curr_x + view_radius
    y_min, y_max = curr_y - view_radius, curr_y + view_radius
    # 窗口的宽度（归一化单位）
    view_width = x_max - x_min
    
    # 缩放比例：像素 / 归一化单位
    scale = img_size / view_width
    
    def to_canvas_xy(coords):
        # 将世界坐标映射到画布像素坐标
        # 核心公式：(相对坐标 / 窗口宽度) * 画布尺寸
        px = int((coords[0] - x_min) * scale)
        # 保持 Y 轴方向一致（上是小，下是大），如果需要翻转 Y 轴在这里改
        py = int((coords[1] - y_min) * scale)
        return (px, py)
    # ------------------------------------

    # 2. 绘制历史路径 (Grey Lines)
    # OpenCV 会自动处理画出界外的情况
    if len(path_history) > 1:
        pts = [to_canvas_xy(locs[idx]) for idx in path_history]
        pts_np = np.array(pts, np.int32)
        # 线条稍细一点，因为现在是放大视图
        cv2.polylines(canvas, [pts_np], isClosed=False, color=(180, 180, 180), thickness=2)

    # 3. 标记起点 (Start Node) - 如果在视野内
    if len(path_history) > 0:
        start_pt = to_canvas_xy(locs[path_history[0]])
        # 画一个黑色方块
        cv2.rectangle(canvas, (start_pt[0]-8, start_pt[1]-8), (start_pt[0]+8, start_pt[1]+8), (0, 0, 0), -1)

    # 4. 绘制所有节点 (背景点)
    for i in range(len(locs)):
        # 跳过当前点和候选点，后面重点画
        is_candidate = any(c['id'] == i for c in top_candidates)
        if i == current_node_idx or is_candidate:
            continue
            
        pt = to_canvas_xy(locs[i])
        # 简单的边界检查，优化性能（可选，OpenCV也能处理）
        if -50 < pt[0] < img_size+50 and -50 < pt[1] < img_size+50:
            if visited_mask[i]:
                cv2.circle(canvas, pt, 4, (220, 220, 220), -1) # 已访问：灰
            else:
                cv2.circle(canvas, pt, 6, (80, 80, 255), -1)   # 未访问：红

    # 5. 绘制当前点 (Current Node) - 永远在画布正中心
    # 理论上 curr_pt 应该计算出来大约是 (img_size/2, img_size/2)
    curr_pt = to_canvas_xy((curr_x, curr_y)) 
    # 绘制醒目的蓝色标记
    cv2.circle(canvas, curr_pt, 18, (255, 200, 100, 0.5), 4) # 蓝色光环
    cv2.circle(canvas, curr_pt, 8, (255, 100, 0), -1)    # 蓝色实心
    
    # 6. 绘制 Top-K 候选 (Green Candidates & Labels)
    for rank, cand in enumerate(top_candidates):
        cand_pt = to_canvas_xy(locs[cand['id']])
        label = chr(65 + rank)
        
        # a. 预览连线 (中心 -> 候选)
        cv2.line(canvas, curr_pt, cand_pt, (180, 255, 180), 2)
        # b. 候选点高亮
        cv2.circle(canvas, cand_pt, 10, (0, 180, 0), 3)
        
        # c. 带引线的标签 (防止在局部视图中依然重叠)
        # 向右上方偏移，引线长度适中
        offset_x, offset_y = 0, -0
        # 简单的边界碰撞检测，防止标签飞出屏幕太远
        if cand_pt[0] > img_size - 80: offset_x = -50
        if cand_pt[1] < 80: offset_y = 50
        
        text_pt = (cand_pt[0] + offset_x, cand_pt[1] + offset_y)
        
        # 文字背景框
        font_scale = 0.8
        thickness = 2
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        pad = 5
        cv2.rectangle(canvas, (text_pt[0]-pad, text_pt[1]-th-pad), (text_pt[0]+tw+pad, text_pt[1]+pad), (255,255,255), -1)
        cv2.rectangle(canvas, (text_pt[0]-pad, text_pt[1]-th-pad), (text_pt[0]+tw+pad, text_pt[1]+pad), (0,180,0), 1)
        
        # 文字
        cv2.putText(canvas, label, text_pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 100, 0), thickness)

    # 7. 添加简单的比例尺/视野说明 (可选)
    # cv2.putText(canvas, f"Egocentric View (Radius: {view_radius:.2f})", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100,100,100), 2)

    # 8. 调试保存 & 编码
    if debug_save_path is not None:
        os.makedirs(os.path.dirname(debug_save_path), exist_ok=True)
        cv2.imwrite(debug_save_path, canvas)

    _, buffer = cv2.imencode('.jpg', canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def build_obs_tsp(td, env_num: int, trajectory: list = None, return_topk_options: bool = False, top_k: int = 5) -> list:
    """
    Args:
        trajectory: List[Tensor], shape of list is Steps. 
                    Each Tensor is (Batch_Size,). 
                    Contains node indices visited at each step.
    """
    obs_list = []
    
    # 提取 TensorDict 数据
    locs = _to_numpy(td["locs"])               # (B, N, 2)
    current_node = _to_numpy(td["current_node"]) # (B,)
    visited = ~_to_numpy(td["action_mask"])         # (B, N) mask
    i_step = _to_numpy(td["i"])                # (B,)
    topk_acts = []
    
    for idx in range(env_num):
        # 1. 解析当前环境的基础数据
        curr_locs = locs[idx]
        curr_idx = current_node[idx] # 当前所在的节点 ID
        curr_visited = visited[idx]
        step = i_step[idx].item() if hasattr(i_step[idx], "item") else i_step[idx]
        
        # 2. 解析历史路径 (Trajectory Processing)
        # trajectory 是 [Step0_Tensor, Step1_Tensor, ...]
        # 我们需要把每个 Step Tensor 的第 idx 个元素取出来
        path_history = []
        if trajectory is not None and len(trajectory) > 0:
            # 遍历列表，取出第 idx 个环境在每一步的选择
            # 注意：需确保 trajectory 里的 tensor 在 CPU 上或者用 .item() 取值
            for t_step in trajectory:
                # 兼容 Tensor 和 numpy
                val = t_step[idx]
                if hasattr(val, 'item'): val = val.item()
                path_history.append(val)
        
        # 确保 history 包含当前点 (有些实现 logic 是先 append 再 step，有些是反的)
        # 如果 path_history 最后一个不是 curr_idx，手动补上，方便画连线
        if len(path_history) == 0 or path_history[-1] != curr_idx:
            path_history.append(curr_idx)

        # 3. 计算 Top-K 最近邻 (KNN)
        # 这里的逻辑是：只看未访问的邻居
        curr_pos = curr_locs[curr_idx].reshape(1, 2)
        dists = cdist(curr_pos, curr_locs, metric='euclidean').flatten()
        
        # 将已访问点距离设为无穷大
        dists[curr_visited == 1] = np.inf
        
        # 排序并取 Top-K
        sorted_indices = np.argsort(dists)
        topk_acts.append(sorted_indices[:top_k])
        candidates = []
        
        for cand_idx in sorted_indices:
            dist_val = dists[cand_idx]
            if dist_val == np.inf: break # 后面都是已访问的
            if len(candidates) >= top_k: break
            
            candidates.append({
                "id": cand_idx,
                "dist": dist_val,
                "x": curr_locs[cand_idx][0],
                "y": curr_locs[cand_idx][1]
            })

        # 4. 生成图像
        debug_path = None
        
        # 调试策略：仅在 Env 0 激活保存，且每 5 步保存一次
        if idx == 0:
            # step 变量来自 i_step[idx]
            if step == 1 or step % 5 == 0 or step == (curr_locs.shape[0]-1):
                # 按照 env_step 格式命名
                debug_path = f"./debug_images/tsp/env{idx}_step{step:03d}.jpg"
        
        # 调用优化的绘图函数
        current_view_radius = 1
        img_b64 = render_tsp_image(
            curr_locs, 
            curr_visited, 
            curr_idx, 
            path_history, 
            candidates,
            view_radius=current_view_radius, # 传入缩放半径
            debug_save_path=debug_path
        )

        # 5. 生成文本 Prompt
        cand_str_list = []
        for rank, cand in enumerate(candidates):
            label = chr(65 + rank)
            cand_str_list.append(
                f"Option {label} [Node {cand['id']}]: "
                f"**Distance: {cand['dist']*1000:.0f}**"
            )
        cand_section = "\n".join(cand_str_list)
        
        # 计算剩余节点数
        remaining = curr_locs.shape[0] - np.sum(curr_visited)

        obs = (
            f"### Task: Traveling Salesperson Problem (TSP)\n"
            f"Step: {step}\n"
            f"Status:\n"
            f"- Current Location: Node {curr_idx}\n"
            f"- Unvisited Nodes: {remaining}\n"
            f"- Path History: {path_history}\n\n"
            # f"### Visual Aid:\n"
            # f"Blue dot is YOU. Grey lines are past path. Red dots are unvisited targets. "
            # f"Green lines show the Top {top_k} recommended moves.\n\n"
            f"### Top {top_k} Nearest Neighbors:\n"
            f"{cand_section}\n\n"
            f"### Instruction:\n"
            f"Select the Option Label (A, B...) to visit next. "
            f"Prefer the closest node (smallest Distance) unless it leads to a dead end."
        )
        
        # 返回 (Text, Image) 元组
        obs_list.append(obs)
    td["topk_acts"] = topk_acts
    return obs_list

def build_obs_cvrp(td: TensorDict, env_num: int, trajectory: List[List[int]] = None, return_topk_options: bool = False, top_k: int = 5) -> List[str]:
    batch_size = td.batch_size[0] if td.batch_size else 1
    obs_list: List[str] = []

    for i in range(batch_size):
        # 1. Base Info (with demands and capacity)
        locs_scaled = _get_locs_scaled(td, i)
        
        demands = td.get("demand", None)
        d_np = _to_numpy(demands[i]) if demands is not None else None
        
        cap_tensor = td.get("capacity", td.get("vehicle_capacity", None))
        capacity = float(_to_numpy(cap_tensor)[0]) if cap_tensor is not None else None

        lines = []
        for node_idx, (x, y) in enumerate(locs_scaled.tolist()):
            demand_val = int(d_np[node_idx]) if (d_np is not None and node_idx < len(d_np)) else 0
            lines.append(f"Node {node_idx}, coordinates: [{x}, {y}], demand: {demand_val};")
        cap_str = f" Vehicle capacity: {int(capacity)}." if capacity is not None else ""
        base_info = " ".join(lines) + cap_str + "\n"
        
        # 2. Metadata
        meta_prefix = _get_common_metadata(td, i, trajectory)
        
        # 3. Top-K Options
        topk_str = _get_topk_str(td, i, trajectory, return_topk_options)
        
        obs_str = base_info + meta_prefix + topk_str
        obs_list.append(obs_str)
        
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
