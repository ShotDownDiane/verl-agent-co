import ray
import gymnasium as gym
import torch
import numpy as np
import os
import cv2
import base64
import math
from typing import Any, Dict, List, Optional, Tuple
from tensordict.tensordict import TensorDict

from rl4co.envs.routing.tsp.env import TSPEnv
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.op.env import OPEnv
from rl4co.envs.routing.tdtsp.env import TDTSPMatrixEnv
from rl4co.envs.routing.tdtsp.env_tw import TDTSPTWEnv, TDTSPTWGenerator
from rl4co.envs.routing.tdvrp.env import TDVRPEnv
from rl4co.envs.routing.lrp.env import LRPEnv

from base_env import BaseCOWorker, BaseCOEnvs
from route_obs import (
    apply_angular_masking, 
    get_cluster_entry_points, 
    _to_numpy, 
    _get_common_metadata, 
    _get_locs_scaled,
    build_obs_tsp,
    build_obs_cvrp,
    build_obs_op
)

# =============================================================================
# TDTSP Specific Observation and Rendering Logic
# =============================================================================

def build_obs_tdtsp(
    td: TensorDict, 
    env_num: int, 
    trajectory: list = None, 
    top_k: int = 24, 
    image_obs: str = "path",
    given_topk_acts = None
) -> list:
    """Observation builder for TDTSPMatrixEnv with time-dependent awareness."""
    
    obs_list = []
    batch_size = td.batch_size[0]
    
    # Extract common data
    locs = td["locs"] # [B, N, 2]
    current_node = td["current_node"] # [B]
    current_time = td["current_time"] # [B]
    visited = td.get("visited", torch.zeros_like(td["action_mask"])) # [B, N]
    action_mask = td["action_mask"] # [B, N]
    
    # TDTSP specific
    matrix = td["travel_time_matrix"] # [B, N, N, T] or [N, N, T]
    duration = td["time_step_duration"] # [B] or scalar
    
    if top_k <= 0:
        top_k = 10 # Default for VLM if not specified
    
    for i in range(env_num):
        # 1. Basic Metadata
        meta_str = _get_common_metadata(td, i, trajectory)
        time_val = float(current_time[i])
        meta_str += f" Current Time: {time_val:.1f}s;"
        
        # 2. Candidate Selection (Time-Aware)
        curr_idx = int(current_node[i])
        curr_pos = _to_numpy(locs[i, curr_idx])
        
        # Get unvisited indices
        unvisited_mask = action_mask[i]
        unvisited_indices = torch.where(unvisited_mask)[0]
        unvisited_locs = _to_numpy(locs[i][unvisited_mask])
        
        if len(unvisited_indices) == 0:
            obs_list.append(meta_str + " All nodes visited. Return to start.")
            continue

        # Calculate dynamic travel times for all unvisited nodes
        # s = int(current_time // duration)
        if duration.dim() > 0:
            curr_duration = float(duration[i])
        else:
            curr_duration = float(duration)
            
        time_step = int(time_val // curr_duration)
        max_s = matrix.shape[-1] - 1
        s = min(time_step, max_s)
        
        if matrix.dim() == 4:
            tt_slice = matrix[i, curr_idx, :, s]
        else:
            tt_slice = matrix[curr_idx, :, s]
            
        # Get TT for unvisited
        unvisited_tt = tt_slice[unvisited_indices].cpu().numpy()
        
        # DEBUG
        # print(f"DEBUG: unvisited_indices: {unvisited_indices}")
        # print(f"DEBUG: unvisited_tt: {unvisited_tt}")
        
        # Combine distance and time for candidate ranking
        # We'll use TT as the primary ranking metric for TDTSP
        candidates = []
        for idx_in_unvisited, real_idx in enumerate(unvisited_indices):
            real_idx = int(real_idx)
            pos = _to_numpy(locs[i, real_idx])
            tt = float(unvisited_tt[idx_in_unvisited])
            candidates.append({
                'id': real_idx,
                'x': float(pos[0]),
                'y': float(pos[1]),
                'tt': tt,
                'eta': time_val + tt
            })
            
        # Rank by Travel Time
        candidates.sort(key=lambda x: x['tt'])
        
        # Apply Top-K and Masking
        # Use cluster-aware strategy similar to TSP if needed, but let's keep it TT-focused
        top_candidates = candidates[:top_k]
        
        # 3. Textual Description of Candidates
        cand_str = "\nTop candidates (Time-Dependent):\n"
        valid_opts = []
        for rank, cand in enumerate(top_candidates):
            label = chr(65 + rank) if rank < 26 else f"Opt{rank}"
            valid_opts.append(f"Option {label}. Node {cand['id']} (Travel Time: {cand['tt']:.1f}s, ETA: {cand['eta']:.1f}s)")
        
        cand_str += "; ".join(valid_opts)
        
        # 4. Image Rendering
        image_save_path = None
        step_idx = len(trajectory) if trajectory else 0
        if image_obs == "path":
            image_save_path = f"debug_tdtsp_step{step_idx}_{uuid_name()}.png"
            
        # Extract path history from trajectory
        path_history = []
        if trajectory:
            for step_acts in trajectory:
                if i < len(step_acts):
                    path_history.append(int(step_acts[i]))

        img_b64, _ = render_tdtsp_smart_dual_view(
            locs=_to_numpy(locs[i]),
            visited_mask=_to_numpy(visited[i]),
            current_node_idx=curr_idx,
            path_history=path_history,
            top_candidates=top_candidates,
            current_time=time_val,
            debug_save_path=image_save_path
        )
        
        # Combine all
        full_obs = f"{meta_str}{cand_str}\n[IMAGE] {img_b64 if image_obs == 'base64' else image_save_path}"
        obs_list.append(full_obs)
        
    return obs_list

def uuid_name():
    import uuid
    return str(uuid.uuid4())[:8]

def render_tdtsp_smart_dual_view(
    locs, visited_mask, current_node_idx, path_history, top_candidates, 
    current_time, img_height=336, debug_save_path=None
):
    """Modified rendering for TDTSP with time-cost visualization."""
    # Reuse base logic from TSP rendering but with TDTSP specific overlays
    
    # 配色方案 (Aligned with route_obs.py)
    COLOR_BG = (255, 255, 255)
    COLOR_CURRENT_FILL = (220, 100, 50)    # Royal Blue
    COLOR_START_FILL = (50, 200, 50)
    COLOR_UNVISITED = (34, 34, 200)        # Deep Red
    COLOR_VISITED = (200, 200, 200)        # Light Grey
    COLOR_START = (20, 20, 20)             # Black
    
    # 辅助
    COLOR_TEXT = (10, 10, 10)
    COLOR_ZOOM_BOX = (0, 0, 255)
    COLOR_BORDER = (180, 180, 180)

    img_width = img_height * 2 
    combined_canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    left_roi = combined_canvas[:, :img_height] 
    right_roi = combined_canvas[:, img_height:]

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

    g_min, g_max = np.min(locs, axis=0), np.max(locs, axis=0)
    g_center, g_span = (g_min + g_max) / 2.0, np.max(g_max - g_min)
    global_transform, _ = get_transform(g_center, g_span, img_height, padding=60)

    curr_pos = locs[current_node_idx]
    zoom_span = g_span * 0.3 # Fixed zoom for simplicity
    zoom_transform, (z_real_min, z_real_max) = get_transform(curr_pos, zoom_span, img_height, padding=40)

    def draw_scene(canvas, transform_fn, is_zoomed=False):
        pts = transform_fn(locs)
        
        # 1. Path History
        if len(path_history) > 1:
            hist_pts = pts[path_history]
            for j in range(len(hist_pts)-1):
                cv2.line(canvas, tuple(hist_pts[j]), tuple(hist_pts[j+1]), (200, 200, 200), 2, cv2.LINE_AA)

        # 2. Nodes
        node_radius = 6 if is_zoomed else 4
        for idx in range(len(locs)):
            pt = tuple(pts[idx])
            if idx == current_node_idx:
                cv2.rectangle(canvas, (pt[0]-8, pt[1]-8), (pt[0]+8, pt[1]+8), COLOR_CURRENT_FILL, -1)
            elif visited_mask[idx]:
                cv2.circle(canvas, pt, node_radius, COLOR_VISITED, -1)
            else:
                cv2.circle(canvas, pt, node_radius, COLOR_UNVISITED, -1)

        # 3. TDTSP Candidates with Time Cost Colors
        # Redder = More Travel Time
        if top_candidates:
            max_tt = max(c['tt'] for c in top_candidates) if top_candidates else 1.0
            for rank, cand in enumerate(top_candidates):
                c_idx = cand['id']
                c_pt = tuple(pts[c_idx])
                
                # Color based on TT ratio
                ratio = cand['tt'] / (max_tt + 1e-6)
                # Green (low TT) to Red (high TT)
                color = (0, int(255 * (1-ratio)), int(255 * ratio))
                
                # Draw Box
                label = chr(65 + rank) if rank < 26 else "!"
                cv2.rectangle(canvas, (c_pt[0]-10, c_pt[1]-10), (c_pt[0]+10, c_pt[1]+10), (255,255,255), -1)
                cv2.rectangle(canvas, (c_pt[0]-10, c_pt[1]-10), (c_pt[0]+10, c_pt[1]+10), color, 2)
                cv2.putText(canvas, label, (c_pt[0]-5, c_pt[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                
                if is_zoomed:
                    cv2.putText(canvas, f"{cand['tt']:.0f}s", (c_pt[0]-15, c_pt[1]+22), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    draw_scene(left_roi, global_transform)
    draw_scene(right_roi, zoom_transform, is_zoomed=True)
    
    # Add Time Overlay
    cv2.putText(left_roi, f"Time: {current_time:.1f}s", (20, img_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    _, buffer = cv2.imencode('.png', combined_canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    img_rgb_np = cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2RGB)
    
    if debug_save_path:
        os.makedirs(os.path.dirname(debug_save_path) if os.path.dirname(debug_save_path) else ".", exist_ok=True)
        cv2.imwrite(debug_save_path, combined_canvas)
        
    return b64_str, img_rgb_np

def build_obs_tdvrp(
    td: TensorDict, 
    env_num: int, 
    trajectory: list = None, 
    top_k: int = 24, 
    image_obs: str = "path",
    given_topk_acts = None
) -> list:
    """Observation builder for TDVRPEnv with time-window and cost awareness."""
    
    obs_list = []
    
    # Extract data
    locs = td["locs"] # [B, N, 2]
    current_node = td["current_node"] # [B]
    current_time = td["current_time"] # [B]
    visited = td.get("visited", torch.zeros_like(td["action_mask"])) # [B, N]
    action_mask = td["action_mask"] # [B, N]
    time_windows = td["time_windows"] # [B, N, 2]
    
    # TDVRP specific
    matrix = td["travel_time_matrix"] # [B, N, N, T]
    duration = td["time_step_duration"] # [B]
    service_time = 180.0 # Standard as requested
    
    FIXED_COST = 200.0
    PER_HOUR_COST = 20.0
    
    if top_k <= 0:
        top_k = 10 
    
    for i in range(env_num):
        # 1. Basic Metadata
        meta_str = _get_common_metadata(td, i, trajectory)
        time_val = float(current_time[i])
        meta_str += f" Current Time: {time_val:.1f}s;"
        
        # 2. Candidate Selection (Cost and Time Aware)
        curr_idx = int(current_node[i])
        
        # Get unvisited indices
        unvisited_mask = action_mask[i]
        unvisited_indices = torch.where(unvisited_mask)[0]
        
        # In TDVRP, if only depot is available and we are already at depot, we are stuck or done
        # if len(unvisited_indices) == 1 and int(unvisited_indices[0]) == 0 and curr_idx == 0:
        #     obs_list.append(meta_str + " No more reachable customers and already at depot. Terminating.")
        #     continue

        if len(unvisited_indices) == 0:
            obs_list.append(meta_str + " No more reachable customers. Return to depot.")
            continue

        # Calculate dynamic travel times
        if duration.dim() > 0:
            curr_duration = float(duration[i])
        else:
            curr_duration = float(duration)
            
        time_step = int(time_val // curr_duration)
        max_s = matrix.shape[-1] - 1
        s = min(time_step, max_s)
        
        if matrix.dim() == 4:
            tt_slice = matrix[i, curr_idx, :, s]
        else:
            tt_slice = matrix[curr_idx, :, s]
            
        unvisited_tt = tt_slice[unvisited_indices].cpu().numpy()
        
        # Candidates with TW and Cost info
        candidates = []
        for idx_in_unvisited, real_idx in enumerate(unvisited_indices):
            real_idx = int(real_idx)
            pos = _to_numpy(locs[i, real_idx])
            tt = float(unvisited_tt[idx_in_unvisited])
            tw = _to_numpy(time_windows[i, real_idx])
            
            # Arrival and Departure
            eta = time_val + tt
            is_late = eta > tw[1]
            wait_time = max(0, tw[0] - eta)
            ready_time = max(eta, tw[0])
            # Service time only for customers
            departure_time = ready_time + (service_time if real_idx > 0 else 0.0)
            
            # Cost calculation
            is_new_trip = (curr_idx == 0) and (real_idx > 0)
            fixed_cost = FIXED_COST if is_new_trip else 0.0
            labor_cost = (departure_time - time_val) / 3600.0 * PER_HOUR_COST
            total_step_cost = fixed_cost + labor_cost
            
            candidates.append({
                'id': real_idx,
                'x': float(pos[0]),
                'y': float(pos[1]),
                'tt': tt,
                'eta': eta,
                'tw_start': float(tw[0]),
                'tw_end': float(tw[1]),
                'is_late': is_late,
                'wait_time': wait_time,
                'cost': total_step_cost,
                'is_depot': real_idx == 0
            })
            
        # Rank by: Not Late > Lower Cost > Travel Time
        candidates.sort(key=lambda x: (x['is_late'], x['cost'], x['tt']))
        
        top_candidates = candidates[:top_k]
        
        # 3. Textual Description
        cand_str = "\nTop candidates (TDVRP - Cost & TW Aware):\n"
        valid_opts = []
        for rank, cand in enumerate(top_candidates):
            label = chr(65 + rank) if rank < 26 else f"Opt{rank}"
            tw_info = f"TW: [{cand['tw_start']:.0f}, {cand['tw_end']:.0f}]"
            status = "LATE" if cand['is_late'] else ("WAITING" if cand['wait_time'] > 0 else "OK")
            type_str = "DEPOT" if cand['is_depot'] else "Customer"
            valid_opts.append(f"Option {label}. Node {cand['id']} ({type_str}, {tw_info}, ETA: {cand['eta']:.1f}s, Cost: ${cand['cost']:.2f}, Status: {status})")
        
        cand_str += "; ".join(valid_opts)
        
        # 4. Image Rendering
        image_save_path = None
        step_idx = len(trajectory) if trajectory else 0
        if image_obs == "path":
            image_save_path = f"debug_tdvrp_step{step_idx}_{uuid_name()}.png"
            
        path_history = []
        if trajectory:
            for step_acts in trajectory:
                if i < len(step_acts):
                    path_history.append(int(step_acts[i]))

        img_b64, _ = render_tdvrp_smart_dual_view(
            locs=_to_numpy(locs[i]),
            visited_mask=_to_numpy(visited[i]),
            current_node_idx=curr_idx,
            path_history=path_history,
            top_candidates=top_candidates,
            current_time=time_val,
            time_windows=_to_numpy(time_windows[i]),
            debug_save_path=image_save_path
        )
        
        full_obs = f"{meta_str}{cand_str}\n[IMAGE] {img_b64 if image_obs == 'base64' else image_save_path}"
        obs_list.append(full_obs)
        
    return obs_list

def render_tdvrp_smart_dual_view(
    locs, visited_mask, current_node_idx, path_history, top_candidates, 
    current_time, time_windows, img_height=336, debug_save_path=None
):
    """Rendering for TDVRP with multiple route visualization."""
    COLOR_BG = (255, 255, 255)
    COLOR_CURRENT_FILL = (220, 100, 50)
    COLOR_UNVISITED = (34, 34, 200)
    COLOR_VISITED = (200, 200, 200)
    COLOR_DEPOT = (50, 200, 50) # Green
    
    img_width = img_height * 2 
    combined_canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    left_roi = combined_canvas[:, :img_height] 
    right_roi = combined_canvas[:, img_height:]

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

    g_min, g_max = np.min(locs, axis=0), np.max(locs, axis=0)
    g_center, g_span = (g_min + g_max) / 2.0, np.max(g_max - g_min)
    global_transform, _ = get_transform(g_center, g_span, img_height, padding=60)

    curr_pos = locs[current_node_idx]
    zoom_span = g_span * 0.3
    zoom_transform, _ = get_transform(curr_pos, zoom_span, img_height, padding=40)

    def draw_scene(canvas, transform_fn, is_zoomed=False):
        pts = transform_fn(locs)
        
        # 1. Path History (Identify separate routes)
        if len(path_history) > 1:
            routes = []
            current_route = []
            for node in path_history:
                current_route.append(node)
                if node == 0 and len(current_route) > 1:
                    routes.append(current_route)
                    current_route = [0]
            if len(current_route) > 1:
                routes.append(current_route)
            
            # Use different colors for different routes
            colors = [
                (200, 100, 100), (100, 200, 100), (100, 100, 200),
                (200, 200, 100), (200, 100, 200), (100, 200, 200)
            ]
            for r_idx, route in enumerate(routes):
                r_color = colors[r_idx % len(colors)]
                hist_pts = pts[route]
                for j in range(len(hist_pts)-1):
                    cv2.line(canvas, tuple(hist_pts[j]), tuple(hist_pts[j+1]), r_color, 2, cv2.LINE_AA)

        # 2. Nodes
        node_radius = 6 if is_zoomed else 4
        for idx in range(len(locs)):
            pt = tuple(pts[idx])
            if idx == 0:
                cv2.rectangle(canvas, (pt[0]-10, pt[1]-10), (pt[0]+10, pt[1]+10), COLOR_DEPOT, -1)
            elif idx == current_node_idx:
                cv2.rectangle(canvas, (pt[0]-8, pt[1]-8), (pt[0]+8, pt[1]+8), COLOR_CURRENT_FILL, -1)
            elif visited_mask[idx]:
                cv2.circle(canvas, pt, node_radius, COLOR_VISITED, -1)
            else:
                cv2.circle(canvas, pt, node_radius, COLOR_UNVISITED, -1)
                if is_zoomed:
                    tw = time_windows[idx]
                    cv2.putText(canvas, f"{tw[0]:.0f}-{tw[1]:.0f}", (pt[0]-20, pt[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        # 3. Candidates
        if top_candidates:
            for rank, cand in enumerate(top_candidates):
                c_idx = cand['id']
                c_pt = tuple(pts[c_idx])
                
                if cand['is_late']:
                    color = (0, 0, 255) # Red
                elif cand['wait_time'] > 0:
                    color = (0, 165, 255) # Orange
                else:
                    color = (0, 200, 0) # Green
                
                label = chr(65 + rank) if rank < 26 else "!"
                cv2.rectangle(canvas, (c_pt[0]-10, c_pt[1]-10), (c_pt[0]+10, c_pt[1]+10), (255,255,255), -1)
                cv2.rectangle(canvas, (c_pt[0]-10, c_pt[1]-10), (c_pt[0]+10, c_pt[1]+10), color, 2)
                cv2.putText(canvas, label, (c_pt[0]-5, c_pt[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                
                if is_zoomed:
                    cv2.putText(canvas, f"${cand['cost']:.1f}", (c_pt[0]-15, c_pt[1]+22), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    draw_scene(left_roi, global_transform)
    draw_scene(right_roi, zoom_transform, is_zoomed=True)
    
    cv2.putText(left_roi, f"Time: {current_time:.1f}s", (20, img_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

    _, buffer = cv2.imencode('.png', combined_canvas)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    img_rgb_np = cv2.cvtColor(combined_canvas, cv2.COLOR_BGR2RGB)
    
    if debug_save_path:
        os.makedirs(os.path.dirname(debug_save_path) if os.path.dirname(debug_save_path) else ".", exist_ok=True)
        cv2.imwrite(debug_save_path, combined_canvas)
        
    return b64_str, img_rgb_np

def build_obs_tdtsp_tw(
    td: TensorDict, 
    env_num: int, 
    trajectory: list = None, 
    top_k: int = 24, 
    image_obs: str = "path",
    given_topk_acts = None
) -> list:
    """Observation builder for TDTSPTWEnv with time-window awareness."""
    
    obs_list = []
    
    # Extract data
    locs = td["locs"] # [B, N, 2]
    current_node = td["current_node"] # [B]
    current_time = td["current_time"] # [B]
    visited = td.get("visited", torch.zeros_like(td["action_mask"])) # [B, N]
    action_mask = td["action_mask"] # [B, N]
    time_windows = td["time_windows"] # [B, N, 2]
    
    # TDTSP specific
    matrix = td["travel_time_matrix"] # [B, N, N, T] or [N, N, T]
    duration = td["time_step_duration"] # [B] or scalar
    
    if top_k <= 0:
        top_k = 10 
    
    for i in range(env_num):
        # 1. Basic Metadata
        meta_str = _get_common_metadata(td, i, trajectory)
        time_val = float(current_time[i])
        meta_str += f" Current Time: {time_val:.1f}s;"
        
        # 2. Candidate Selection (Time-Aware)
        curr_idx = int(current_node[i])
        
        # Get unvisited indices
        unvisited_mask = action_mask[i]
        unvisited_indices = torch.where(unvisited_mask)[0]
        
        if len(unvisited_indices) == 0:
            obs_list.append(meta_str + " All nodes visited. Return to start.")
            continue

        # Calculate dynamic travel times
        if duration.dim() > 0:
            curr_duration = float(duration[i])
        else:
            curr_duration = float(duration)
            
        time_step = int(time_val // curr_duration)
        max_s = matrix.shape[-1] - 1
        s = min(time_step, max_s)
        
        if matrix.dim() == 4:
            tt_slice = matrix[i, curr_idx, :, s]
        else:
            tt_slice = matrix[curr_idx, :, s]
            
        unvisited_tt = tt_slice[unvisited_indices].cpu().numpy()
        
        # Candidates with TW info
        candidates = []
        for idx_in_unvisited, real_idx in enumerate(unvisited_indices):
            real_idx = int(real_idx)
            pos = _to_numpy(locs[i, real_idx])
            tt = float(unvisited_tt[idx_in_unvisited])
            tw = _to_numpy(time_windows[i, real_idx])
            
            eta = time_val + tt
            is_late = eta > tw[1]
            wait_time = max(0, tw[0] - eta)
            
            candidates.append({
                'id': real_idx,
                'x': float(pos[0]),
                'y': float(pos[1]),
                'tt': tt,
                'eta': eta,
                'tw_start': float(tw[0]),
                'tw_end': float(tw[1]),
                'is_late': is_late,
                'wait_time': wait_time
            })
            
        # Rank by Travel Time + Early Deadline
        candidates.sort(key=lambda x: (x['is_late'], x['tw_end'], x['tt']))
        
        top_candidates = candidates[:top_k]
        
        # 3. Textual Description
        cand_str = "\nTop candidates (Time-Window Aware):\n"
        valid_opts = []
        for rank, cand in enumerate(top_candidates):
            label = chr(65 + rank) if rank < 26 else f"Opt{rank}"
            tw_info = f"TW: [{cand['tw_start']:.0f}, {cand['tw_end']:.0f}]"
            status = "LATE" if cand['is_late'] else ("WAITING" if cand['wait_time'] > 0 else "OK")
            valid_opts.append(f"{label}. Node {cand['id']} ({tw_info}, ETA: {cand['eta']:.1f}s, Status: {status})")
        
        cand_str += "; ".join(valid_opts)
        
        # 4. Image Rendering
        image_save_path = None
        step_idx = len(trajectory) if trajectory else 0
        if image_obs == "path":
            image_save_path = f"debug_tdtsptw_step{step_idx}_{uuid_name()}.png"
            
        path_history = []
        if trajectory:
            for step_acts in trajectory:
                if i < len(step_acts):
                    path_history.append(int(step_acts[i]))

        img_b64, _ = render_tdtsptw_smart_dual_view(
            locs=_to_numpy(locs[i]),
            visited_mask=_to_numpy(visited[i]),
            current_node_idx=curr_idx,
            path_history=path_history,
            top_candidates=top_candidates,
            current_time=time_val,
            time_windows=_to_numpy(time_windows[i]),
            debug_save_path=image_save_path
        )
        
        full_obs = f"{meta_str}{cand_str}\n[IMAGE] {img_b64 if image_obs == 'base64' else image_save_path}"
        obs_list.append(full_obs)
        
    return obs_list

def render_tdtsptw_smart_dual_view(
    locs, visited_mask, current_node_idx, path_history, top_candidates, 
    current_time, time_windows, img_height=336, debug_save_path=None
):
    """Rendering for TDTSPTW with time-window visualization."""
    # 配色方案 (Aligned with route_obs.py)
    COLOR_BG = (255, 255, 255)
    COLOR_CURRENT_FILL = (220, 100, 50)    # Royal Blue
    COLOR_START_FILL = (50, 200, 50)
    COLOR_UNVISITED = (34, 34, 200)        # Deep Red
    COLOR_VISITED = (200, 200, 200)        # Light Grey
    COLOR_START = (20, 20, 20)             # Black

    COLOR_LATE = (0, 0, 255) # Red in BGR
    COLOR_WAITING = (0, 165, 255) # Orange in BGR
    COLOR_OK = (0, 255, 0) # Green in BGR
    COLOR_TEXT = (10, 10, 10)

    img_width = img_height * 2 
    combined_canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    
    left_roi = combined_canvas[:, :img_height] 
    right_roi = combined_canvas[:, img_height:]

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

    g_min, g_max = np.min(locs, axis=0), np.max(locs, axis=0)
    g_center, g_span = (g_min + g_max) / 2.0, np.max(g_max - g_min)
    global_transform, _ = get_transform(g_center, g_span, img_height, padding=60)

    curr_pos = locs[current_node_idx]
    zoom_span = g_span * 0.3
    zoom_transform, _ = get_transform(curr_pos, zoom_span, img_height, padding=40)

    def draw_scene(canvas, transform_fn, is_zoomed=False):
        pts = transform_fn(locs)
        
        # 1. Path History
        if len(path_history) > 1:
            hist_pts = pts[path_history]
            for j in range(len(hist_pts)-1):
                cv2.line(canvas, tuple(hist_pts[j]), tuple(hist_pts[j+1]), (200, 200, 200), 2, cv2.LINE_AA)

        # 2. Nodes
        node_radius = 6 if is_zoomed else 4
        for idx in range(len(locs)):
            pt = tuple(pts[idx])
            if idx == current_node_idx:
                cv2.rectangle(canvas, (pt[0]-8, pt[1]-8), (pt[0]+8, pt[1]+8), COLOR_CURRENT_FILL, -1)
            elif visited_mask[idx]:
                cv2.circle(canvas, pt, node_radius, COLOR_VISITED, -1)
            else:
                cv2.circle(canvas, pt, node_radius, COLOR_UNVISITED, -1)
                # Show TW on zoomed view
                if is_zoomed:
                    tw = time_windows[idx]
                    cv2.putText(canvas, f"{tw[0]:.0f}-{tw[1]:.0f}", (pt[0]-20, pt[1]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)

        # 3. Candidates with TW Status Colors
        if top_candidates:
            for rank, cand in enumerate(top_candidates):
                c_idx = cand['id']
                c_pt = tuple(pts[c_idx])
                
                if cand['is_late']:
                    color = (0, 0, 255) # Red
                elif cand['wait_time'] > 0:
                    color = (0, 165, 255) # Orange
                else:
                    color = (0, 200, 0) # Green
                
                label = chr(65 + rank) if rank < 26 else "!"
                cv2.rectangle(canvas, (c_pt[0]-10, c_pt[1]-10), (c_pt[0]+10, c_pt[1]+10), (255,255,255), -1)
                cv2.rectangle(canvas, (c_pt[0]-10, c_pt[1]-10), (c_pt[0]+10, c_pt[1]+10), color, 2)
                cv2.putText(canvas, label, (c_pt[0]-5, c_pt[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
                
                if is_zoomed:
                    cv2.putText(canvas, f"ETA:{cand['eta']:.0f}", (c_pt[0]-15, c_pt[1]+22), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

    draw_scene(left_roi, global_transform)
    draw_scene(right_roi, zoom_transform, is_zoomed=True)
    
    cv2.putText(left_roi, f"Time: {current_time:.1f}s", (20, img_height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

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
    LRP Dual-View Renderer (Global + Egocentric Zoom).
    Visualizes Depots (Open/Closed), Customers (Demand sizes), and Vehicle Load.
    """
    
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
    
    # Load Bar Colors
    COLOR_BAR_BG       = (200, 200, 200)
    COLOR_BAR_FILL     = (50, 150, 250)     # Orange/Yellow for load

    # --- 2. Canvas Setup ---
    img_width = img_height * 2 
    combined_canvas = np.ones((img_height, img_width, 3), dtype=np.uint8) * 255
    left_roi = combined_canvas[:, :img_height] 
    right_roi = combined_canvas[:, img_height:]

    # --- 3. Coordinate Transformation Logic ---
    # Normalize coordinates to handle any scale (0-1 or 0-100)
    def get_transform(center, span, output_size, padding=40):
        half_span = span / 2.0
        min_xy = center - half_span
        # Aspect ratio correction is assumed square for simplicity in LRP usually
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
        
        # A. Draw Trajectory Lines
        if len(path_history) > 1:
            hist_pts = pts[path_history]
            # In zoom view, only show recent history to reduce clutter
            if is_zoomed and len(hist_pts) > 10:
                hist_pts = hist_pts[-10:]
            
            # Draw lines
            cv2.polylines(canvas, [hist_pts], False, (200, 200, 200), 2, cv2.LINE_AA)
            
            # Highlight last segment (Current Move)
            if len(hist_pts) >= 2:
                cv2.line(canvas, tuple(hist_pts[-2]), tuple(hist_pts[-1]), 
                         (100, 100, 100), 2, cv2.LINE_AA)

        # B. Draw All Nodes (Base Layer)
        # Determine base size
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
                # Square shape
                tl = (pt[0] - base_r, pt[1] - base_r)
                br = (pt[0] + base_r, pt[1] + base_r)
                
                if open_depots_mask[i]:
                    cv2.rectangle(canvas, tl, br, color, -1, cv2.LINE_AA) # Filled
                else:
                    cv2.rectangle(canvas, tl, br, color, 1, cv2.LINE_AA)  # Hollow
            else:
                # === CUSTOMER ===
                if visited_mask[i]:
                    cv2.circle(canvas, pt, base_r, COLOR_CUST_VISITED, -1, cv2.LINE_AA)
                else:
                    # Dynamic size based on demand relative to capacity?
                    # Simply add small variation: 
                    # If demands are available, scale radius slightly.
                    dem_r = base_r
                    if demands is not None:
                        # Simple logic: higher demand = slightly larger
                        ratio = demands[i] / (vehicle_capacity + 1e-6)
                        dem_r = int(base_r + (4 * ratio)) if is_zoomed else base_r
                        
                    cv2.circle(canvas, pt, dem_r, COLOR_CUST_UNVISIT, -1, cv2.LINE_AA)

        # C. Draw Candidates (Highlighted)
        font_scale = 0.5 if is_zoomed else 0.4
        for rank, cand in reversed(list(enumerate(top_candidates))):
            idx = cand['id']
            pt = tuple(pts[idx])
            label = chr(65 + rank) if rank < 26 else str(rank)
            
            # Draw Node Shape
            if idx < num_depots:
                # Candidate Depot
                color = COLOR_DEPOT_OPEN if cand.get('is_open', True) else COLOR_DEPOT_CLOSED
                sz = base_r + 2
                cv2.rectangle(canvas, (pt[0]-sz, pt[1]-sz), (pt[0]+sz, pt[1]+sz), color, -1)
                cv2.rectangle(canvas, (pt[0]-sz, pt[1]-sz), (pt[0]+sz, pt[1]+sz), (0,0,0), 1) # Border
            else:
                # Candidate Customer
                sz = base_r + 2
                cv2.circle(canvas, pt, sz, COLOR_CUST_UNVISIT, -1)
                cv2.circle(canvas, pt, sz, (0,0,0), 1, cv2.LINE_AA)
            
            # Label Box
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
            box_tl = (pt[0] - w//2 - 2, pt[1] - h//2 - 2)
            box_br = (pt[0] + w//2 + 2, pt[1] + h//2 + 2)
            
            # White background for text
            cv2.rectangle(canvas, box_tl, box_br, (255,255,255), -1)
            cv2.rectangle(canvas, box_tl, box_br, (50, 50, 50), 1)
            cv2.putText(canvas, label, (pt[0]-w//2, pt[1]+h//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_TEXT, 1, cv2.LINE_AA)

        # D. Draw Current Vehicle (Agent)
        curr_pt = tuple(pts[current_node_idx])
        agent_size = 7 if is_zoomed else 4
        
        # Agent Body (Green Square/Diamond)
        cv2.rectangle(canvas, (curr_pt[0]-agent_size, curr_pt[1]-agent_size),
                      (curr_pt[0]+agent_size, curr_pt[1]+agent_size), COLOR_CURRENT_NODE, -1, cv2.LINE_AA)
        
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
    
    # Zoom Box visualization on Left ROI
    if top_candidates:
        box_p1 = global_transform(z_real_min)
        box_p2 = global_transform(z_real_max)
        x1, y1 = min(box_p1[0], box_p2[0]), min(box_p1[1], box_p2[1])
        x2, y2 = max(box_p1[0], box_p2[0]), max(box_p1[1], box_p2[1])
        
        # Clamp to image
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_height-1, x2), min(img_height-1, y2)
        
        if x2 > x1 and y2 > y1:
            cv2.rectangle(left_roi, (x1, y1), (x2, y2), COLOR_ZOOM_BOX, 2)
            # Connecting lines
            cv2.line(combined_canvas, (x2, y1), (img_height, 0), COLOR_ZOOM_BOX, 1, cv2.LINE_AA)
            cv2.line(combined_canvas, (x2, y2), (img_height, img_height), COLOR_ZOOM_BOX, 1, cv2.LINE_AA)

    # Local View
    draw_scene(right_roi, zoom_transform, is_zoomed=True)

    # --- 6. Legend & Info ---
    # Draw Legend on Left
    leg_x, leg_y = 15, img_height - 15
    
    def draw_legend_item(img, txt, col, shape="circle", is_filled=True, y_pos=0):
        # Icon
        cx, cy = leg_x, y_pos
        if shape == "square":
            if is_filled:
                cv2.rectangle(img, (cx-4, cy-4), (cx+4, cy+4), col, -1)
            else:
                cv2.rectangle(img, (cx-4, cy-4), (cx+4, cy+4), col, 1)
        else:
            cv2.circle(img, (cx, cy), 4, col, -1)
            
        # Text
        cv2.putText(img, txt, (cx + 15, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80,80,80), 1, cv2.LINE_AA)
        return y_pos - 20

    current_y = leg_y
    current_y = draw_legend_item(left_roi, "Customer (Unvisited)", COLOR_CUST_UNVISIT, "circle", True, current_y)
    current_y = draw_legend_item(left_roi, "Depot (Open)", COLOR_DEPOT_OPEN, "square", True, current_y)
    current_y = draw_legend_item(left_roi, "Depot (Closed)", COLOR_DEPOT_CLOSED, "square", False, current_y)
    
    # Titles
    cv2.putText(left_roi, "LRP Global Map", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 2, cv2.LINE_AA)
    cv2.putText(right_roi, "Local View", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,50,50), 2, cv2.LINE_AA)
    
    # Load Text in Right Corner
    load_pct = (current_load / vehicle_capacity) * 100
    load_str = f"Load: {load_pct:.0f}%"
    cv2.putText(right_roi, load_str, (img_height - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1, cv2.LINE_AA)
    
    # Border Divider
    cv2.line(combined_canvas, (img_height, 0), (img_height, img_height), (150,150,150), 2)

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
        
        # Trajectory History
        path_history = []
        if trajectory:
            for t_step in trajectory:
                val = t_step[idx]
                if hasattr(val, 'item'): val = val.item()
                path_history.append(int(val))
        if not path_history or path_history[-1] != curr_idx:
            path_history.append(curr_idx)

        # --- 3. Candidate Selection (Sorting & Info) ---
        candidates = []
        valid_indices = np.where(curr_mask)[0]
        
        # Calculate distances to all valid nodes
        diff = curr_locs[valid_indices] - curr_pos
        dists = np.linalg.norm(diff, axis=1)
        
        # Sort by distance (Nearest Neighbor heuristic base)
        sorted_arg = np.argsort(dists)
        sorted_indices = valid_indices[sorted_arg][:top_k]
        
        for i, node_idx in enumerate(sorted_indices):
            node_idx = int(node_idx)
            is_depot = node_idx < num_depots_val
            dist = np.linalg.norm(curr_locs[node_idx] - curr_pos)
            
            cand_info = {
                "id": node_idx,
                "type": "Depot" if is_depot else "Cust",
                "dist": dist,
                "x": curr_locs[node_idx][0],
                "y": curr_locs[node_idx][1],
            }
            
            # LRP Specific: Add Demand/Capacity Info
            if is_depot:
                # 如果是 Depot，显示剩余容量和开启状态
                # 注意：这里假设 depot_cap 是标量或者数组
                d_cap = cur_depot_cap if np.isscalar(cur_depot_cap) else cur_depot_cap[node_idx]
                d_use = cur_depot_usage[node_idx]
                cand_info["rem_cap"] = d_cap - d_use
                cand_info["is_open"] = bool(cur_open_depots[node_idx])
            else:
                # 如果是 Customer，显示需求量
                cand_info["demand"] = demands[idx][node_idx]
            
            candidates.append(cand_info)
            
        # Update TopK Acts for TD
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
        
        status_str = f"Stp:{step}|Nd:{curr_idx}|Ld:{cur_veh_load:.2f}/{cur_veh_cap:.1f}"
        
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
        # else: numpy array...

        obs_list.append(obs_dict)

    # Update TensorDict
    if given_topk_acts is None and len(topk_acts_list) > 0:
        try:
             td["topk_acts"] = torch.tensor(np.array(topk_acts_list), device=td.device)
        except: pass
        
    return obs_list


class RouteWorker(BaseCOWorker):
    """Wrapper for RL4CO routing environments (TSP / CVRP / OP / TDTSP)."""
    
    ENV_CONFIG = {
        'tsp': {'cls': TSPEnv, 'builder': build_obs_tsp},
        'cvrp': {'cls': CVRPEnv, 'builder': build_obs_cvrp},
        'op': {'cls': OPEnv, 'builder': build_obs_op},
        'tdtsp_matrix': {'cls': TDTSPMatrixEnv, 'builder': build_obs_tdtsp},
        'tdtsp_tw': {'cls': TDTSPTWEnv, 'builder': build_obs_tdtsp_tw},
        'tdvrp': {'cls': TDVRPEnv, 'builder': build_obs_tdvrp},
        'lrp': {'cls': LRPEnv, 'builder': build_obs_lrp},
    }
    
    def __init__(
        self,
        env_name: str = "tsp",
        seed: int = 0,
        env_num: int = 1,
        device: str = "cpu",
        num_loc: int = 10,
        loc_distribution: str = "uniform",
        return_topk_options: int = 0,
        env_kwargs: Optional[Dict[str, Any]] = None,
        image_obs: str = "rgb", 
    ):
        self.num_loc = num_loc
        self.loc_distribution = loc_distribution
        self.env_kwargs = env_kwargs or {}
        self.image_obs = image_obs
        
        super().__init__(
            env_name=env_name,
            seed=seed,
            env_num=env_num,
            device=device,
            return_topk_options=return_topk_options
        )

    def _init_env(self, seed: int, **kwargs):
        env_key = self.env_name.lower()
        if env_key not in self.ENV_CONFIG:
            raise ValueError(f"Unsupported RL4CO routing env: {self.env_name}")
            
        env_cls = self.ENV_CONFIG[env_key]['cls']
        
        # Handle TDTSP/TDTSPTW/TDVRP special initialization
        if env_key in ['tdtsp_matrix', 'tdtsp_tw', 'tdvrp']:
            # These environments require matrix and instance paths
            data_path = self.env_kwargs.get("data_path")
            base_path = self.env_kwargs.get("base_data_path")
            matrix_path = self.env_kwargs.get("matrix_path")
            service_time = self.env_kwargs.get("service_time", 180.0)
            penalty_value = self.env_kwargs.get("penalty_value", 0.0)

            if env_key == 'tdtsp_tw':
                return env_cls(
                    data_file_path=data_path,
                    base_data_path=base_path,
                    matrix_path=matrix_path,
                    service_time=service_time,
                    penalty_value=penalty_value,
                    seed=seed,
                    device=self.device
                )
            elif env_key == 'tdvrp':
                # TDVRPEnv initialization
                return env_cls(
                    generator_params={
                        "data_path": data_path,
                        "base_data_path": base_path,
                        "matrix_path": matrix_path,
                        "service_time": service_time,
                    },
                    penalty_value=penalty_value,
                    seed=seed,
                    device=self.device
                )
            else:
                # tdtsp_matrix
                generator = self.env_kwargs.get("generator")
                if generator is None:
                    generator = TDTSPTWGenerator(
                        data_path=data_path or "/root/autodl-tmp/tdtsp_dataset/berlin_20_1000.npz",
                        base_data_path=base_path or "/root/autodl-tmp/vrptdt-benchmark/instances",
                        matrix_path=matrix_path or "/root/autodl-tmp/vrptdt-benchmark/instances",
                        seed=seed
                    )
                return env_cls(
                    generator=generator,
                    seed=seed,
                    device=self.device
                )

        # Standard TSP/CVRP/OP initialization
        generator_params = {
            "num_loc": self.num_loc,
            "loc_distribution": self.loc_distribution,
        }
        if "generator_params" in self.env_kwargs:
            generator_params.update(self.env_kwargs["generator_params"])
            
        generator = self.env_kwargs.get("generator")
        return env_cls(
            generator=generator,
            generator_params=generator_params,
            seed=seed,
            device=self.device
        )

    def build_obs(self, td: TensorDict) -> List[str]:
        env_key = self.env_name.lower()
        builder = self.ENV_CONFIG[env_key]['builder']
        return builder(
            td=td, 
            env_num=self.env_num, 
            trajectory=self.actions,
            top_k=self.topk_k,
            image_obs=self.image_obs,
        )

    def step(self, action) -> Tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
        obs, rewards, dones, infos = super().step(action)
        
        # Add cumulative reward to info for TDTSP/TDVRP
        if "cumulative_reward" in self._td.keys():
            cum_rewards = _to_numpy(self._td["cumulative_reward"])
            for i in range(self.env_num):
                infos[i]["cumulative_reward"] = float(cum_rewards[i])
        
        return obs, rewards, dones, infos

# Import dependencies that might be needed from original route_obs

class RouteEnvs(BaseCOEnvs):
    def __init__(self, env_name, seed, env_num, group_n, device, resources_per_worker, is_train=True, return_topk_options=True, env_kwargs=None):
        self.num_loc_list = [env_kwargs.get("generator_params", {}).get("num_loc", 20)] * env_num
        self.loc_distribution_list = [env_kwargs.get("generator_params", {}).get("loc_distribution", "uniform")] * env_num
        
        super().__init__(
            worker_cls=RouteWorker,
            env_name=env_name,
            seed=seed,
            env_num=env_num,
            group_n=group_n,
            device=device,
            resources_per_worker=resources_per_worker,
            return_topk_options=return_topk_options,
            env_kwargs=env_kwargs
        )

    def _get_worker_args(self, worker_idx, env_name, seed, group_n, device, return_topk_options, env_kwargs):
        image_obs = env_kwargs.get("image_obs", "rgb")
        return (env_name, seed + worker_idx, group_n, device, self.num_loc_list[worker_idx], self.loc_distribution_list[worker_idx], return_topk_options, env_kwargs, image_obs), {}

def build_route_envs_temp(env_name="tdtsp_matrix", **kwargs):
    return RouteEnvs(env_name=env_name, **kwargs)
