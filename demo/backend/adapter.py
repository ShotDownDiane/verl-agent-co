import sys
import os
import torch
import numpy as np
import base64
import cv2
from typing import Dict, Any, List

# Add project root to sys.path
sys.path.append("/root/autodl-tmp/verl-agent-co")

# Import Environment
from env.tdtsp.env_tw import TDTSPTWEnv

# Import Observation Builder
# We need to make sure agent_system is in path or relative import works
# Since we added root to sys.path, we can import agent_system...
from agent_system.environments.env_package.rl4co.route_obs import build_obs_tdtsp_tw

class RL4COEnvAdapter:
    def __init__(self):
        self.device = "cpu" # Demo on CPU for simplicity, or "cuda" if available
        
        # Hardcoded paths matching the project configuration
        self.data_path = "/root/autodl-tmp/tdtsp_dataset_random/berlin_50_random_test.npz" # Use test set for demo
        self.base_data_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
        self.matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
        
        # Initialize Environment
        self.env = TDTSPTWEnv(
            data_file_path=self.data_path,
            base_data_path=self.base_data_path,
            matrix_path=self.matrix_path,
            penalty_value=3.0,
            device=self.device
        )
        
        # Load raw data for coordinate mapping
        self.raw_dataset = np.load(self.data_path)
        self.raw_locs = self.raw_dataset['locs'] # [1000, 51, 2]
        
        self.td = None # Current TensorDict state
        self.env_num = 1 # Single environment for demo

    def reset(self) -> Dict[str, Any]:
        """
        Resets the environment and returns the initial state dictionary.
        """
        # Reset environment with batch_size=1
        self.td = self.env.reset(batch_size=[self.env_num])
        
        # Build initial observation
        return self._build_state_response()

    def step(self, action_node_id: int) -> Dict[str, Any]:
        """
        Takes a step in the environment.
        """
        if self.td is None:
            raise ValueError("Environment not initialized. Call reset() first.")
            
        # Update action in TensorDict
        action_tensor = torch.tensor([action_node_id], dtype=torch.long, device=self.device)
        self.td["action"] = action_tensor
        
        # Step environment
        # rl4co env.step returns a dict containing 'next' state
        step_result = self.env.step(self.td)
        self.td = step_result["next"]
        
        return self._build_state_response()

    def _build_state_response(self) -> Dict[str, Any]:
        """
        Constructs the dictionary response required by the frontend/backend.
        Includes:
        1. Virtual Map (Base64 Image)
        2. Text Prompt (Context)
        3. Raw State (Nodes, Path, Cost, etc.)
        """
        # 1. Get Observation (Image + Text)
        # build_obs_tdtsp_tw returns a list of dicts (one per env)
        obs_list = build_obs_tdtsp_tw(
            self.td, 
            env_num=self.env_num, 
            image_obs="base64" # This triggers image generation
        )
        
        obs = obs_list[0]
        text_prompt = obs["text"]
        
        # Image is returned as base64 string in 'image' key if render logic succeeded
        # In route_obs.py, it calls render_tdtsptw_smart_dual_view which returns b64_str
        # But wait, looking at route_obs.py code I read earlier:
        # img_b64, image_rgb_np = render_tdtsptw_smart_dual_view(...)
        # obs_item = { "text": ..., "image": img_b64, ... }
        virtual_map_b64 = obs.get("image", "")
        
        # 2. Extract Raw State for Frontend
        nodes_data = self._extract_nodes()
        current_path = self._extract_path()
        current_cost = self._extract_cost()
        capacity_info = self._extract_capacity()
        
        return {
            "virtual_map": virtual_map_b64,
            "text_prompt": text_prompt,
            "raw_state": {
                "nodes": nodes_data,
                "current_path": current_path,
                "current_cost": current_cost,
                "capacity": capacity_info["capacity"],
                "remaining_capacity": capacity_info["remaining_capacity"]
            }
        }

    def _extract_nodes(self) -> List[Dict[str, Any]]:
        """
        Extracts node information: ID, Lat/Lon (simulated or real), x/y, demand, time window.
        """
        # Normalized coords from environment (used for Virtual Map alignment)
        norm_locs = self.td["locs"][0].cpu().numpy() # [N, 2]
        
        # Real coords from raw dataset (used for Real Map)
        # We need to find the index of the current instance
        real_locs = None
        if "locs_idx" in self.td.keys():
            try:
                # Debug info
                print(f"DEBUG: locs_idx shape: {self.td['locs_idx'].shape}")
                print(f"DEBUG: locs_idx values: {self.td['locs_idx']}")
                
                # Check if it's scalar or vector
                idx_tensor = self.td["locs_idx"]
                if idx_tensor.numel() == 1:
                    instance_idx = idx_tensor.item()
                else:
                    # If it has multiple elements, maybe it's [Batch, 1] or [Batch]
                    # Or maybe [Batch, Nodes] but all same?
                    instance_idx = idx_tensor.flatten()[0].item()
                    
                print(f"DEBUG: instance_idx: {instance_idx}")
                real_locs = self.raw_locs[instance_idx] # [N, 2]
            except Exception as e:
                print(f"ERROR getting real locs: {e}")
                real_locs = None
        
        if real_locs is None:
            # Fallback if locs_idx is missing (unlikely given previous checks)
            # Use heuristic mapping based on Berlin bounds
            real_locs = norm_locs.copy()
            # This path shouldn't be hit with standard RL4CO envs
        
        # Check if 'demand' key exists.
        has_demand = "demand" in self.td.keys()
        if has_demand:
            demands = self.td["demand"][0].cpu().numpy() # [N]
        else:
            demands = None
        
        # Time Windows
        tws = self.td["time_windows"][0].cpu().numpy() # [N, 2]
        
        nodes = []
        num_nodes = norm_locs.shape[0]
        
        # Use norm_locs for x/y to ensure consistency with Virtual Map (0-224)
        # Note: norm_locs are usually [0,1].
        # In route_obs.py:
        # x_pix = x * 224
        # y_pix = 224 - y * 224
        
        for i in range(num_nodes):
            # Real Lat/Lon
            lat = float(real_locs[i][0])
            lon = float(real_locs[i][1])
            
            # Virtual Map Coords
            nx = float(norm_locs[i][0])
            ny = float(norm_locs[i][1])
            
            x = nx * 224
            y = (1.0 - ny) * 224 # Flip Y
            
            node_type = "depot" if i == 0 else "customer"
            demand = float(demands[i]) if demands is not None else 1.0
            
            nodes.append({
                "id": i,
                "type": node_type,
                "lat": lat,
                "lon": lon,
                "x": x,
                "y": y,
                "demand": demand,
                "time_window": [float(tws[i][0]), float(tws[i][1])]
            })
        return nodes

    def _extract_path(self) -> List[int]:
        # visited is [1, N] mask? Or we need to track history.
        # rl4co env usually tracks 'visited' mask, but for path history we might need to rely on 
        # what we stored or if the env keeps it.
        # TDTSPTWEnv might not keep full history in `td` by default unless we modify it.
        # However, `build_obs_tdtsp_tw` takes `trajectory` arg.
        # But wait, `route_obs.py` says:
        # if trajectory is not None: ...
        # if len(path_history) == 0 ... path_history.append(curr_idx)
        
        # The environment itself doesn't automatically store the full sequence of actions in `td` 
        # in a way that is easily accessible as a list, except maybe via `visited` which is unordered.
        # BUT, `main.py` (or this adapter) should maintain the history if the env doesn't.
        # Actually, `TDTSPTWEnv` state usually has `current_node`.
        # Let's verify if we need to track it ourselves.
        # For this adapter, since `step` is called incrementally, I can just store the history locally.
        
        # Let's verify if I can just extract it from my own tracking.
        # I'll initialize a history list in `reset`.
        return self.path_history

    def _extract_cost(self) -> float:
        # Used capacity / cost
        # `reward` in rl4co is usually negative cost.
        # We can track cumulative cost manually or check if env has it.
        # TDTSPTWEnv has `current_time` which acts as cost/time.
        current_time = self.td["current_time"][0].item()
        return float(current_time)

    def _extract_capacity(self) -> Dict[str, float]:
        # Capacity
        # If it's TSP, capacity might be infinite or irrelevant.
        # If it's VRP, `used_capacity` or `demand` is tracked.
        # Let's check keys.
        remaining = 100.0
        capacity = 100.0
        
        if "used_capacity" in self.td.keys():
            used = self.td["used_capacity"][0].item()
            if "vehicle_capacity" in self.td.keys():
                capacity = self.td["vehicle_capacity"][0].item()
            remaining = capacity - used
            
        return {"capacity": capacity, "remaining_capacity": remaining}

    # Override step to update history
    def step(self, action_node_id: int) -> Dict[str, Any]:
        if self.td is None:
            raise ValueError("Environment not initialized.")
            
        action_tensor = torch.tensor([action_node_id], dtype=torch.long, device=self.device)
        self.td["action"] = action_tensor
        
        step_result = self.env.step(self.td)
        self.td = step_result["next"]
        
        # Update history
        self.path_history.append(action_node_id)
        
        return self._build_state_response()

    def reset(self) -> Dict[str, Any]:
        self.td = self.env.reset(batch_size=[self.env_num])
        
        # Initialize history with depot (assuming start at 0)
        # Check current node
        current_node = self.td["current_node"][0].item()
        self.path_history = [int(current_node)]
        
        return self._build_state_response()

if __name__ == "__main__":
    # Simple Test
    adapter = RL4COEnvAdapter()
    state = adapter.reset()
    print("Keys in state:", state.keys())
    print("Nodes count:", len(state["raw_state"]["nodes"]))
    print("Initial Path:", state["raw_state"]["current_path"])
    
    # Step to a node (e.g. 1)
    # Ensure 1 is not 0 (depot)
    next_node = 1
    new_state = adapter.step(next_node)
    print("Path after step:", new_state["raw_state"]["current_path"])
    print("Virtual Map (first 50 chars):", new_state["virtual_map"][:50])
