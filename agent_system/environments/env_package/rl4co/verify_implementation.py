
import torch
import numpy as np
import cv2
import os
import sys
from tensordict import TensorDict

# Add path to sys to allow imports
sys.path.append("/root/autodl-tmp/verl-agent-co/agent_system/environments/env_package/rl4co")

# Import the functions
from route_obs import build_obs_tdtsp_tw, build_obs_tdvrp

def test_tdtsp_tw():
    print("Testing TDTSP-TW Observation Builder...")
    batch_size = 1
    num_loc = 10
    
    # Mock Data
    locs = torch.rand(batch_size, num_loc, 2)
    current_node = torch.zeros(batch_size, dtype=torch.long)
    current_time = torch.tensor([100.0])
    visited = torch.zeros(batch_size, num_loc, dtype=torch.bool)
    visited[:, 0] = True # Start at 0
    action_mask = torch.ones(batch_size, num_loc, dtype=torch.bool)
    action_mask[:, 0] = False
    
    time_windows = torch.zeros(batch_size, num_loc, 2)
    time_windows[:, :, 1] = 1000.0 # Large end time
    
    # Travel Time Matrix [B, N, N, T] -> Simple mock
    matrix = torch.zeros(batch_size, num_loc, num_loc, 5)
    
    td = TensorDict({
        "locs": locs,
        "current_node": current_node,
        "current_time": current_time,
        "visited": visited,
        "action_mask": action_mask,
        "time_windows": time_windows,
        "travel_time_matrix": matrix,
        "time_step_duration": torch.tensor(10.0),
        "reward": torch.zeros(batch_size),
    }, batch_size=batch_size)
    
    try:
        obs = build_obs_tdtsp_tw(td, env_num=batch_size, image_obs="path")
        print("TDTSP-TW Success!")
        print(f"Generated {len(obs)} observations.")
        print("Sample Text Obs:\n", obs[0]['text'][:200] + "...")
        if 'image' in obs[0]:
            print(f"Image generated at: {obs[0]['image']}")
    except Exception as e:
        print(f"TDTSP-TW Failed: {e}")
        import traceback
        traceback.print_exc()

def test_tdvrp():
    print("\nTesting TDVRP Observation Builder...")
    batch_size = 1
    num_loc = 10
    
    # Mock Data
    locs = torch.rand(batch_size, num_loc, 2)
    current_node = torch.zeros(batch_size, dtype=torch.long)
    current_time = torch.tensor([100.0])
    visited = torch.zeros(batch_size, num_loc, dtype=torch.bool)
    # action_mask for VRP: usually 1 if visitable. 
    action_mask = torch.ones(batch_size, num_loc, dtype=torch.bool)
    action_mask[:, 0] = True # Depot is visitable (return)
    
    time_windows = torch.zeros(batch_size, num_loc, 2)
    time_windows[:, :, 1] = 1000.0
    
    matrix = torch.zeros(batch_size, num_loc, num_loc, 5)
    
    td = TensorDict({
        "locs": locs,
        "current_node": current_node,
        "current_time": current_time,
        "visited": visited,
        "action_mask": action_mask,
        "time_windows": time_windows,
        "travel_time_matrix": matrix,
        "time_step_duration": torch.tensor(10.0),
        "reward": torch.zeros(batch_size),
    }, batch_size=batch_size)
    
    try:
        obs = build_obs_tdvrp(td, env_num=batch_size, image_obs="path")
        print("TDVRP Success!")
        print(f"Generated {len(obs)} observations.")
        print("Sample Text Obs:\n", obs[0]['text'][:200] + "...")
        if 'image' in obs[0]:
            print(f"Image generated at: {obs[0]['image']}")
    except Exception as e:
        print(f"TDVRP Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_tdtsp_tw()
    test_tdvrp()
