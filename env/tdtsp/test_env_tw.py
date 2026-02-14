
import os
import torch
import numpy as np
from .env_tw import TDTSPTWEnv
from tensordict import TensorDict

def test_env_tw():
    print("Initializing TDTSPTWEnv...")
    
    # Define paths
    data_path = "/root/autodl-tmp/tdtsp_dataset_split/berlin_20_test.npz"
    base_data_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    
    # Create Environment
    # Use smaller num_matrix_steps for faster test if needed, but 200 is default
    env = TDTSPTWEnv(
        data_file_path=data_path,
        base_data_path=base_data_path,
        matrix_path=matrix_path,
        num_matrix_steps=37, # Reduce for speed in test
        force_rebuild_matrix=False
    )
    
    print("Environment initialized.")
    
    # Reset
    print("Resetting environment...")
    td = env.reset(batch_size=[2]) # Small batch size
    print("Reset complete.")
    
    print("Initial TensorDict keys:", td.keys())
    print("Locs shape:", td["locs"].shape)
    print("Time Windows shape:", td["time_windows"].shape)
    print("Locs Indices shape:", td["locs_idx"].shape)
    print("Current Time:", td["current_time"])
    
    # Check if matrix is loaded in generator
    print("Matrix shape in generator:", env.generator.matrix.shape)
    
    # Step 1
    print("\n--- Step 1 ---")
    # Action: pick node 1 for batch 0, node 2 for batch 1
    action = torch.tensor([1, 2], dtype=torch.long)
    td["action"] = action
    
    td = env.step(td)["next"]
    
    print("Current Node:", td["current_node"])
    print("Current Time:", td["current_time"])
    print("Reward:", td["reward"])
    print("Done:", td["done"])
    print("Action Mask sum:", td["action_mask"].sum(dim=-1))
    
    # Check for violation (should be none if we picked valid nodes)
    # Note: we don't know if node 1 or 2 are valid w.r.t TW.
    # But let's see if current_time updated.
    
    # Step 2
    print("\n--- Step 2 ---")
    action = torch.tensor([2, 1], dtype=torch.long) # Swap
    td["action"] = action
    td = env.step(td)["next"]
    
    print("Current Node:", td["current_node"])
    print("Current Time:", td["current_time"])
    print("Reward:", td["reward"])
    print("Done:", td["done"])
    
    # Run until done
    print("\n--- Running until done ---")
    step = 2
    while not td["done"].all():
        # Greedy policy: pick first available
        mask = td["action_mask"]
        # Find first true index
        action = torch.argmax(mask.float(), dim=-1)
        td["action"] = action
        td = env.step(td)["next"]
        step += 1
        if step > 25:
            print("Exceeded max steps!")
            break
            
    print(f"Finished in {step} steps.")
    print("Final Reward:", td["reward"])
    print("Final Current Time:", td["current_time"])
    
    # Check validity
    # If reward is very negative, it means violation or high cost
    # We expect negative reward = -makespan
    print("Test passed (basic execution).")

if __name__ == "__main__":
    test_env_tw()
