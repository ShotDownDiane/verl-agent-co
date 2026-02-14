
import torch
from route_envs import RouteWorker

def test_tdvrp_vlm():
    # Data paths (using Berlin 20 nodes as example)
    data_path = "/root/autodl-tmp/tdtsp_dataset/berlin_20_1000.npz"
    base_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    
    print("Initializing TDVRP RouteWorker...")
    worker = RouteWorker(
        env_name="tdvrp",
        env_num=1,
        device="cpu",
        env_kwargs={
            "data_path": data_path,
            "base_data_path": base_path,
            "matrix_path": matrix_path,
            "service_time": 180.0,
            "penalty_value": 0.0 # Hard time windows
        },
        image_obs="path"
    )
    
    print("Resetting environment...")
    obs_list, infos = worker.reset()
    
    current_obs_data = obs_list[0]
    current_obs = current_obs_data["text"] if isinstance(current_obs_data, dict) else current_obs_data
    print(f"Initial Observation:\n{current_obs[:500]}...") 
    
    # Check if image was generated
    import os
    if isinstance(current_obs_data, dict):
        image_path = current_obs_data.get("image", "None")
    else:
        image_path = current_obs_data.split("[IMAGE] ")[-1].strip()
    
    if image_path and os.path.exists(image_path):
        print(f"Success: Image generated at {image_path}")
    else:
        print(f"Error: Image NOT found at {image_path}")
        
    # Step 1: Pick a customer
    # Parse available options from obs
    import re
    options = re.findall(r"([A-Z])\. Node (\d+)", current_obs)
    if options:
        label, node_id = options[0]
        print(f"Picking Option {label} (Node {node_id})")
        
        actions = torch.tensor([[int(node_id)]])
        obs_list, rewards, dones, infos = worker.step(actions)
        
        current_obs_data = obs_list[0]
        current_obs = current_obs_data["text"] if isinstance(current_obs_data, dict) else current_obs_data
        print(f"Step Reward: {rewards[0]}")
        print(f"Cumulative Reward: {infos[0].get('cumulative_reward', 0.0):.2f}")
        print(f"Done: {dones[0]}")
        
        # Check next obs
        print(f"Next Observation:\n{current_obs[:500]}...")
        
        # Step 2: Pick Depot (if possible)
        options = re.findall(r"([A-Z])\. Node (\d+) \(DEPOT", current_obs)
        if options:
            label, node_id = options[0]
            print(f"Picking Option {label} (Depot Node {node_id})")
            actions = torch.tensor([[int(node_id)]])
            obs_list, rewards, dones, infos = worker.step(actions)
            current_obs_data = obs_list[0]
            current_obs = current_obs_data["text"] if isinstance(current_obs_data, dict) else current_obs_data
            print(f"Returned to Depot. Step Reward: {rewards[0]}")
            print(f"Cumulative Reward: {infos[0].get('cumulative_reward', 0.0):.2f}")
            print(f"Done: {dones[0]}")
        else:
            print("Depot not available or not in top-K yet.")
    else:
        print("No options found in observation.")

if __name__ == "__main__":
    test_tdvrp_vlm()
