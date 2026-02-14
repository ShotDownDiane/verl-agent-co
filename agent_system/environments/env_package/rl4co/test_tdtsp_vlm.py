import torch
from rl4co.envs.routing.tdtsp.env import TDTSPMatrixEnv
from rl4co.envs.routing.tdtsp.env_tw import TDTSPTWGenerator
from route_envs import RouteWorker
import os

def test_tdtsp_obs():
    print("Testing TDTSP Matrix Observation Building...")
    
    # 1. Setup Generator (using real data if available, or dummy)
    data_path = "/root/autodl-tmp/tdtsp_dataset/berlin_20_1000.npz"
    base_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found. Skipping real data test.")
        return

    generator = TDTSPTWGenerator(
        data_path=data_path,
        base_data_path=base_path,
        matrix_path=matrix_path,
    )
    
    # 2. Setup Worker
    worker = RouteWorker(
        env_name="tdtsp_matrix",
        env_num=1,
        device="cpu",
        env_kwargs={"generator": generator},
        image_obs="path" # Save to local files
    )
    
    # 3. Reset and Get Obs
    print("Resetting environment...")
    obs_list = worker.reset()
    current_obs_data = obs_list[0]
    current_obs = current_obs_data["text"] if isinstance(current_obs_data, dict) else current_obs_data
    print("\nInitial Observation:")
    print(current_obs)
    
    # 4. Step and Get Obs
    print("\nStepping environment with action 5...")
    # Get available actions
    td = worker._td
    mask = td["action_mask"][0]
    available = torch.where(mask)[0]
    action = int(available[1]) # Pick one
    
    obs_list, reward, done, info = worker.step([action])
    current_obs_data = obs_list[0]
    current_obs = current_obs_data["text"] if isinstance(current_obs_data, dict) else current_obs_data
    print(f"\nObservation after stepping to Node {action}:")
    print(current_obs)
    
    print("\nTest completed successfully!")

def test_tdtsptw_obs():
    print("\n" + "="*50)
    print("Testing TDTSPTW Observation Building...")
    print("="*50)
    
    # 1. Setup Generator
    data_path = "/root/autodl-tmp/tdtsp_dataset/berlin_20_1000.npz"
    base_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    
    if not os.path.exists(data_path):
        print(f"Data path {data_path} not found. Skipping.")
        return

    generator = TDTSPTWGenerator(
        data_path=data_path,
        base_data_path=base_path,
        matrix_path=matrix_path,
    )
    
    # 2. Setup Worker for tdtsp_tw
    worker = RouteWorker(
        env_name="tdtsp_tw",
        env_num=1,
        device="cpu",
        env_kwargs={
            "data_path": data_path,
            "base_data_path": base_path,
            "matrix_path": matrix_path,
            "service_time": 180.0,
            "penalty_value": 1.0
        },
        image_obs="path"
    )
    
    # 3. Reset
    print("Resetting TDTSPTW environment...")
    obs_list = worker.reset()
    current_obs_data = obs_list[0]
    current_obs = current_obs_data["text"] if isinstance(current_obs_data, dict) else current_obs_data
    print("\nInitial Observation (TW-Aware):")
    print(current_obs)
    
    # 4. Step
    td = worker._td
    mask = td["action_mask"][0]
    available = torch.where(mask)[0]
    action = int(available[1])
    
    print(f"\nStepping to Node {action}...")
    obs, reward, done, info = worker.step([action])
    print(f"\nObservation after Step 1:")
    print(obs[0])
    
    # Step again to see step index in filename
    mask = worker._td["action_mask"][0]
    available = torch.where(mask)[0]
    action = int(available[2])
    print(f"\nStepping to Node {action}...")
    obs, reward, done, info = worker.step([action])
    print(f"\nObservation after Step 2:")
    print(obs[0])

if __name__ == "__main__":
    # test_tdtsp_obs()
    test_tdtsptw_obs()
