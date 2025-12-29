import os
import torch
import numpy as np
import requests
import tarfile
import gzip
import shutil
import tsplib95
from tqdm.auto import tqdm
from rl4co.envs import TSPEnv
from rl4co.envs.routing.tsp.generator import TSPGenerator
from tensordict import TensorDict
from typing import List, Optional

# Import RouteWorker from local envs.py
import sys
import os

# Add project root to path to allow importing local modules as packages
project_root = "/root/autodl-tmp/verl-agent-co"
if project_root not in sys.path:
    sys.path.append(project_root)

from agent_system.environments.env_package.rl4co.route_envs import RouteWorker

# --- Utils from Notebook ---

def download_and_extract_tsplib(url, directory="tsplib_data", delete_after_unzip=True):
    os.makedirs(directory, exist_ok=True)
    
    tar_path = os.path.join(directory, "tsplib.tar.gz")
    
    # Check if data already exists to avoid re-downloading
    if os.path.exists(tar_path) or (os.path.exists(directory) and len(os.listdir(directory)) > 1):
        print("TSPLib data seems to be present. Skipping download.")
        # If tar exists but not extracted, we might want to extract. 
        # But simple check: if directory has many files, assume done.
        files = os.listdir(directory)
        if len([f for f in files if f.endswith('.tsp')]) > 0:
            return

    print(f"Downloading TSPLib from {url}...")
    # Download with progress bar
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total_size = int(r.headers.get('content-length', 0))
        with open(tar_path, 'wb') as f, tqdm(total=total_size, unit='B', unit_scale=True) as pbar:
            for chunk in r.iter_content(8192):
                f.write(chunk)
                pbar.update(len(chunk))

    print("Extracting...")
    # Extract tar.gz
    with tarfile.open(tar_path, 'r:gz') as tar:
        tar.extractall(directory)

    # Decompress .gz files inside directory
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".gz"):
                path = os.path.join(root, file)
                with gzip.open(path, 'rb') as f_in, open(path[:-3], 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
                os.remove(path)

    if delete_after_unzip and os.path.exists(tar_path):
        os.remove(tar_path)
    print("Download and extraction complete.")

def normalize_coord(coord: torch.Tensor) -> torch.Tensor:
    x, y = coord[:, 0], coord[:, 1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    # Avoid division by zero if all points are same (unlikely for TSP)
    if x_max == x_min: x_max += 1e-6
    if y_max == y_min: y_max += 1e-6
    
    x_scaled = (x - x_min) / (x_max - x_min) 
    y_scaled = (y - y_min) / (y_max - y_min)
    coord_scaled = torch.stack([x_scaled, y_scaled], dim=1)
    return coord_scaled

def load_tsplib_problems(directory):
    files = os.listdir(directory)
    # Recursively find files if they are in subdirs
    # But tar extraction usually puts them flat or in one folder.
    # The notebook code assumes flat or we just look at top level.
    # Let's walk to be safe.
    tsp_files = []
    for root, _, fs in os.walk(directory):
        for f in fs:
            if f.endswith('.tsp'):
                tsp_files.append(os.path.join(root, f))
    
    problems = []
    print(f"Found {len(tsp_files)} TSP files.")
    
    for prob_path in tsp_files:
        try:
            # Suppress warnings from tsplib95
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                problem = tsplib95.load(prob_path)
            
            # Skip if no node coords
            if not hasattr(problem, 'node_coords') or not problem.node_coords:
                continue
                
            node_coords = torch.tensor([v for v in problem.node_coords.values()], dtype=torch.float32)
            
            # Normalize
            node_coords_norm = normalize_coord(node_coords)
            
            problems.append({
                "name": problem.name,
                "node_coords": node_coords_norm,
                "original_coords": node_coords,
                "dimension": problem.dimension
            })
        except Exception as e:
            print(f"Failed to load {prob_path}: {e}")
            continue
            
    # Order by dimension
    problems = sorted(problems, key=lambda x: x['dimension'])
    return problems

# --- Custom Generator and Worker ---

class TSPLibGenerator(TSPGenerator):
    """Generator that returns a specific TSPLib instance."""
    def __init__(self, locs: torch.Tensor, **kwargs):
        self.fixed_locs = locs
        super().__init__(num_loc=locs.size(0), **kwargs)

    def _generate(self, batch_size) -> TensorDict:
        # Ensure batch_size is torch.Size
        if not isinstance(batch_size, torch.Size):
            batch_size = torch.Size(batch_size) if isinstance(batch_size, list) else torch.Size([batch_size])
        
        # Expand fixed_locs to match batch_size
        # fixed_locs is [num_loc, 2]
        # output should be [*batch_size, num_loc, 2]
        
        # Use expand
        locs = self.fixed_locs.expand(*batch_size, *self.fixed_locs.shape)
        
        return TensorDict(
            {
                "locs": locs,
            },
            batch_size=batch_size,
        )

class TSPLibWorker(RouteWorker):
    def __init__(self, tsplib_locs, **kwargs):
        self.tsplib_locs = tsplib_locs
        # Override num_loc to match data
        kwargs['num_loc'] = tsplib_locs.shape[0]
        super().__init__(**kwargs)
    
    def _init_env(self, seed: int, **kwargs):
        # Create generator with the specific data
        generator = TSPLibGenerator(self.tsplib_locs, min_loc=0.0, max_loc=1.0)
        # Initialize TSPEnv with this generator
        return TSPEnv(
            generator=generator, 
            seed=seed, 
            device=self.device
        )

# --- Main Execution ---

def main():
    device = "cpu"
    print(f"Using device: {device}")
    
    # 1. Prepare Data
    data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tsplib_data")
    download_url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/ALL_tsp.tar.gz"
    
    # download_and_extract_tsplib(download_url, directory=data_dir)
    # User confirmed data is downloaded.
    if not os.path.exists(data_dir):
        print(f"Warning: Data directory {data_dir} does not exist.")
    
    problems = load_tsplib_problems(data_dir)
    print(f"Loaded {len(problems)} valid problems.")
    
    if not problems:
        print("No problems found. Exiting.")
        return

    # 2. Select a problem to test (e.g., eil51 or the smallest one)
    # Let's pick a small one for testing
    problem = problems[0] # Smallest dimension
    print(f"Testing on problem: {problem['name']} (Dimension: {problem['dimension']})")
    
    # 3. Initialize Worker
    # We use env_num=2 to test batching capability, though they will be identical instances
    env_num = 1
    worker = TSPLibWorker(
        tsplib_locs=problem['node_coords'].to(device),
        env_name="tsp",
        env_num=env_num,
        device=device,
        return_topk_options=5
    )
    
    # 4. Run Test Loop
    print("\nResetting environment...")
    obs_list, info_list = worker.reset()
    
    print("\nInitial Observation (Env 0):")
    print(obs_list[0])
    
    # Step through with random actions just to verify mechanics
    done = False
    step = 0
    
    # Need to access the internal TD to check done status if we were running raw env,
    # but worker.step returns dones list.
    
    while not done and step < 10: # Limit steps for safety
        # Simple policy: choose random valid action
        # We need to know which actions are valid. 
        # The worker doesn't expose mask directly in step() return, 
        # but we can access worker._td["action_mask"]
        
        mask = worker._td["action_mask"]
        actions = []
        for i in range(env_num):
            actions.append(1)
        
        print(f"Step {step}: Actions {actions}")
        
        obs_list, rewards, dones, infos = worker.step(actions)
        
        if step < 2:
            print(f"\nObservation (Env 0) at Step {step+1}:\n{obs_list[0]}")
            
        done = all(dones)
        step += 1
        
    print("\nTest finished.")

if __name__ == "__main__":
    main()
