
import torch
import sys
import os
import pickle
import numpy as np
from tensordict.tensordict import TensorDict

# Add the project root to path
sys.path.append("/root/autodl-tmp/verl-agent-co")

from agent_system.environments.env_package.rl4co.graph_env import GraphWorker
from rl4co.envs.common.utils import Generator

class CustomMCPGenerator(Generator):
    """
    Custom MCP Generator that loads data from a file.
    Reference data format: list of (unused, weights, membership)
    - weights: list of item weights
    - membership: list of lists, where membership[i] = list of item indices covered by set i
    """
    def __init__(self, data_path, n_sets_to_choose=50, device="cpu", **kwargs):
        self.data_path = data_path
        self.n_sets_to_choose = n_sets_to_choose
        self.device = device
        
        # Load data
        print(f"Loading data from {data_path}...")
        with open(data_path, "rb") as f:
            self.data = pickle.load(f)
            
        self.num_samples = len(self.data)
        
        # Analyze first sample to get dimensions
        _, weights_0, membership_0 = self.data[0]
        self.num_items = len(weights_0)
        self.num_sets = len(membership_0)
        
        print(f"Loaded {self.num_samples} samples.")
        print(f"Num Items: {self.num_items}, Num Sets: {self.num_sets}")

    def _generate(self, batch_size) -> TensorDict:
        bs = batch_size[0] if isinstance(batch_size, torch.Size) else batch_size
        
        if bs > self.num_samples:
            print(f"Warning: Requested batch size {bs} > available samples {self.num_samples}. Recycling data.")
            indices = torch.randint(0, self.num_samples, (bs,), device=self.device)
        else:
            # Use first bs samples
            indices = torch.arange(bs, device=self.device)
            
        # Process batch data
        batch_weights = []
        batch_membership = []
        
        max_set_size = 0
        
        # First pass: collect data and find max set size
        selected_data = [self.data[i] for i in indices.cpu().numpy()]
        
        for _, weights, membership in selected_data:
            batch_weights.append(weights)
            # membership is list of lists
            # Find max size in this sample
            sample_max = max(len(s) for s in membership)
            if sample_max > max_set_size:
                max_set_size = sample_max
                
        # Second pass: build tensors
        # Membership tensor: (B, NumSets, MaxSize)
        # Items are 1-based (0 is padding)
        membership_tensor = torch.zeros((bs, self.num_sets, max_set_size), dtype=torch.long, device=self.device)
        weights_tensor = torch.tensor(np.array(batch_weights), dtype=torch.float, device=self.device)
        
        for i, (_, _, membership) in enumerate(selected_data):
            for j, s in enumerate(membership):
                # s is list of item indices (0-based from reference)
                # Convert to 1-based for MCPEnv
                # Pad with 0
                if len(s) > 0:
                    items = torch.tensor(s, dtype=torch.long, device=self.device) + 1
                    membership_tensor[i, j, :len(items)] = items
                    
        return TensorDict(
            {
                "membership": membership_tensor.float(), # MCPEnv expects float for some reason? 
                # Wait, MCPEnv uses it as indices. Generator returns float?
                # MCPGenerator returns: membership_tensor.float()
                # But MCPEnv uses: chosen_membership[...].long()
                # So float is fine, but long is better. Let's stick to float to match Generator.
                
                "weights": weights_tensor,
                "n_sets_to_choose": torch.full((bs, 1), self.n_sets_to_choose, dtype=torch.float, device=self.device),
                
                # BaseEnv requires these for tracking
                "chosen": torch.zeros(bs, self.num_sets, dtype=torch.bool, device=self.device),
                "i": torch.zeros(bs, dtype=torch.long, device=self.device),
            },
            batch_size=bs,
            device=self.device
        )

def main():
    # Configuration
    data_path = "/root/autodl-tmp/unsupervised-CO-ucom2/facility_location_and_max_cover/data/max_covering_500_test.pkl"
    
    env_name = "mcp"
    n_choose = 50 # From reference
    env_num = 10 
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Initialize Generator
    generator = CustomMCPGenerator(data_path, n_sets_to_choose=n_choose, device=device)

    # Initialize GraphWorker
    print(f"Initializing {env_name.upper()} environment...")
    worker = GraphWorker(
        env_name=env_name,
        seed=1234,
        env_num=env_num,
        device=device,
        generator=generator,
        check_solution=False
    )
    
    # Reset Environment
    print("Resetting environment...")
    obs, infos = worker.reset()
    
    print(f"Observation (first instance):\n{obs[0][:500]}...")
    
    # Check loaded data
    td = worker._td
    print("\nEnvironment State:")
    print(f"Batch size: {td.batch_size}")
    print(f"Membership shape: {td['membership'].shape}")
    print(f"Weights shape: {td['weights'].shape}")
    
    # Random Rollout
    print("\nRunning random rollout...")
    total_reward = 0
    steps = 0
    
    while not worker.done:
        mask = td["action_mask"]
        actions = []
        for i in range(env_num):
            valid_indices = torch.nonzero(mask[i]).squeeze(-1)
            if len(valid_indices) > 0:
                act = valid_indices[torch.randint(0, len(valid_indices), (1,)).item()]
            else:
                act = 0 
            actions.append(act.item())
        
        obs, rewards, dones, infos = worker.step(actions)
        steps += 1
        
        if steps % 10 == 0:
            print(f"Step {steps}: Chosen {td['i'][0]} / {n_choose}")

    print("Rollout complete.")
    
    rewards = torch.tensor(rewards)
    avg_reward = rewards.float().mean().item()
    print(f"Average Reward (Total Covered Weight): {avg_reward:.4f}")

if __name__ == "__main__":
    main()
