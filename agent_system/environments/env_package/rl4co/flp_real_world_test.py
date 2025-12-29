
import torch
import sys
import os
import math
from tensordict.tensordict import TensorDict

# Add the project root to path
sys.path.append("/root/autodl-tmp/verl-agent-co")

from agent_system.environments.env_package.rl4co.graph_env import GraphWorker
from rl4co.envs.common.utils import Generator
from rl4co.utils.ops import get_distance_matrix

class CustomFLPGenerator(Generator):
    """
    Custom FLP Generator that loads data from a file instead of generating it randomly.
    """
    def __init__(self, data_path, n_choose=30, device="cpu", **kwargs):
        self.data_path = data_path
        self.n_choose = n_choose
        self.device = device
        
        # Load data
        print(f"Loading data from {data_path}...")
        self.data = torch.load(data_path, map_location=device)
        self.num_samples = len(self.data)
        self.num_loc = self.data.shape[1]
        
        # Compute distances
        # Reference uses squared Euclidean distance, but RL4CO FLPEnv typically uses Euclidean.
        # We will use Euclidean distance to be consistent with standard FLP,
        # unless strict adherence to the reference's squared distance is required.
        # Given "real world test" usually implies standard metric, we stick to Euclidean.
        # But if the reference model was trained on squared, we might need squared.
        # For now, let's use Euclidean as it's more standard for "distance".
        self.distances = get_distance_matrix(self.data)
        
        # Calculate max distance for initialization (upper bound)
        # Assuming coordinates are roughly normalized or bounded
        self.max_dist = math.sqrt(2) * 2.0 # Heuristic max
        
        print(f"Loaded {self.num_samples} samples with {self.num_loc} locations.")

    def _generate(self, batch_size) -> TensorDict:
        # We ignore batch_size and return the full dataset or a slice?
        # GraphWorker passes batch_size=env_num to reset.
        # We should ensure the requested batch_size matches our data or we slice/sample it.
        
        bs = batch_size[0] if isinstance(batch_size, torch.Size) else batch_size
        
        if bs > self.num_samples:
            print(f"Warning: Requested batch size {bs} > available samples {self.num_samples}. Recycling data.")
            indices = torch.randint(0, self.num_samples, (bs,), device=self.device)
        else:
            # Use first bs samples (deterministic for testing)
            indices = torch.arange(bs, device=self.device)
            
        locs = self.data[indices]
        orig_distances = self.distances[indices]
        
        return TensorDict(
            {
                "locs": locs,
                "orig_distances": orig_distances,
                "distances": torch.full(
                    (bs, self.num_loc), self.max_dist, dtype=torch.float, device=self.device
                ),
                "chosen": torch.zeros(bs, self.num_loc, dtype=torch.bool, device=self.device),
                "to_choose": torch.ones(bs, dtype=torch.long, device=self.device) * self.n_choose,
                "i": torch.zeros(bs, dtype=torch.long, device=self.device), # Added counter
            },
            batch_size=bs,
            device=self.device
        )

def main():
    # Configuration
    data_path = "/root/autodl-tmp/unsupervised-CO-ucom2/facility_location_and_max_cover/data/facility_location_rand500_test.pt"
    # Note: Reference file uses 'train.pt' even for 'data_test', but 'test.pt' exists. 
    # Using 'test.pt' for a test script seems more appropriate.
    
    env_name = "flp"
    n_choose = 30
    env_num = 10 # Test with 10 instances
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Check if data file exists
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        return

    # Initialize Generator
    generator = CustomFLPGenerator(data_path, n_choose=n_choose, device=device)

    # Initialize GraphWorker
    print(f"Initializing {env_name.upper()} environment...")
    worker = GraphWorker(
        env_name=env_name,
        seed=1234,
        env_num=env_num,
        device=device,
        generator=generator, # Pass custom generator
        check_solution=False
    )
    
    # Reset Environment
    print("Resetting environment...")
    obs, infos = worker.reset()
    
    print(f"Observation (first instance):\n{obs[0][:500]}...")
    
    # Check loaded data in TensorDict
    td = worker._td
    print("\nEnvironment State:")
    print(f"Batch size: {td.batch_size}")
    print(f"Locs shape: {td['locs'].shape}")
    print(f"To Choose: {td['to_choose'][0]}")
    
    # Simple Random Rollout
    print("\nRunning random rollout...")
    total_reward = 0
    steps = 0
    
    while not worker.done:
        # Generate random actions
        # Action mask is in td['action_mask'] (True = invalid/masked out?)
        # RL4CO usually: mask True = invalid (set to -inf)
        # FLPEnv action_mask: ~chosen (1 = valid/not chosen, 0 = invalid/chosen) -> Wait.
        # Let's check FLPEnv._step: "action_mask = ~chosen". 
        # If chosen is 0 (not chosen), mask is 1 (valid).
        # BaseEnv action_projection checks if mask[action] is True.
        
        # So we should pick indices where mask is True.
        
        mask = td["action_mask"]
        actions = []
        for i in range(env_num):
            valid_indices = torch.nonzero(mask[i]).squeeze(-1)
            if len(valid_indices) > 0:
                # Pick random valid action
                act = valid_indices[torch.randint(0, len(valid_indices), (1,)).item()]
            else:
                act = 0 # Fallback
            actions.append(act.item())
        
        # Step
        obs, rewards, dones, infos = worker.step(actions)
        steps += 1
        
        if steps % 5 == 0:
            print(f"Step {steps}: Chosen {td['i'][0]} / {td['to_choose'][0]}")

    print("Rollout complete.")
    
    # Calculate final cost (Negative Reward)
    # RL4CO FLPEnv reward is 0 until done?
    # FLPEnv source: "The reward is calculated outside via get_reward for efficiency, so we set it to zero here"
    # Wait, GraphWorker uses base_env.get_reward() if done.
    
    rewards = torch.tensor(rewards)
    avg_reward = rewards.float().mean().item()
    print(f"Average Reward (Negative Cost): {avg_reward:.4f}")

if __name__ == "__main__":
    main()
