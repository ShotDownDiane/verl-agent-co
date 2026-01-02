
import torch
import sys
import os
import pickle
import math
import re
import base64
import cv2
import numpy as np
from tensordict.tensordict import TensorDict

# Add the project root to path
sys.path.append("/root/autodl-tmp/verl-agent-co")

from agent_system.environments.env_package.rl4co.graph_env import GraphWorker
from rl4co.envs.common.utils import Generator

# Import Agents
from examples.prompt_agent.llm_agent import LLMAgent
from examples.prompt_agent.vlm_agent import VLMAgent

# Mock API Client for testing without real keys
class MockResponse:
    def __init__(self, content):
        self.choices = [type('Choice', (), {'message': type('Message', (), {'content': content})()})()]

class MockAgentClient:
    def __init__(self, mode="llm"):
        self.mode = mode
        self.chat = type('Chat', (), {'completions': self})()

    def create(self, model, messages, **kwargs):
        # Simulate an agent response
        # Always choose Option A for simplicity in testing
        return MockResponse("Based on the analysis, I select Option A.")

class MCLPGenerator(Generator):
    """
    MCLP Generator that supports multiple data sources:
    - random: Pre-generated random datasets (pickle)
    - twitch: Twitch social network datasets
    - rail: OR-Lib Rail datasets
    """
    def __init__(self, type="random", sub_type=None, data_root="/root/autodl-tmp/unsupervised-CO-ucom2/facility_location_and_max_cover/data", n_sets_to_choose=50, device="cpu", **kwargs):
        self.type = type
        self.sub_type = sub_type
        self.data_root = data_root
        self.n_sets_to_choose = n_sets_to_choose
        self.device = device
        
        self.data = self._load_data()
        self.num_samples = len(self.data)
        
        if self.num_samples > 0:
            # Analyze first sample to get dimensions and validate
            # Data format expected: list of (name/id, weights, membership_sets)
            _, weights_0, membership_0 = self.data[0]
            self.num_items = len(weights_0)
            self.num_sets = len(membership_0)
            print(f"Loaded {self.num_samples} samples from {type}/{sub_type}.")
            print(f"Sample 0 - Num Items: {self.num_items}, Num Sets: {self.num_sets}")
        else:
            print(f"Warning: No data loaded for {type}/{sub_type}.")
            self.num_items = 0
            self.num_sets = 0

    def _load_data(self):
        if self.type == "random":
            return self._load_random()
        elif self.type == "twitch":
            return self._load_twitch()
        elif self.type == "rail":
            return self._load_rail()
        else:
            raise ValueError(f"Unknown data type: {self.type}")

    def _load_random(self):
        # sub_type is num_sets, e.g., 500 or 1000. Default to 500.
        num_sets = self.sub_type if self.sub_type else "500"
        # We can support train/test selection via another param or just default to test
        # Let's try test first
        filename = f"max_covering_{num_sets}_test.pkl"
        path = os.path.join(self.data_root, filename)
        
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return []
            
        print(f"Loading random data from {path}...")
        with open(path, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    def _load_twitch(self):
        # Logic adapted from get_twitch_dataset in max_covering_data.py
        # sub_type could be a specific language, or we load all. 
        # The reference implementation loads ALL languages. 
        # If sub_type is specified, we can filter, otherwise load all.
        
        languages = ['DE', 'ENGB', 'ES', 'FR', 'PTBR', 'RU']
        if self.sub_type and self.sub_type.upper() in languages:
            languages = [self.sub_type.upper()]
            
        dataset = []
        for language in languages:
            edge_path = os.path.join(self.data_root, f'twitch/{language}/musae_{language}_edges.csv')
            target_path = os.path.join(self.data_root, f'twitch/{language}/musae_{language}_target.csv')
            
            if not os.path.exists(edge_path) or not os.path.exists(target_path):
                print(f"Warning: Twitch data for {language} not found at {edge_path}")
                continue
                
            print(f"Loading Twitch data for {language}...")
            
            # Read Edges
            edges = []
            node_ids = set()
            with open(edge_path) as f:
                for e in f.readlines():
                    e_str = e.strip().split(',')
                    if e_str[0] == 'from' and e_str[1] == 'to':
                        continue
                    n1, n2 = int(e_str[0]), int(e_str[1])
                    edges.append((n1, n2))
                    node_ids.add(n1)
                    node_ids.add(n2)
            
            id_map = {n: i for i, n in enumerate(node_ids)}
            num_nodes = len(node_ids)
            weights = [-1 for _ in range(num_nodes)]
            
            # Read Targets (Weights)
            with open(target_path) as f:
                for line in f.readlines():
                    line_str = line.strip().split(',')
                    if line_str[0] == 'id':
                        continue
                    # Format: id, days, mature, views, partner, new_id
                    # Reference uses: weights[id_map[int(line_str[5])]] = math.floor(math.log(int(line_str[3]) + 1))
                    # line_str[5] is new_id, line_str[3] is views
                    idx = int(line_str[5])
                    views = int(line_str[3])
                    if idx in id_map:
                         weights[id_map[idx]] = math.floor(math.log(views + 1))
            
            # Ensure all weights are set (some nodes might not be in target file?)
            # The reference asserts min(weights) >= 0.
            if any(w == -1 for w in weights):
                 print(f"Warning: Some weights missing for {language}, filling with 0")
                 weights = [max(0, w) for w in weights]

            # Build Sets (Adjacency List)
            sets = [[] for _ in range(num_nodes)]
            for n1, n2 in edges:
                if n1 in id_map and n2 in id_map:
                    sets[id_map[n1]].append(id_map[n2])
            
            dataset.append((language, weights, sets))
            
        return dataset

    def _load_rail(self):
        path = os.path.join(self.data_root, "orlib/rail.data")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return []
            
        print(f"Loading Rail data from {path}...")
        with open(path, "rb") as f:
            dataset = pickle.load(f)
        return dataset

    def _generate(self, batch_size) -> TensorDict:
        bs = batch_size[0] if isinstance(batch_size, torch.Size) else batch_size
        
        if self.num_samples == 0:
            raise ValueError("No data available to generate from.")

        if bs > self.num_samples:
            # If batch size is larger than samples, we must repeat or sample with replacement
            indices = torch.randint(0, self.num_samples, (bs,), device=self.device)
        else:
            # Deterministic slice for testing
            indices = torch.arange(bs, device=self.device) % self.num_samples
            
        # Process batch data
        batch_weights = []
        
        # We need to pad membership to the max set size in this batch
        max_set_size = 0
        selected_data = [self.data[i] for i in indices.cpu().numpy()]
        
        for _, weights, membership in selected_data:
            batch_weights.append(weights)
            # Find max size of any set in this sample
            sample_max = 0
            if membership:
                sample_max = max(len(s) for s in membership)
            if sample_max > max_set_size:
                max_set_size = sample_max
        
        # Ensure at least size 1 to avoid empty tensor issues
        max_set_size = max(max_set_size, 1)

        # Build tensors
        # Weights: (B, NumItems) - Assuming all samples in batch have same NumItems?
        # Actually, different samples (especially Twitch/Rail) might have different NumItems/NumSets.
        # If so, we can't batch them easily without padding EVERYTHING.
        # For 'random', NumItems/NumSets are usually consistent per file.
        # For 'twitch', they vary wildly.
        # RL4CO usually expects consistent batch shapes.
        # If sizes vary, we must pad weights and membership to the MAX in the batch.
        
        max_num_items = max(len(w) for w in batch_weights)
        max_num_sets = max(len(d[2]) for d in selected_data)
        
        weights_tensor = torch.zeros((bs, max_num_items), dtype=torch.float, device=self.device)
        membership_tensor = torch.zeros((bs, max_num_sets, max_set_size), dtype=torch.long, device=self.device)
        
        for i, (_, weights, membership) in enumerate(selected_data):
            # Fill weights
            w_len = len(weights)
            weights_tensor[i, :w_len] = torch.tensor(weights, dtype=torch.float, device=self.device)
            
            # Fill membership
            for j, s in enumerate(membership):
                if len(s) > 0:
                    # s is 0-based indices. MCPEnv expects 1-based indices (0 is padding).
                    items = torch.tensor(s, dtype=torch.long, device=self.device) + 1
                    # Ensure items don't exceed max_num_items? They shouldn't if data is consistent.
                    membership_tensor[i, j, :len(items)] = items
                    
        return TensorDict(
            {
                "membership": membership_tensor.float(), # Kept as float to match prior implementation/RL4CO conventions often using float for obs
                "weights": weights_tensor,
                "n_sets_to_choose": torch.full((bs, 1), self.n_sets_to_choose, dtype=torch.float, device=self.device),
                
                # Helper fields
                "chosen": torch.zeros(bs, max_num_sets, dtype=torch.bool, device=self.device),
                "i": torch.zeros(bs, dtype=torch.long, device=self.device),
            },
            batch_size=bs,
            device=self.device
        )

def parse_action_from_response(response: str, num_options: int) -> int:
    """
    Parse the LLM/VLM response to find the selected option (A, B, C...).
    Returns the index of the selected option (0 for A, 1 for B...).
    Defaults to 0 if parsing fails.
    """
    # Look for patterns like "Option A", "Option B", etc.
    match = re.search(r"Option\s+([A-Z])", response, re.IGNORECASE)
    if match:
        letter = match.group(1).upper()
        index = ord(letter) - ord('A')
        if 0 <= index < num_options:
            return index
    
    # Fallback: look for single letter A-Z near the end? Too risky.
    # Just return 0 (safe default)
    print(f"Warning: Could not parse action from response: '{response}'. Defaulting to Option A.")
    return 0

def ensure_image_exists(path, size=512):
    """Ensure a dummy image exists if the real one wasn't generated."""
    if not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Create a black image with text "Missing Image"
        img = np.zeros((size, size, 3), dtype=np.uint8)
        cv2.putText(img, "Missing Image", (50, 256), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(path, img)

def main():
    # --- Test Configuration ---
    agent_type = "vlm"  # Options: "llm", "vlm"
    use_mock_api = True # Set to True to use Mock client (no real API call)

    # Options:
    # type="random", sub_type="500"
    # type="twitch", sub_type="DE"
    # type="rail"
    
    test_type = "twitch"
    test_sub_type = "DE" # Test with German Twitch dataset
    
    env_name = "mcp" # Maps to MCLP/MCP
    n_choose = 50
    env_num = 1 # Test with 1 instance since Twitch graphs vary in size
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- Agent Initialization ---
    print(f"Initializing {agent_type.upper()} Agent...")
    
    mock_client = MockAgentClient() if use_mock_api else None
    
    if agent_type == "llm":
        agent = LLMAgent(
            api_client=mock_client,
            api_base_url="https://api.openai.com/v1" if not mock_client else None,
            api_key="sk-..." if not mock_client else None,
            model_name="gpt-4"
        )
    else:
        agent = VLMAgent(
            api_client=mock_client,
            api_base_url="https://api.openai.com/v1" if not mock_client else None,
            api_key="sk-..." if not mock_client else None,
            model_name="gpt-4-vision-preview"
        )

    # Initialize Generator
    generator = MCLPGenerator(type=test_type, sub_type=test_sub_type, n_sets_to_choose=n_choose, device=device)

    if generator.num_samples == 0:
        print("Generator initialization failed (no data). Exiting.")
        return

    # Initialize GraphWorker
    print(f"Initializing {env_name.upper()} environment with {test_type}/{test_sub_type}...")
    
    env_kwargs = {"generator": generator}
    
    worker = GraphWorker(
        env_name=env_name,
        seed=1234,
        env_num=env_num,
        device=device,
        return_topk_options=10,
        env_kwargs=env_kwargs
    )
    
    # Reset Environment
    print("Resetting environment...")
    obs_list, infos = worker.reset()
    
    td = worker._td
    # print(f"\nObservation (first instance):\n{obs_list[0][:200] if obs_list is not None else 'None'}...")
    
    print("\nEnvironment State:")
    print(f"Batch size: {td.batch_size}")
    print(f"Membership shape: {td['membership'].shape}")
    print(f"Weights shape: {td['weights'].shape}")
    
    # --- Interaction Loop ---
    print("\nStarting interaction loop...")
    total_reward = 0
    steps = 0
    
    while not worker.done:
        print(f"\n--- Step {steps + 1} ---")
        
        # Get observation for the first environment
        obs_text = obs_list[0]
        
        # Get available top-k actions (indices in the original node list)
        if "topk_acts" in worker._td.keys():
            topk_acts = worker._td["topk_acts"][0] # Shape (K,)
        else:
            print("Warning: topk_acts not found in td. Using valid actions from mask.")
            mask = worker._td["action_mask"][0]
            topk_acts = torch.nonzero(mask).squeeze(-1)[:10] # Fallback to first 10 valid
            
        num_options = len(topk_acts)
        
        # Determine image path for VLM
        img_path = f"./debug_images/env0_step{steps}.jpg"
        
        # Generate Agent Response
        if agent_type == "vlm":
            # Ensure image exists (since GraphWorker might skip some steps or we want robust testing)
            ensure_image_exists(img_path)
            
            print(f"Input: Text ({len(obs_text)} chars) + Image ({img_path})")
            
            # Helper to load image as base64
            with open(img_path, "rb") as image_file:
                encoded_string = cv2.imencode('.jpg', cv2.imread(img_path))[1].tobytes()
                base64_image = base64.b64encode(encoded_string).decode('utf-8')
                
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": obs_text},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }
            ]
            response = agent.generate(text=messages)
            
        else: # LLM
            print(f"Input: Text ({len(obs_text)} chars)")
            response = agent.generate(text=obs_text)
            
        print(f"Agent Response: {response}")
        
        # Parse Action
        selected_idx = parse_action_from_response(response, num_options)
        
        # Map selected index (0..K-1) to actual node index
        if selected_idx < len(topk_acts):
            action_node_idx = topk_acts[selected_idx].item()
        else:
            print(f"Warning: Selected index {selected_idx} out of bounds for {len(topk_acts)} options. Using option 0.")
            action_node_idx = topk_acts[0].item()

        print(f"Selected Option: {chr(65+selected_idx)} -> Node Index: {action_node_idx}")
        
        # Step Environment
        # GraphWorker expects a list of actions (one per env)
        actions = [action_node_idx]
        obs_list, rewards, dones, infos = worker.step(actions)
        steps += 1
        
        if steps % 10 == 0:
            print(f"Step {steps}: Chosen {td['i'][0]} / {n_choose}")

    print("\nRollout complete.")
    
    rewards = torch.tensor(rewards)
    avg_reward = rewards.float().mean().item()
    print(f"Average Reward (Total Covered Weight): {avg_reward:.4f}")

if __name__ == "__main__":
    main()
