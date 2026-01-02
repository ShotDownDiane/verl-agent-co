
import torch
import sys
import os
import math
import re
import numpy as np
import base64
import json
import datetime
from tensordict.tensordict import TensorDict
from time import sleep

# Add the project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from agent_system.environments.env_package.rl4co.graph_env import GraphWorker
from agent_system.environments.env_package.rl4co.projection import co_projection_selected
from agent_system.environments.prompts.rl4co import RL4CO_FLP_TEMPLATE, RL4CO_FLP_TEMPLATE_COT
from rl4co.envs.common.utils import Generator
from rl4co.utils.ops import get_distance_matrix

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

class FLPGenerator(Generator):
    """
    FLP Generator that supports multiple data sources.
    (Same as before)
    """
    def __init__(self, type="random", sub_type=None, data_root="/root/autodl-tmp/unsupervised-CO-ucom2/facility_location_and_max_cover/data", n_choose=30, device="cpu", **kwargs):
        self.type = type
        self.sub_type = sub_type
        self.data_root = data_root
        self.n_choose = n_choose
        self.device = device
        
        self.data = self._load_data()
        self.num_samples = len(self.data)
        if self.num_samples > 0:
            self.num_loc = self.data.shape[1]
            self.distances = get_distance_matrix(self.data)
            self.max_dist = math.sqrt(2) * 2.0 
            print(f"Loaded {self.num_samples} samples with {self.num_loc} locations from {type}/{sub_type}.")
        else:
            print(f"Warning: No data loaded for {type}/{sub_type}.")
            self.num_loc = 0

    def _load_data(self):
        if self.type == "random":
            return self._load_random()
        elif self.type == "starbucks":
            return self._load_starbucks()
        elif self.type == "mcd":
            return self._load_mcd()
        elif self.type == "subway":
            return self._load_subway()
        else:
            raise ValueError(f"Unknown data type: {self.type}")

    def _load_random(self):
        num_data = self.sub_type if self.sub_type else "500"
        path = os.path.join(self.data_root, f"facility_location_rand{num_data}_test.pt")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return torch.empty(0)
        return torch.load(path, map_location=self.device)

    def _load_starbucks(self):
        city = self.sub_type if self.sub_type else "london"
        path = os.path.join(self.data_root, "starbucks", f"{city}.csv")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return torch.empty(0)

        locations = []
        with open(path, encoding="utf-8-sig") as f:
            for l in f.readlines():
                l_str = l.strip().split(",")
                if l_str[0] == "latitude" and l_str[1] == "longitude":
                    continue
                n1, n2 = (float(l_str[0]) / 365 * 400, float(l_str[1]) / 365 * 400)
                locations.append((n1, n2))
        
        locations = torch.tensor(locations, device=self.device)
        return self._normalize(locations).unsqueeze(0)

    def _load_mcd(self):
        state = self.sub_type if self.sub_type else "NY"
        path = os.path.join(self.data_root, "locations/mcd", f"mcd_{state}_data.pt")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return torch.empty(0)
        locations = torch.load(path, map_location=self.device)
        locations = torch.tensor(locations, device=self.device)
        return self._normalize(locations).unsqueeze(0)

    def _load_subway(self):
        state = self.sub_type if self.sub_type else "NY"
        path = os.path.join(self.data_root, "locations/subway_states", f"subway_{state}_data.pt")
        if not os.path.exists(path):
            print(f"File not found: {path}")
            return torch.empty(0)
        locations = torch.load(path, map_location=self.device)
        locations = torch.tensor(locations, device=self.device)
        return self._normalize(locations).unsqueeze(0)

    def _normalize(self, locations):
        locations_x = locations[:, 0]
        locations_y = locations[:, 1]
        xmin, xmax = locations_x.min(), locations_x.max()
        ymin, ymax = locations_y.min(), locations_y.max()
        
        if xmax - xmin > 1e-6:
            locations[:, 0] = (locations_x - xmin) / (xmax - xmin)
        else:
            locations[:, 0] = 0.5
            
        if ymax - ymin > 1e-6:
            locations[:, 1] = (locations_y - ymin) / (ymax - ymin)
        else:
            locations[:, 1] = 0.5
            
        return locations

    def _generate(self, batch_size) -> TensorDict:
        bs = batch_size[0] if isinstance(batch_size, torch.Size) else batch_size
        
        if self.num_samples == 0:
            raise ValueError("No data available to generate from.")

        if bs > self.num_samples:
            if self.num_samples == 1:
                indices = torch.zeros(bs, dtype=torch.long, device=self.device)
            else:
                indices = torch.randint(0, self.num_samples, (bs,), device=self.device)
        else:
            indices = torch.arange(bs, device=self.device) % self.num_samples
            
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
                "i": torch.zeros(bs, dtype=torch.long, device=self.device),
            },
            batch_size=bs,
            device=self.device
        )



def main():
    # --- Test Configuration ---
    agent_type = "llm"  # Options: "llm", "vlm"
    use_mock_api = False # Set to True to use Mock client (no real API call)
    
    api_key = "sk-saxqqtlyqrpconxlgcslqhrgvhwnfmuhnimiyzfvpcxqgmkh"
    experimental_name = "VLM" #LLM is Qwen3-235B-A22B-Instruct-2507
                              #VLM is Qwen3-VL-235B-A22B-Instruct
    CoT_states = "cot"
    test_type = "random"
    test_sub_type = "500"
    
    env_name = "flp"
    n_choose = 30 # Short rollout for testing
    env_num = 10   # Number of parallel environments
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Output directory for logs
    output_dir = "/root/autodl-tmp/verl-agent-co/agent_system/environments/env_package/rl4co/flp_test_results/{}_{}_{}".format(experimental_name, CoT_states, test_sub_type)
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Agent Initialization ---
    print(f"Initializing {agent_type.upper()} Agent...")
    
    mock_client = MockAgentClient() if use_mock_api else None
    
    if agent_type == "llm":
        agent = LLMAgent(
            api_client=mock_client,
            api_base_url="https://api.siliconflow.cn/v1" if not mock_client else None,
            api_key=api_key if not mock_client else None,
            model_name="Qwen/Qwen3-235B-A22B-Instruct-2507"
        )
    else:
        agent = VLMAgent(
            api_client=mock_client,
            api_base_url="https://api.siliconflow.cn/v1" if not mock_client else None,
            api_key=api_key if not mock_client else None,
            model_name="Qwen/Qwen3-VL-235B-A22B-Instruct"
        )

    # --- Environment Initialization ---
    generator = FLPGenerator(type=test_type, sub_type=test_sub_type, n_choose=n_choose, device=device)
    print(f"Initializing {env_name.upper()} environment with {test_type}/{test_sub_type}...")
    
    env_kwargs = {"generator": generator, "synchronous": False}
    worker = GraphWorker(
        env_name=env_name,
        seed=1234,
        env_num=env_num,
        device=device,
        return_topk_options=10,
        image_obs=(agent_type == "vlm"), # Enable image observation if VLM
        env_kwargs=env_kwargs
    )
    
    print(f"Resetting {env_num} environments...")
    obs_list, infos = worker.reset()
    
    # Initialize logs per environment
    env_logs = {i: [] for i in range(env_num)}
    
    # --- Interaction Loop ---
    steps = 0
    
    while not worker.done:
        print(f"\n--- Step {steps + 1} ---")
        
        current_step_actions = []
        
        # Get topk actions for all envs: (Batch, K)

        # if "topk_acts" in worker._td.keys():
        #     batch_topk_acts = worker._td["topk_acts"]
        # else:
        #     # Fallback if topk_acts missing
        #     mask = worker._td["action_mask"]
        #     batch_topk_acts = []
        #     for i in range(env_num):
        #          batch_topk_acts.append(torch.nonzero(mask[i]).squeeze(-1)[:10])
        #     batch_topk_acts = torch.stack(batch_topk_acts)

        # Iterate over each environment
        for env_idx in range(env_num):
            raw_obs = obs_list[env_idx]
            
            # Handle obs type (str or dict)
            if isinstance(raw_obs, dict):
                obs_text = raw_obs["text"]
                obs_image = raw_obs["image"]
            else:
                obs_text = raw_obs
                obs_image = None
            if CoT_states == "cot":
                obs_text_formatted = RL4CO_FLP_TEMPLATE_COT.format(
                    text_obs=obs_text
                )
            else:
                obs_text_formatted = RL4CO_FLP_TEMPLATE.format(
                    text_obs=obs_text
                )
            
            topk_acts = worker._td["topk_acts"][env_idx]
            num_options = len(topk_acts)
            
            # Generate Agent Response
            try:
                if agent_type == "vlm":
                    if obs_image:
                        # Pass text and image separately; VLMAgent handles formatting
                        response = agent.generate(text=obs_text_formatted, image=obs_image)
                    else:
                        # Fallback if no image provided even if VLM agent (should not happen with correct setup)
                        print("Warning: VLM Agent expects image but none provided in observation.")
                        response = agent.generate(text=obs_text_formatted)
                    
                else: # LLM
                    response = agent.generate(text=obs_text_formatted)
            except Exception as e:
                print(f"Error rate limits")
                response = "Therefore, the final answer is: \boxed{A}"
                # sleep 30s
                sleep(30)
                
            
            # Parse Action
            selected_idx, valids = co_projection_selected([response], env_name=env_name)
            selected_idx_val = selected_idx[0]
            
            # Map selected index (0..K-1) to actual node index
            if selected_idx_val < len(topk_acts):
                action_node_idx = topk_acts[selected_idx_val].item()
            else:
                action_node_idx = topk_acts[0].item() # Fallback
            
            if env_idx == 0: # Print only first env to avoid clutter
                print(f"Env 0 Observation: {obs_text_formatted}")
                print(f"Env 0 Response: {response[:50]}... -> Option {chr(65+selected_idx_val)}")

            current_step_actions.append(action_node_idx)
            
            # Log data
            log_entry = {
                "step": steps,
                "observation": obs_text_formatted,
                "response": response,
                "selected_option": chr(65+selected_idx_val),
                "selected_option_idx": int(selected_idx_val),
                "action_node_idx": int(action_node_idx),
                "timestamp": datetime.datetime.now().isoformat()
            }
            env_logs[env_idx].append(log_entry)

        # Step Environment with all actions
        obs_list, rewards, dones, infos = worker.step(current_step_actions)
        
        # Update logs with rewards
        for env_idx in range(env_num):
            # The log we just added is the last one in env_logs[env_idx]
            env_logs[env_idx][-1]["reward"] = float(rewards[env_idx])
            env_logs[env_idx][-1]["done"] = bool(dones[env_idx])

        steps += 1
        
    print("\nRollout complete.")
    rewards = torch.tensor(rewards)
    avg_reward = rewards.float().mean().item()
    print(f"Average Reward: {avg_reward:.4f}")
    
    # Save logs to separate JSON files
    print(f"Saving logs to {output_dir}...")
    for env_idx, logs in env_logs.items():
        output_file = os.path.join(output_dir, f"env_{env_idx}_results.json")
        with open(output_file, "w") as f:
            json.dump(logs, f, indent=2)
    print("Done.")
if __name__ == "__main__":
    main()
