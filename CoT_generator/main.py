import json
import os
import sys
import dataclasses
import numpy as np
from typing import Dict, List, Any

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from CoT_generator.geometry_engine import GeometryEngine
from CoT_generator.generation_pipeline import CoTPipeline

# Import Agents (Mock or Real)
try:
    from examples.prompt_agent.vlm_agent import VLMAgent
    from examples.prompt_agent.llm_agent import LLMAgent
except ImportError:
    # ... Mock classes as defined before ...
    pass

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

def main(input_file: str, output_file: str):
    # Load Data
    with open(input_file, 'r') as f:
        data = json.load(f)
    
    # Init Agents
    # Note: Replace with real keys/urls
    api_key = os.environ.get("API_KEY", "mock") 
    base_url = os.environ.get("API_BASE", "mock")
    
    try:
        vlm_agent = VLMAgent(api_key=api_key, api_base_url=base_url, model_name="glm-4v")
        llm_agent = LLMAgent(api_key=api_key, api_base_url=base_url, model_name="glm-4")
    except:
        print("Using Mock Agents")
        class MockAgent:
            def generate(self, **kwargs): return "1. Yes\n2. No\n<Observation>Obs</Observation><Thought>Thk</Thought>"
        vlm_agent = MockAgent()
        llm_agent = MockAgent()

    pipeline = CoTPipeline(vlm_agent, llm_agent)
    all_processed = []

    for traj_idx, traj_data in enumerate(data):
        print(f"Processing Trajectory {traj_idx}...")
        
        # Init Geometry Engine for this trajectory
        node_coords = {int(k): tuple(v) for k, v in traj_data.get("node_coords", {}).items()}
        demands = {i: d for i, d in enumerate(traj_data.get("demand", []))} # Assuming list index = node id
        # Ensure 0 demand for depot if not present
        if 0 not in demands: demands[0] = 0.0
        
        geo_engine = GeometryEngine(node_coords, demands)
        vehicle_capacity = traj_data.get("capacity", 1.0) # Default or read

        # Process Steps
        current_idx = 0
        traj_steps = []
        
        # Align lists
        actions = traj_data.get("trajectory", [])
        obs_list = traj_data.get("obs_list", [])
        img_list = traj_data.get("image_list", [])
        cand_list = traj_data.get("candidates", [])
        load_list = traj_data.get("load_list", [])

        for i in range(len(actions)):
            step_data = {
                "step_idx": i,
                "trajectory": actions[i],
                "obs": obs_list[i] if i < len(obs_list) else "",
                "image": img_list[i] if i < len(img_list) else None,
                "candidates": cand_list[i] if i < len(cand_list) else [],
                "current_load": load_list[i] if i < len(load_list) else 0,
                "current_node_idx": current_idx
            }
            
            # Run Pipeline
            cot = pipeline.process_step(step_data, geo_engine, vehicle_capacity)
            
            if cot:
                step_data['cot'] = cot
                traj_steps.append(step_data)
            
            # Update State
            # Parse action to get next node
            # ... (Simple parsing logic as in pipeline) ...
            action_raw = str(actions[i]).replace("\\boxed{", "").replace("}", "").strip()
            # Map action back to ID
            # This logic assumes we can map back. If candidates provided, use index.
            # If not, we might need a mapping.
            # For robustness, we assume candidates[option_index] is the next node.
            # If standard dataset, 'trajectory' might be just the action string.
            # Here we need to update current_idx carefully.
            # Fallback: if we can't parse, keep 0.
            if step_data['candidates']:
                opt_idx = ord(action_raw) - ord('A')
                if 0 <= opt_idx < len(step_data['candidates']):
                    current_idx = step_data['candidates'][opt_idx]
            
        all_processed.extend(traj_steps)

    # Save
    with open(output_file, 'w') as f:
        json.dump(all_processed, f, indent=2, cls=EnhancedJSONEncoder)
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main("input.json", "output.json")
