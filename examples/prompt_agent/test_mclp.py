import os
import sys
import pickle
import torch
import numpy as np
from types import SimpleNamespace

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from examples.prompt_agent.llm_agent import LLMAgent
from examples.prompt_agent.vlm_agent import VLMAgent
from functools import partial
from agent_system.environments.env_package.rl4co.graph_env import GraphWorker, GraphEnvs
from agent_system.environments.env_manager import RL4COEnvironmentManager
from agent_system.environments.env_package.rl4co.projection import co_projection_selected

class LoadedDataGenerator:
    def __init__(self, data_list, device="cpu"):
        self.data_list = data_list
        self.idx = 0
        self.device = device
        self.min_loc = 0.0
        self.max_loc = 1.0
        
        self.num_facility = 0
        self.num_facilities_to_select = 0
        self.num_demand = 0
        self.num_terminals = 0
        
        if len(data_list) > 0:
            first_item = data_list[0]
            if 'td' in first_item:
                td = first_item['td']
                if 'loc' in td.keys():
                    self.num_loc = td['loc'].shape[-2]
                elif 'locs' in td.keys():
                    self.num_loc = td['locs'].shape[-2]
                else:
                    self.num_loc = 0
                
                if 'num_facility' in td.keys():
                    val = td['num_facility']
                    if hasattr(val, 'item'):
                        self.num_facility = val.item() if val.numel() == 1 else val[0].item()
                    else:
                        self.num_facility = val
                else:
                     if 'facility_locs' in td.keys():
                         self.num_facility = td['facility_locs'].shape[-2]
                     else:
                         self.num_facility = 0
                
                if 'num_facilities_to_select' in td.keys():
                    val = td['num_facilities_to_select']
                    if hasattr(val, 'item'):
                        self.num_facilities_to_select = val.item() if val.numel() == 1 else val[0].item()
                    else:
                        self.num_facilities_to_select = val
                else:
                    self.num_facilities_to_select = 0

                if 'num_demand' in td.keys():
                     val = td['num_demand']
                     if hasattr(val, 'item'):
                         self.num_demand = val.item() if val.numel() == 1 else val[0].item()
                     else:
                         self.num_demand = val
                else:
                     if 'demand_locs' in td.keys():
                         self.num_demand = td['demand_locs'].shape[-2]
                     else:
                         self.num_demand = 0
                
                if 'num_terminals' in td.keys():
                    val = td['num_terminals']
                    if hasattr(val, 'item'):
                        self.num_terminals = val.item() if val.numel() == 1 else val[0].item()
                    else:
                        self.num_terminals = val
                else:
                    if 'terminals' in td.keys():
                         t = td['terminals']
                         if t.dim() >= 2:
                             self.num_terminals = t.shape[-1]
                         else:
                             self.num_terminals = 0
                    else:
                        self.num_terminals = 0

            else:
                self.num_loc = 0
        else:
            self.num_loc = 0

        self.capacity = 1.0 
        self.vehicle_capacity = 1.0 
        self.min_demand = 0.0
        self.max_demand = 1.0

    def __call__(self, batch_size):
        if isinstance(batch_size, torch.Size):
            batch_size = batch_size[0] if len(batch_size) > 0 else 1
        elif isinstance(batch_size, list):
            batch_size = batch_size[0] if len(batch_size) > 0 else 1
            
        collected = []
        current_count = 0
        
        while current_count < batch_size:
            data_item = self.data_list[self.idx % len(self.data_list)]
            td = data_item['td'].clone()
            
            if self.device != "cpu":
                td = td.to(self.device)
            
            b = td.batch_size[0]
            collected.append(td)
            current_count += b
            self.idx += 1
            
        if len(collected) > 1:
            loc_key = None
            for k in collected[0].keys():
                if k in ['loc', 'locs', 'coords', 'coordinates']:
                    loc_key = k
                    break
            
            if loc_key is None:
                for k, v in collected[0].items():
                    if isinstance(v, torch.Tensor) and v.dim() == 3 and v.shape[2] == 2:
                        loc_key = k
                        break
            
            if loc_key:
                max_loc = max([td[loc_key].shape[1] for td in collected])
                
                for i in range(len(collected)):
                    td = collected[i]
                    curr_loc = td[loc_key].shape[1]
                    
                    if curr_loc < max_loc:
                        pad_len = max_loc - curr_loc
                        for key in td.keys():
                            val = td[key]
                            if isinstance(val, torch.Tensor) and val.dim() > 1 and val.shape[1] == curr_loc:
                                if val.dim() == 3:
                                    val_padded = torch.nn.functional.pad(val, (0, 0, 0, pad_len), value=0)
                                    td[key] = val_padded
                                elif val.dim() == 2:
                                    val_padded = torch.nn.functional.pad(val, (0, pad_len), value=0)
                                    td[key] = val_padded
                        collected[i] = td
            
        full_td = torch.cat(collected, dim=0)
        return full_td[:batch_size]

def load_data():
    base_path = "/root/autodl-tmp/rl4co-urban"
    if not os.path.exists(base_path):
        print(f"Warning: Data path {base_path} does not exist. Using dummy data.")
        return None, {'mclp': []}

    try:
        with open(os.path.join(base_path, "results.pkl"), "rb") as f:
            graph_data = pickle.load(f)
        with open(os.path.join(base_path, "routing_results.pkl"), "rb") as f:
            routing_data = pickle.load(f)
        return graph_data, routing_data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, {'mclp': []}

def run_agent_loop(env_manager, agent, solution_tour=None, env_name="mclp", instance_idx=0, heuristic_objs=None):
    # 1. Reset Environment (Batch)
    observations, infos = env_manager.reset()
    
    trajectory = []
    obs_list = []
    image_list = []
    candidates_list = []
    
    metrics = {}

    os.makedirs(f"debug_images/{env_name}", exist_ok=True)
    
    # 2. Loop
    i = 0
    while True:
        # Access batch data
        prompts = observations['text']
        system_template = observations['system_template']
        images = observations.get('image', None)
        
        # Log first item in batch
        print(f"\n--- Step {i+1} (Batch Size: {len(prompts)}) ---")
        
        obs_list.append(prompts[0])
        
        # Save step image if available
        current_img_path = ""
        if images and len(images) > 0 and images[0] is not None:
            img_data = images[0]
            if isinstance(img_data, np.ndarray):
                from PIL import Image as PILImage
                try:
                    img = PILImage.fromarray(img_data.astype('uint8'))
                    step_img_path = f"debug_images/{env_name}/inst_{instance_idx}_step_{i}.png"
                    img.save(step_img_path)
                    current_img_path = step_img_path
                except Exception as e:
                    print(f"Error saving numpy image: {e}")
            elif isinstance(img_data, str) and os.path.exists(img_data):
                current_img_path = img_data
        
        image_list.append(current_img_path)

        # 3. Agent Inference (Batch)
        # Prepare images for batch (convert numpy to PIL if needed)
        batch_images = None
        if images is not None and len(images) > 0:
            batch_images = []
            from PIL import Image as PILImage
            for img_data in images:
                if isinstance(img_data, np.ndarray):
                    batch_images.append(PILImage.fromarray(img_data.astype('uint8')))
                elif isinstance(img_data, str) and os.path.exists(img_data):
                    batch_images.append(img_data)
                else:
                    batch_images.append(None)

        actions_str_raw = agent.batch_generate(
            system_prompts=system_template,
            texts=prompts,
            images=batch_images,
            max_tokens=256, 
            temperature=0.7
        )
        actions_str = [resp.strip() for resp in actions_str_raw]
            
        print(f"Action (first in batch): {actions_str[0]}")
        trajectory.append(actions_str)

        # 4. Environment Step (Batch)
        next_observations, rewards, dones, infos = env_manager.step(actions_str)
        
        if np.all(dones):
            print("All environments done.")
            
            if heuristic_objs is not None:
                reward_scale = 0.1
                
                batch_gaps = []
                batch_agent_objs = []

                print("\n" + "-"*50)
                print(f"Evaluation Results (Batch starting at Instance {instance_idx}):")
                print(f"{'Idx':<5} | {'Heuristic':<10} | {'Agent':<10} | {'Gap (%)':<10}")
                print("-" * 45)

                for k, info in enumerate(infos):
                    h_obj = heuristic_objs[k] if k < len(heuristic_objs) else None
                    if h_obj is None:
                        continue
                        
                    agent_reward = info.get('sum_env_reward', 0.0)
                    
                    # For MCLP, reward is positive (coverage)
                    agent_obj = agent_reward / reward_scale
                    
                    # Gap for maximization: (Heuristic - Agent) / Heuristic
                    if h_obj != 0:
                        gap = (h_obj - agent_obj) / h_obj * 100
                    else:
                        gap = 0.0 if agent_obj >= 0 else float('inf')

                    batch_gaps.append(gap)
                    batch_agent_objs.append(agent_obj)
                    
                    print(f"{instance_idx+k:<5} | {h_obj:<10.4f} | {agent_obj:<10.4f} | {gap:<10.2f}")

                print("-" * 50 + "\n")
                
                if batch_gaps:
                    avg_batch_gap = sum(batch_gaps) / len(batch_gaps)
                    metrics['gap'] = avg_batch_gap 
                    metrics['all_gaps'] = batch_gaps
                    metrics['agent_objs'] = batch_agent_objs
                    metrics['heuristic_objs'] = heuristic_objs

            break
            
        observations = next_observations
        i += 1
        
        if i > 100:
            print("Max steps reached.")
            break

    return metrics

def main():
    graph_data, routing_data = load_data()
    
    if 'mclp' not in graph_data:
        print("No MCLP data found.")
        return

    api_base_url = "http://localhost:8000/v1"
    api_key = "token-abc123456"
    if not api_key or "sk-" not in api_key:
        print("Please provide a valid API key.")
        
    agent = VLMAgent(
        api_key=api_key,
        api_base_url=api_base_url,
        model_name="vlm"
    )

    print("\n" + "="*50)
    print("1. Single Worker Execution (MCLP) with RL4COEnvironmentManager")
    print("="*50)

    mclp_data = graph_data['mclp']
    n = len(mclp_data)
    if n == 0:
        print("No MCLP instances.")
        return

    gaps = []

    # Run 5 instances in one batch
    batch_size = 5
    num_to_run = min(5, n)
    for i in range(0, num_to_run, batch_size):
        current_bs = min(batch_size, num_to_run - i)
        batch_data = mclp_data[i : i + current_bs]
        
        # Create a list of generators, one for each environment/worker
        generators = [LoadedDataGenerator([item]) for item in batch_data]
        # Assume all instances have same num_loc or we take from first
        num_loc = generators[0].num_loc

        # Extract heuristic solution objectives
        heuristic_objs = []
        for item in batch_data:
            h_obj = None
            if 'objs' in item:
                objs = item['objs']
                if isinstance(objs, list) and len(objs) > 0:
                    h_obj = objs[0]
                elif hasattr(objs, 'item'):
                    h_obj = objs.item() if objs.numel() == 1 else objs[0].item()
            elif 'obj' in item:
                h_obj = item['obj']
            heuristic_objs.append(h_obj)
        
        config = SimpleNamespace(
            env=SimpleNamespace(
                env_name="mclp",
                rl4co=SimpleNamespace(
                    use_format_reward=True,
                    format_reward_weight=0.1,
                    format_penalty=-1.0,
                    env_reward_scale=0.1
                )
            ),
            data=SimpleNamespace(
                return_topk_options=10, 
                image_max_pixels=448,
                train_batch_size=current_bs, 
                val_batch_size=current_bs
            )
        )

        worker = GraphEnvs(
            env_name="mclp",
            seed=42,
            env_num=current_bs,
            group_n=1,
            device="cpu",
            resources_per_worker={},
            return_topk_options=10,
            env_kwargs={
                "generator": generators,
                "image_obs": "path",
                "generator_params": {"num_loc": num_loc},
                "synchronous": False
            }
        )
        
        projection_f = partial(co_projection_selected, env_name="mclp")
        
        env_manager = RL4COEnvironmentManager(worker, projection_f, config)
            
        metrics = run_agent_loop(env_manager, agent, solution_tour=None, env_name="mclp", instance_idx=i, heuristic_objs=heuristic_objs)
        
        if 'all_gaps' in metrics:
            gaps.extend(metrics['all_gaps'])
        
        print(f"Test complete for batch starting at {i}")

    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        print("\n" + "="*50)
        print(f"Final Average Gap over {len(gaps)} instances: {avg_gap:.2f}%")
        print("="*50 + "\n")

if __name__ == "__main__":
    main()
