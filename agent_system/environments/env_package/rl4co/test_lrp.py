import sys
import os
import torch
import numpy as np
import re
from functools import partial
from PIL import Image
import swanlab
import time

# Add the project root to path so we can import the new modules
sys.path.append("/root/autodl-tmp/verl-agent-co")

from agent_system.environments.env_package.rl4co.route_envs import RouteWorker
from rl4co.envs.routing.lrp.generator import LRPBenchmarkGenerator
from rl4co.envs.routing.lrp.baselines.utils import evaluate_lrp
from agent_system.environments.env_package.rl4co.projection import co_projection_selected as co_projection
from agent_system.environments.prompts import *
from examples.prompt_agent.llm_agent import LLMAgent
from examples.prompt_agent.vlm_agent import VLMAgent

class MockAgent:
    """A mock agent that always picks Option A for testing purposes."""
    def batch_generate(self, system_prompts, texts, images=None, **kwargs):
        # Always pick Option A
        return ["<OBS>xxxx</OBS><Thought> Picking Option A to test the environment loop. </Thought>\n<Decision> \\boxed{A} </Decision>" for _ in range(len(texts))]

class WorkerManager:
    SYSTEM_TEMPLATEs ={
        "tsp": RL4CO_TSP_SYSTEM_TEMPLATE,
        "cvrp": RL4CO_CVRP_SYSTEM_TEMPLATE,
        "flp": RL4CO_FLP_SYSTEM_TEMPLATE,
        "mclp": RL4CO_MCLP_SYSTEM_TEMPLATE,
        "stp": RL4CO_STP_SYSTEM_TEMPLATE,
        "tdtsp": RL4CO_TDTSP_SYSTEM_TEMPLATE,
        "tdtsp_tw": RL4CO_TDTSP_TW_SYSTEM_TEMPLATE,
        "tdvrp": RL4CO_TDVRP_SYSTEM_TEMPLATE,
        "lrp": RL4CO_LRP_SYSTEM_TEMPLATE,
    } 
    USER_TEMPLATEs ={
        "tsp": RL4CO_TSP_USER_TEMPLATE,
        "cvrp": RL4CO_CVRP_USER_TEMPLATE,
        "flp": RL4CO_FLP_USER_TEMPLATE,
        "mclp": RL4CO_MCLP_USER_TEMPLATE,
        "stp": RL4CO_STP_USER_TEMPLATE,
        "tdtsp": RL4CO_TDTSP_USER_TEMPLATE,
        "tdtsp_tw": RL4CO_TDTSP_TW_USER_TEMPLATE,
        "tdvrp": RL4CO_TDVRP_USER_TEMPLATE,
        "lrp": RL4CO_LRP_USER_TEMPLATE,
    }
    def __init__(self, env_name):
        self.env_name = env_name
        self.system_template = self.SYSTEM_TEMPLATEs[env_name]
        self.user_template = self.USER_TEMPLATEs[env_name]
        self.action_projection = partial(co_projection, env_name=env_name)

    def process_text_obs(self, next_obs):
        postprocess_text_obs = []
        obs_str = [obs['obs'] for obs in next_obs]
        candidates_str = [obs['candidates'] for obs in next_obs]
        for i in range(len(next_obs)):
            obs = self.user_template.format(obs_text=obs_str[i], candidates=candidates_str[i])
            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def process_action(self, action_text):
        parsed_actions, valids = self.action_projection(action_text)
        return parsed_actions, valids
    
    def process_image(self, img) -> np.ndarray:
        # if isinstance(img, np.ndarray):
        #     return img
        # # If it's None (e.g. not image mode), return dummy
        # if img is None:
        #     return np.zeros((100, 100, 3), dtype=np.uint8)
            
        # assert isinstance(img, str), f"img must be a path string or a numpy array, got {type(img)}"
        
        # image = Image.open(img)
        # if image.mode != 'RGB':
        #     image = image.convert('RGB')        
        # return np.array(image)
        return img
    
    def process_step(self, next_obs):
        text_obs = self.process_text_obs(next_obs)
        # Handle case where image might be missing
        img_obs = [self.process_image(obs.get('image', None)) for obs in next_obs]
        next_observations = {
            "texts": text_obs, 
            "images": img_obs,
            "system_prompts": [self.system_template for _ in range(len(text_obs))]
        }
        return next_observations

def classify_benchmarks(all_files):
    """Classify benchmark files by size and benchmark set."""
    instances_by_category = {20: [], 50: [], 100: [], 200: []}
    
    print("Collecting and categorizing instances...")
    
    # Simple heuristic to extract benchmark name from path
    # Expected path: .../benchmark_instances/BENCHMARK_NAME/FILE_NAME
    
    for f_path in all_files:
        try:
            # Create a temporary generator to read num_loc without loading full env if possible,
            # or just load it. LRPBenchmarkGenerator reads file on init.
            gen = LRPBenchmarkGenerator(path=f_path)
            num_cust = gen.num_loc # This is num_customers usually
            
            # Determine benchmark name from parent directory
            parent_dir = os.path.basename(os.path.dirname(f_path))
            benchmark_name = parent_dir
            
            # Categorize
            category = None
            if num_cust <= 20:
                category = 20
            elif num_cust <= 50:
                category = 50
            elif num_cust <= 100:
                category = 100
            elif num_cust <= 200:
                category = 200
            else:
                print(f"Skipping {os.path.basename(f_path)} (Size: {num_cust} > 200)")
                continue
            
            instances_by_category[category].append({
                'benchmark': benchmark_name,
                'name': os.path.basename(f_path),
                'path': f_path,
                'num_cust': num_cust,
                'generator': gen
            })
            
        except Exception as e:
            print(f"Error processing {f_path}: {e}")
            
    return instances_by_category

def test_lrp_env(env_name="lrp"):
    print(f"\n{'='*20} Testing {env_name.upper()} {'='*20}")
    
    # 1. Get Benchmark Files
    bench_discover_gen = LRPBenchmarkGenerator()
    all_files = bench_discover_gen.files
    
    if not all_files:
        print("No benchmark files found. Exiting.")
        return

    # 2. Classify Instances
    instances_by_category = classify_benchmarks(all_files)
    
    # Flatten for total count
    all_instances = [i for cat in instances_by_category.values() for i in cat]
    benchmarks = list(set(i['benchmark'] for i in all_instances))
    
    # 3. Initialize SwanLab
    try:
        swanlab.init(
            project="LRP-Baselines-VLM",
            experiment_name="MockAgent-Test",
            config={
                "dataset": "NEO-LRP",
                "benchmarks": benchmarks,
                "total_instances": len(all_instances),
                "count_20": len(instances_by_category[20]),
                "count_50": len(instances_by_category[50]),
                "count_100": len(instances_by_category[100]),
                "count_200": len(instances_by_category[200]),
                "agent": "MockAgent"
            }
        )
    except Exception as e:
        print(f"SwanLab init failed: {e}")

    # 4. Initialize Agent and Manager
    # agent = MockAgent()

    api_base_url = "http://localhost:8000/v1"
    api_key = "token-abc123456"
    agent = VLMAgent(
        api_key=api_key,
        api_base_url=api_base_url,
        model_name="vlm"
    )

    manager = WorkerManager(env_name=env_name)
    
    # 5. Process by Category
    total_costs = []
    total_times = []
    
    for category in [20,50,100]:#, 50, 100, 200]:
        instances = instances_by_category[category]
        if not instances:
            continue
            
        print(f"\n=== Processing Category: {category} (Count: {len(instances)}) ===")
        
        cat_costs = []
        cat_times = []
        
        for i, instance in enumerate(instances):
            instance_name = instance['name']
            benchmark_name = instance['benchmark']
            generator = instance['generator']
            num_loc = instance['num_cust']
            
            print(f"\n[{i+1}/{len(instances)}] Processing [{benchmark_name}] {instance_name} (Size: {num_loc})")
            
            try:
                start_time = time.time()
                
                # Setup Environment
                env_kwargs = {'generator': generator}
                worker = RouteWorker(
                    env_name=env_name,
                    seed=1234,
                    env_num=1,
                    device="cpu",
                    num_loc=num_loc,
                    return_topk_options=26,
                    image_obs="base64",
                    env_kwargs=env_kwargs
                )
                
                obs_list, infos = worker.reset()
                done = False
                step_count = 0
                total_reward = 0
                
                # Interaction Loop
                while not done: # Safety limit
                    step_count += 1
                    
                    next_observations = manager.process_step(obs_list)
                    raw_actions = agent.batch_generate(**next_observations)
                    actions, valids = manager.process_action(raw_actions)
                    
                    obs_list, rewards, dones, infos = worker.step(actions)
                    done = all(dones)
                    total_reward += rewards[0]
                
                
                duration = time.time() - start_time

                # Calculate cost using baseline evaluator for consistency
                # Flatten actions from batch (assuming env_num=1)
                actions = [step_acts[0] for step_acts in worker.actions]
                
                # Get final state
                td_final = worker._td
                
                # Get scale (max_coord) and prohdon scale
                if "scale_factor" in td_final.keys():
                    # Generator returns max_coord as scale_factor
                    scale = td_final["scale_factor"][0].item()
                else:
                    scale = td_final.get("scale", torch.tensor([1.0]))[0].item()
                
                rc_cal_index = td_final.get("rc_cal_index", torch.tensor([-1.0]))[0].item()
                prodhon_scale = 100.0 if abs(rc_cal_index) < 1e-5 else 1.0

                # Un-normalize locations
                locs_norm = td_final["locs"][0].cpu().numpy()
                locs_raw = locs_norm * scale
                
                # Costs are assumed to be un-normalized in the generator (based on previous request)
                depot_open_cost_raw = td_final["depot_open_cost"][0].cpu().numpy()
                vehicle_cost_raw = td_final["vehicle_cost"][0].item()
                
                num_depots = td_final["num_depots"][0].item()
                
                # Reconstruct routes (Same logic as train_lrp_am.py)
                routes = []
                curr_d = 0
                current_route = []
                
                iterator = iter(actions)
                try:
                    first_node = next(iterator)
                    if first_node < num_depots:
                        curr_d = first_node
                    else:
                        # Starts with customer, assume depot 0
                        curr_d = 0 
                        current_route.append(first_node - num_depots)
                except StopIteration:
                    pass
                
                for node in iterator:
                    if node < num_depots:
                        # Switch depot
                        if current_route:
                            routes.append((curr_d, current_route))
                            current_route = []
                        curr_d = node
                    else:
                        current_route.append(node - num_depots)
                        
                if current_route:
                    routes.append((curr_d, current_route))
                
                # Evaluate
                # We pass scale_factor=prodhon_scale because locs_raw is already scaled by max_coord (scale)
                final_cost = evaluate_lrp(
                    routes, 
                    locs=locs_raw[num_depots:], 
                    depot_locs=locs_raw[:num_depots], 
                    depot_open_cost=depot_open_cost_raw,
                    vehicle_cost=vehicle_cost_raw,
                    scale_factor=prodhon_scale 
                )
                
                print(f"  Result: Cost = {final_cost:.2f}, Steps = {step_count}, Time = {duration:.4f}s")
                
                cat_costs.append({
                    'cost': final_cost, 
                    'time': duration,
                    'benchmark': benchmark_name
                })
                cat_times.append(duration)
                total_costs.append(final_cost)
                
                # Log individual run
                if swanlab.get_run():
                    swanlab.log({
                        f"{benchmark_name}/{category}/cost": final_cost,
                        f"{benchmark_name}/{category}/time": duration,
                    })
                
            except Exception as e:
                print(f"Error processing {instance_name}: {e}")
                # import traceback
                # traceback.print_exc()
        
        # Log category stats
        if cat_costs:
            all_costs = [x['cost'] for x in cat_costs]
            avg_cat_cost = np.mean(all_costs)
            avg_cat_time = np.mean(cat_times)
            
            print(f"--- Category {category} Summary ---")
            print(f"Avg Cost: {avg_cat_cost:.2f}")
            print(f"Avg Time: {avg_cat_time:.4f}s")

            # Calculate and log per-benchmark average within this category
            if swanlab.get_run():
                benchmarks_in_cat = set(x['benchmark'] for x in cat_costs)
                for b_name in benchmarks_in_cat:
                    b_costs = [x['cost'] for x in cat_costs if x['benchmark'] == b_name]
                    b_times = [x['time'] for x in cat_costs if x['benchmark'] == b_name]
                    
                    if b_costs:
                        avg_b_cost = np.mean(b_costs)
                        avg_b_time = np.mean(b_times)
                        
                        swanlab.log({
                            f"{b_name}/{category}/avg_cost": avg_b_cost,
                            f"{b_name}/{category}/avg_time": avg_b_time
                        })
    
    print("\n" + "="*30)
    print("Benchmark Completed")
    if total_costs:
        avg_cost = np.mean(total_costs)
        avg_time = np.mean(total_times)
        print(f"Overall Average Cost: {avg_cost:.2f}")
        print(f"Overall Average Time: {avg_time:.4f}s")
        
        if swanlab.get_run():
            swanlab.log({
                "avg_cost_total": avg_cost,
                "avg_time_total": avg_time
            })
    print("="*30)

if __name__ == "__main__":
    test_lrp_env()
