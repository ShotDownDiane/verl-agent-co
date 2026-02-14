
import os
import sys
import time
import numpy as np
import torch
import swanlab
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Add rl4co-urban to python path
sys.path.append('/root/autodl-tmp/rl4co-urban')
# Add baselines dir to path
sys.path.append(os.path.dirname(__file__))

from rl4co.envs.routing.tdvrp.baselines.evaluator import TDVRPEvaluator
from rl4co.envs.routing.tdvrp.baselines.solve import solve_tdvrp_from_data
from rl4co.envs.routing.tdvrp.baselines.ortools_solver import solve_ortools
from rl4co.envs.routing.tdvrp.generator import TDVRPGenerator

def solve_instance(args):
    """Worker function for parallel solving"""
    matrix, duration, time_windows, service_times, method, config = args
    
    # solve_tdvrp_from_data returns (routes, cost)
    start_t = time.time()
    try:
        if method == "ortools":
            routes, cost = solve_ortools(
                matrix=matrix,
                duration=duration,
                time_windows=time_windows,
                service_times=service_times,
                config=config
            )
        else:
            routes, cost = solve_tdvrp_from_data(
            matrix=matrix,
            duration=duration,
            time_windows=time_windows,
            service_times=service_times,
            method=method,
            iterations=config.get("iterations", 100),
            ants=config.get("ants", 20),
            penalty_value=3.0
        )
        end_t = time.time()
        
        # Re-evaluate with details if needed
        evaluator = TDVRPEvaluator(
            matrix=matrix,
            duration=duration,
            time_windows=time_windows,
            service_times=service_times,
            penalty_value=3.0
        )
        cost, details = evaluator.calculate_cost(routes, return_details=True)
        was_late = 1 if details["total_violation"] > 0 else 0
        
        return cost, end_t - start_t, was_late, len(routes), details
    except Exception as e:
        print(f"Error solving instance: {e}")
        import traceback
        traceback.print_exc()
        return float('inf'), 0.0, 0, 0, None

def main():
    parser = argparse.ArgumentParser(description="Run TDVRP Baseline")
    parser.add_argument("--method", type=str, default="sa", choices=["greedy", "sa", "sah", "aco", "ortools", "alns", "grasp"], help="Solver method")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "slow"], help="Configuration mode")
    args = parser.parse_args()

    # --- Configuration ---
    num_time_steps = 37
    base_data_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    method = args.method # method: greedy, sa, sah, aco, ortools, alns
    mode = args.mode      # mode: fast, slow
    num_workers = 16
    num_samples_to_test = 500 # Reduced for testing
    
    cities = ["berlin"]
    node_configs = [500]
    
    # --- Solver Configurations ---
    SOLVER_CONFIGS = {
        "greedy": {
            "fast": {"k": 1},
            "slow": {"k": 10}
        },
        "sa": {
            "fast": {"iterations": 200, "initial_temp": 100.0},
            "slow": {"iterations": 10000, "initial_temp": 1000.0}
        },
        "sah": {
            "fast": {"iterations": 200, "initial_temp": 100.0},
            "slow": {"iterations": 10000, "initial_temp": 1000.0}
        },
        "aco": {
            "fast": {"ants": 10, "iterations": 20},
            "slow": {"ants": 50, "iterations": 100}
        },
        "ortools": {
            "fast": {"time_limit": 10},
            "slow": {"time_limit": 30}
        },
        "alns": {
            "fast": {"iterations": 50},
            "slow": {"iterations": 5000}
        },
        "grasp": {
            "fast": {"iterations": 20, "time_limit": 5},
            "slow": {"iterations": 100, "time_limit": 30}
        }
    }

    # Select config based on method and mode
    solver_config = SOLVER_CONFIGS.get(method, {}).get(mode, {})
    if not solver_config:
        print(f"Warning: No configuration found for method '{method}' and mode '{mode}'. Using default.")
        solver_config = {
            "iterations": 100, 
            "ants": 10,
            "time_limit": 5,
            "initial_temp": 100.0
        }
        
    print(f"Running {method} in {mode} mode with config: {solver_config}")

    # --- Initialize SwanLab ---
    swanlab.init(
        project="TDVRP-Baselines-large-0204",
        experiment_name=f"{method}-{mode}-test",
        config={
            "method": method,
            "mode": mode,
            "cities": cities,
            "node_configs": node_configs,
            "num_workers": num_workers,
            "num_samples": num_samples_to_test,
            **solver_config
        }
    )

    total_results = {}

    for num_nodes in node_configs:
            print(f"\n================ CONFIG: {num_nodes} Nodes ================")
            for city in cities:
                if num_nodes >= 500:
                    # Large instances: direct json loading
                    base_large_path = "/root/autodl-tmp/rl4co-urban/rl4co/envs/routing/tdvrp/data/vrptdt-benchmark/instances"
                    instance_path = os.path.join(base_large_path, f"{city}_{num_nodes}.json")
                    matrix_file = os.path.join(base_large_path, f"{city}_{num_nodes}_tt.json.bz2")
                    test_data_path = None
                    
                    if not os.path.exists(instance_path):
                        print(f"Skipping {city}_{num_nodes}: Instance not found at {instance_path}")
                        continue
                else:
                    test_data_path = f"/root/autodl-tmp/tdtsp_dataset_random/{city}_{num_nodes}_random_test.npz"
                    instance_path = None
                    matrix_file = matrix_path
                    
                    if not os.path.exists(test_data_path):
                        print(f"Skipping {city}_{num_nodes}: NPZ not found at {test_data_path}")
                        continue

                city_node_key = f"{city}_{num_nodes}"
                print(f"\n>>> Testing: {city_node_key}")
                
                try:
                    if num_nodes >= 500:
                        generator = TDVRPGenerator(
                            instance_path=instance_path,
                            matrix_path=matrix_file,
                            num_matrix_steps=num_time_steps,
                            data_path=None,
                            num_nodes=num_nodes + 1,
                            random_sample=False,
                            phase="all"
                        )
                        generator.num_samples = 1
                    else:
                        generator = TDVRPGenerator(
                            data_path=test_data_path,
                            base_data_path=base_data_path,
                            matrix_path=matrix_path,
                            num_matrix_steps=num_time_steps,
                            random_sample=False,
                            phase="all"
                        )

                    if generator.num_samples == 0:
                        continue

                    # Get batch of samples (only take first num_samples_to_test)
                    td = generator(batch_size=[min(num_samples_to_test, generator.num_samples)])
                    actual_samples = td.batch_size[0]
                    
                    tasks = []
                    for i in range(actual_samples):
                        matrix = td["travel_time_matrix"][i].numpy()
                        duration = td["time_step_duration"][i].item()
                        time_windows = td["time_windows"][i].numpy()
                        service_times = generator.service_time
                        
                        tasks.append((matrix, duration, time_windows, service_times, method, solver_config))    

                    # Run Parallel Solvers
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        all_results = list(tqdm(executor.map(solve_instance, tasks), total=len(tasks), desc=f"Solving {city_node_key}"))
                    
                except Exception as e:
                    print(f"Error loading {city_node_key}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue

                if not all_results:
                    continue
                    
                # Calculate Metrics
                city_costs = [r[0] for r in all_results if r[0] != float('inf')]
                city_times = [r[1] for r in all_results if r[0] != float('inf')]
                city_lates = sum([1 for r in all_results if r[2]])
                city_routes = [r[3] for r in all_results if r[0] != float('inf')]
                late_details = [r[4] for r in all_results if r[2]]
                
                if not city_costs:
                    continue

                avg_cost = np.mean(city_costs)
                avg_time = np.mean(city_times)
                late_rate = city_lates / len(city_costs)
                avg_routes = np.mean(city_routes)
                
                print(f"  Results for {city_node_key}:")
                print(f"    Avg Cost: {avg_cost:.2f}")
                print(f"    Avg Time: {avg_time:.4f}s")
                print(f"    Avg Routes: {avg_routes:.2f}")
                print(f"    Late Rate: {late_rate:.2%} ({city_lates}/{len(city_costs)})")
                
                # Print one late example if exists
                if late_details:
                    print(f"\n>>> Analyzing a Late Example from {city_node_key}:")
                    example = late_details[0]
                    for r_idx, route in enumerate(example["routes"]):
                        if route["violation_sec"] > 0:
                            print(f"  Route {r_idx} (Violation: {route['violation_sec']:.2f}s):")
                            print(f"    {'Node':<6} {'Arrival (s)':<15} {'Window (s)':<25} {'Late (s)':<10}")
                            for h in route["history"]:
                                win_str = f"[{h['window'][0]:.1f}, {h['window'][1]:.1f}]"
                                late_str = f"{h['late']:.2f}" if h['late'] > 0 else "-"
                                print(f"    {h['node']:<6} {h['arrival']:<15.2f} {win_str:<25} {late_str:<10}")
                    print("-" * 50)
                
                # Log to SwanLab
                swanlab.log({
                    f"{num_nodes}nodes/{city}/avg_cost": avg_cost,
                    f"{num_nodes}nodes/{city}/avg_time": avg_time,
                    f"{num_nodes}nodes/{city}/late_rate": late_rate,
                    f"{num_nodes}nodes/{city}/avg_routes": avg_routes,
                })
                
                total_results[city_node_key] = avg_cost

if __name__ == "__main__":
    main()
