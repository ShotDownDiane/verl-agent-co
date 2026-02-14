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

from evaluator import TDTSPEvaluator
from heuristics import GreedyRandomized
from sa import SimulatedAnnealing
from alns import ALNS
from sah import SAH
from aco import ACO
from grasp import TDTSPGRASPSolver
from ortools_solver import solve_ortools
from rl4co.envs.routing.tdtsp.env_tw import TDTSPTWGenerator

def solve_instance(args):
    """Worker function for parallel solving"""
    matrix, duration, time_windows, service_times, method, config, label = args
    
    # 1. Evaluator
    evaluator = TDTSPEvaluator(
        matrix=matrix,
        duration=duration,
        time_windows=time_windows,
        service_times=service_times,
        start_time=0.0,
        penalty_value=3.0
    )

    # 2. Select Solver
    if method == "bks":
        # Simply evaluate the provided label (best known solution from dataset)
        start_t = time.time()
        cost = evaluator.calculate_cost(label)
        end_t = time.time()
        return cost, end_t - start_t, evaluator.was_late
    
    elif method == "greedy":
        solver = GreedyRandomized(evaluator, k=config.get("k", 3))
    elif method == "sa":
        solver = SimulatedAnnealing(
            evaluator, 
            initial_temp=config.get("initial_temp", 100.0),
            max_iter=config.get("max_iter", 1000)
        )
    elif method == "alns":
        solver = ALNS(
            evaluator,
            initial_temp=config.get("initial_temp", 100.0),
            max_iter=config.get("max_iter", 1000),
        )
    elif method == "sah":
        solver = SAH(
            evaluator,
            initial_temp=config.get("initial_temp", 100.0),
            max_iter=config.get("max_iter", 1000)
        )
    elif method == "aco":
        solver = ACO(
            evaluator,
            num_ants=config.get("num_ants", 20),
            iterations=config.get("iterations", 100)
        )
    elif method == "grasp":
        solver = TDTSPGRASPSolver(
            evaluator,
            max_iterations=config.get("max_iterations", 100),
            alpha=config.get("alpha", 0.1),
            time_limit=config.get("time_limit", None)
        )
    elif method == "ortools":
        start_t = time.time()
        tour, cost = solve_ortools(matrix, duration, time_windows, service_times, config)
        end_t = time.time()
        
        # Validate with evaluator (optional but recommended for consistency)
        if tour:
            cost_check, details = evaluator.calculate_cost(tour, return_details=True)
            was_late = 1 if details["total_violation"] > 0 else 0
            # Use evaluator cost if successful, or stick to solver cost
            # cost = cost_check 
        else:
            was_late = 1
            cost = float('inf')
            
        return cost, end_t - start_t, was_late
    else:
        raise ValueError(f"Unknown method: {method}")

    start_t = time.time()
    tour, cost = solver.solve()
    end_t = time.time()

    cost, details = evaluator.calculate_cost(tour,return_details=True)

    was_late = 1 if details["total_violation"] > 0 else 0
    
    return cost, end_t - start_t, was_late

def main():
    parser = argparse.ArgumentParser(description="Run TDTSP Baseline")
    parser.add_argument("--method", type=str, default="greedy", choices=["greedy", "sa", "sah", "aco", "ortools", "alns", "bks", "grasp", "tabu"], help="Solver method")
    parser.add_argument("--mode", type=str, default="fast", choices=["fast", "slow"], help="Configuration mode")
    args = parser.parse_args()

    # --- Configuration ---
    num_time_steps = 37
    base_data_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    method = args.method # method: greedy, sa, sah, aco, ortools, alns
    mode = args.mode     # mode: fast, slow
    num_workers = 1 # Adjust based on CPU cores
    
    cities = ["new"] # Only test berlin
    node_configs = [20,50]
    
    # --- Solver Configurations ---
    SOLVER_CONFIGS = {
        "sah": {
            "fast": {"max_iter": 1000, "initial_temp": 1000.0},
            "slow": {"max_iter": 10000, "initial_temp": 1000.0}
        },
        "aco": {
            "fast": {"num_ants": 2, "iterations": 10},
            "slow": {"num_ants": 30, "iterations": 5000}
        },
        "ortools": {
            "fast": {"time_limit": 30},
            "slow": {"time_limit": 120}
        },
        "alns": {
            "fast": {"max_iter": 1000, "initial_temp": 1000.0},
            "slow": {"max_iter": 5000, "initial_temp": 1000.0}
        },
        "grasp": {
            "fast": {"max_iterations": 20, "alpha": 0.1, "time_limit": 5},
            "slow": {"max_iterations": 1000, "alpha": 0.2, "time_limit": 30}
        },
    }
    
    # Select config based on method and mode
    if method == "bks":
        solver_config = {}
    else:
        solver_config = SOLVER_CONFIGS.get(method, {}).get(mode, {})
        if not solver_config:
            print(f"Warning: No configuration found for method '{method}' and mode '{mode}'. Using default.")
            solver_config = {
                "max_iter": 200,
                "initial_temp": 100.0,
                "time_limit": 5
            }
    
    print(f"Running {method} in {mode} mode with config: {solver_config}")
    
    # --- Initialize SwanLab ---
    swanlab.init(
        project="TDTSPTW-Baselines-01281",
        experiment_name=f"{method}-{mode}-test",
        config={
            "method": method,
            "mode": mode,
            "cities": cities,
            "node_configs": node_configs,
            "num_workers": num_workers,
            **solver_config
        }
    )

    total_results = {}

    for num_nodes in node_configs:
        print(f"\n================ CONFIG: {num_nodes} Nodes ================")
        for city in cities:
            test_data_path = f"/root/autodl-tmp/tdtsp_dataset_split/{city}_{num_nodes}_test.npz"
            
            if not os.path.exists(test_data_path):
                print(f"Skipping {city}_{num_nodes}: NPZ not found at {test_data_path}")
                continue

            city_node_key = f"{city}_{num_nodes}"
            print(f"\n>>> Testing: {city_node_key}")
            
            # 1. Initialize Generator for this specific file
            try:
                # Detect unique base files in the NPZ to avoid errors with multi-base files
                with np.load(test_data_path) as data:
                    unique_bases = np.unique(data['base_file'])
                
                all_results = []
                
                for target_base in unique_bases:
                    print(f"  >>> Sub-group: {target_base}")
                    generator = TDTSPTWGenerator(
                        data_path=test_data_path,
                        base_data_path=base_data_path,
                        matrix_path=matrix_path,
                        num_matrix_steps=num_time_steps,
                        random_sample=False,
                        phase="all",
                        target_base_file=target_base
                    )

                    if generator.num_samples == 0:
                        continue

                    # 2. Get instances and labels for this sub-group
                    td = generator(batch_size=[generator.num_samples])
                    labels = generator.labels # Global indices
                    locs_idx = td["locs_idx"].numpy()
                    
                    # 3. Prepare tasks for parallel execution
                    tasks = []
                    for i in range(1):
                        matrix = td["travel_time_matrix"][i].numpy()
                        duration = td["time_step_duration"][i].item()
                        time_windows = td["time_windows"][i].numpy()
                        service_times = [generator.service_time] * matrix.shape[0]
                        service_times[0] = 0.0
                        
                        # Map global label to local indices
                        g2l = {g: l for l, g in enumerate(locs_idx[i])}
                        local_label = [g2l[g] for g in labels[i]]
                        
                        tasks.append((matrix, duration, time_windows, service_times, method, solver_config, local_label))    

                    # 4. Run Parallel Solvers
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        sub_results = list(tqdm(executor.map(solve_instance, tasks), total=len(tasks), desc=f"Solving {city_node_key} ({target_base})"))
                    
                    all_results.extend(sub_results)

            except Exception as e:
                print(f"Error loading {city_node_key}: {e}")
                import traceback
                traceback.print_exc()
                continue

            if not all_results:
                continue
                
            # 5. Calculate Metrics
            city_costs = [r[0] for r in all_results]
            city_times = [r[1] for r in all_results]
            city_lates = sum([1 for r in all_results if r[2]])
            
            avg_cost = np.mean(city_costs)
            avg_time = np.mean(city_times)
            late_rate = city_lates / len(all_results)
            
            print(f"  Results for {city_node_key}:")
            print(f"    Avg Cost: {avg_cost:.2f}")
            print(f"    Avg Time: {avg_time:.4f}s")
            print(f"    Late Rate: {late_rate:.2%} ({city_lates}/{len(all_results)})")
            
            # 6. Log to SwanLab
            swanlab.log({
                f"{num_nodes}nodes/{city}/avg_cost": avg_cost,
                f"{num_nodes}nodes/{city}/avg_time": avg_time,
                f"{num_nodes}nodes/{city}/late_rate": late_rate,
            })
            
            total_results[city_node_key] = avg_cost

    # Final Overall Summary
    for num_nodes in node_configs:
        node_costs = [v for k, v in total_results.items() if f"_{num_nodes}" in k]
        if node_costs:
            node_avg = np.mean(node_costs)
            # swanlab.log({f"overall/{num_nodes}nodes_avg_cost": node_avg})
            print(f"\n>>> Final {num_nodes} Nodes Overall Avg Cost: {node_avg:.2f}")

if __name__ == "__main__":
    main()
