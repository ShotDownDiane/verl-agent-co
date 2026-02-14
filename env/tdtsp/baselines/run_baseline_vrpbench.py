import os
import sys
import time
import numpy as np
import torch
import swanlab
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# Add rl4co-urban to python path
sys.path.append('/root/autodl-tmp/rl4co-urban')
# Add baselines dir to path
sys.path.append(os.path.dirname(__file__))

from evaluator import TDTSPEvaluator
from heuristics import GreedyRandomized
from sa import SimulatedAnnealing
from aco import ACO
from rl4co.envs.routing.tdtsp.env_tw import TDTSPTWGenerator

def solve_instance(args):
    """Worker function for parallel solving"""
    matrix, duration, service_times, method, config = args
    
    # 1. Evaluator
    evaluator = TDTSPEvaluator(
        matrix=matrix,
        duration=duration,
        time_windows=None, # Already handled by duration and service_times in this context
        service_times=service_times,
        start_time=0.0
    )

    # 2. Select Solver
    if method == "greedy":
        solver = GreedyRandomized(evaluator, k=config.get("k", 3))
    elif method == "sa":
        solver = SimulatedAnnealing(
            evaluator, 
            initial_temp=config.get("initial_temp", 100.0),
            max_iter=config.get("max_iter", 5000)
        )
    elif method == "aco":
        solver = ACO(
            evaluator,
            num_ants=config.get("num_ants", 10),
            iterations=config.get("iterations", 100)
        )
    else:
        raise ValueError(f"Unknown method: {method}")

    start_t = time.time()
    tour, cost = solver.solve()
    end_t = time.time()
    
    return cost, end_t - start_t

def main():
    # --- Configuration ---
    num_time_steps = 37
    base_data_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    method = "aco"
    num_workers = 48 # Adjust based on CPU cores
    
    cities = ["berlin","london", "newyork", "nairobi"]
    node_configs = [20, 50]
    
    solver_config = {
        "num_ants": 20,
        "iterations": 100
    }
    
    # --- Initialize SwanLab ---
    swanlab.init(
        project="TDTSP-Baselines",
        experiment_name=f"{method}-multi-city-multi-node",
        config={
            "method": method,
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
            test_data_path = f"/root/autodl-tmp/tdtsp_dataset_random/{city}_{num_nodes}_random_test.npz"
            
            if not os.path.exists(test_data_path):
                print(f"Skipping {city}_{num_nodes}: NPZ not found at {test_data_path}")
                continue

            city_node_key = f"{city}_{num_nodes}"
            print(f"\n>>> Testing: {city_node_key}")
            
            # 1. Initialize Generator for this specific file
            try:
                generator = TDTSPTWGenerator(
                    data_path=test_data_path,
                    base_data_path=base_data_path,
                    matrix_path=matrix_path,
                    num_matrix_steps=num_time_steps,
                    random_sample=False,
                    phase="all"
                )
            except Exception as e:
                print(f"Error loading {city_node_key}: {e}")
                continue

            print(f"  Samples: {generator.num_samples}")
            if generator.num_samples == 0:
                continue

            # 2. Get all instances
            td = generator(batch_size=[generator.num_samples])
            
            # 3. Prepare tasks for parallel execution
            tasks = []
            for i in range(generator.num_samples):
                matrix = td["travel_time_matrix"][i].numpy()
                duration = td["time_step_duration"][i].item()
                service_times = [generator.service_time] * matrix.shape[0]
                service_times[0] = 0.0
                tasks.append((matrix, duration, service_times, method, solver_config))

            # 4. Run Parallel Solvers
            city_costs = []
            city_times = []
            
            start_city_t = time.time()
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                results = list(tqdm(executor.map(solve_instance, tasks), total=len(tasks), desc=f"Solving {city_node_key}"))
            
            for cost, solve_t in results:
                city_costs.append(cost)
                city_times.append(solve_t)
            
            end_city_t = time.time()
            
            # 5. Calculate Metrics
            avg_cost = np.mean(city_costs)
            avg_time = np.mean(city_times)
            total_city_time = end_city_t - start_city_t
            
            print(f"  Results for {city_node_key}:")
            print(f"    Avg Cost: {avg_cost:.2f}")
            print(f"    Avg Time: {avg_time:.4f}s")

            # 6. Log to SwanLab
            swanlab.log({
                f"{num_nodes}nodes/{city}/avg_cost": avg_cost,
                f"{num_nodes}nodes/{city}/avg_time": avg_time,
                f"{num_nodes}nodes/{city}/total_time": total_city_time,
            })
            
            total_results[city_node_key] = avg_cost

    # Final Overall Summary
    for num_nodes in node_configs:
        node_costs = [v for k, v in total_results.items() if f"_{num_nodes}" in k]
        if node_costs:
            node_avg = np.mean(node_costs)
            swanlab.log({f"overall/{num_nodes}nodes_avg_cost": node_avg})
            print(f"\n>>> Final {num_nodes} Nodes Overall Avg Cost: {node_avg:.2f}")

if __name__ == "__main__":
    main()
