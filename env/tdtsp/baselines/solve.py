import argparse
import os
import json
import time
from .utils import load_tdtsp_benchmark_matrix, load_tdtsp_benchmark_instance
from .evaluator import TDTSPEvaluator
from .heuristics import GreedyRandomized
from .sa import SimulatedAnnealing
from .sah import SAH
from .aco import ACO

def solve_tdtsp(instance_path, matrix_path, method="sah", **kwargs):
    """
    Solve a TDTSP instance using the specified method.
    
    Args:
        instance_path: Path to the instance file (.txt or similar)
        matrix_path: Path to the travel time matrix file
        method: 'greedy', 'sa', 'sah', or 'aco'
        **kwargs: Arguments passed to the solver
    
    Returns:
        tour: List of node indices
        cost: Total travel time
    """
    # 1. Load Matrix
    matrix, duration = load_tdtsp_benchmark_matrix(matrix_path)
    
    # 2. Load Instance
    # Note: Benchmark instance loading might need adjustment based on specific file format
    # The utils.py loader assumes a specific format (lines with ID, coords, TW, etc.)
    node_indices, time_windows, service_times = load_tdtsp_benchmark_instance(instance_path)
    
    # 3. Create Evaluator
    evaluator = TDTSPEvaluator(
        matrix=matrix,
        duration=duration,
        time_windows=time_windows,
        service_times=service_times,
        start_time=0.0 # Adjust if needed
    )
    
    # 4. Select Solver
    if method == "greedy":
        solver = GreedyRandomized(evaluator, k=kwargs.get("k", 3))
    elif method == "sa":
        solver = SimulatedAnnealing(
            evaluator, 
            initial_temp=kwargs.get("initial_temp", 100),
            max_iter=kwargs.get("max_iter", 5000)
        )
    elif method == "sah":
        solver = SAH(
            evaluator,
            initial_temp=kwargs.get("initial_temp", 100),
            max_iter=kwargs.get("max_iter", 5000)
        )
    elif method == "aco":
        solver = ACO(
            evaluator,
            num_ants=kwargs.get("num_ants", 20),
            iterations=kwargs.get("iterations", 100)
        )
    else:
        raise ValueError(f"Unknown method: {method}")
        
    # 5. Solve
    tour, cost = solver.solve()
    return tour, cost
