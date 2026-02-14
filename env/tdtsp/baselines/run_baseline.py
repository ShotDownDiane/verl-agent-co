
import os
import sys

# Add rl4co-urban to python path if needed
sys.path.append('/root/autodl-tmp/rl4co-urban')

# Add baselines dir to path
sys.path.append(os.path.dirname(__file__))

import numpy as np

# Import directly
from utils import load_tdtsp_benchmark_matrix, load_tdtsp_benchmark_instance
from evaluator import TDTSPEvaluator
from heuristics import GreedyRandomized

def main():
    base_dir = os.path.dirname(os.path.dirname(__file__)) # rl4co/envs/routing/tdtsp
    data_dir = os.path.join(base_dir, 'data', 'TDTSPBenchmark')
    
    instance_path = os.path.join(data_dir, 'Instances', 'Instances_TW', '10', 'inst_10_1_TW.txt')
    matrix_path = os.path.join(data_dir, 'Matrices', 'matrix00.txt')
    
    print(f"Testing TDTSP Solver with:")
    print(f"  Instance: {instance_path}")
    print(f"  Matrix: {matrix_path}")
    
    if not os.path.exists(instance_path):
        print("Error: Instance file not found!")
        return
    if not os.path.exists(matrix_path):
        print("Error: Matrix file not found!")
        return
        
    # Manual Solve Logic
    try:
        # 1. Load
        matrix, duration = load_tdtsp_benchmark_matrix(matrix_path)
        node_indices, time_windows, service_times = load_tdtsp_benchmark_instance(instance_path)
        
        print(f"  Matrix Shape (Original): {matrix.shape}")
        print(f"  Instance Nodes: {len(node_indices)}")
        print(f"  Node Indices: {node_indices}")

        # Slice Matrix if necessary
        # matrix is [N_all, N_all, T]
        # node_indices maps 0..N_inst -> Global ID
        if len(node_indices) <= matrix.shape[0]:
            print("  Slicing matrix to match instance nodes...")
            # Use numpy advanced indexing to extract submatrix
            # We want matrix[i, j, :] where i, j are in node_indices
            # ix_ does cartesian product
            sub_matrix = matrix[np.ix_(node_indices, node_indices)]
            # sub_matrix shape will be [N_inst, N_inst, T]
            matrix = sub_matrix
            print(f"  Matrix Shape (Sliced): {matrix.shape}")
        
        print(f"  Time Windows Length: {len(time_windows)}")
        print(f"  Service Times Length: {len(service_times)}")

        # 2. Evaluator
        evaluator = TDTSPEvaluator(
            matrix=matrix,
            duration=duration,
            time_windows=time_windows,
            service_times=service_times,
            start_time=0.0
        )
        
        # 3. Solver
        print("\nRunning Greedy Solver...")
        solver = GreedyRandomized(evaluator, k=1)
        
        tour, cost = solver.solve()
        print(f"  Result: Cost = {cost:.2f}")
        print(f"  Tour: {tour}")
        
    except Exception as e:
        print(f"  Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
