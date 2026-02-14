import numpy as np
import os

def load_tdtsp_benchmark_matrix(matrix_path):
    # print(f"Loading matrix from {matrix_path}...")
    with open(matrix_path, 'r') as f:
        lines = f.readlines()
    
    header = lines[0].strip().split()
    num_nodes = int(header[0])
    num_steps = int(header[1])
    step_duration = float(header[2])
    
    # Check lines count
    # expected_lines = 1 + num_nodes * (num_nodes + 1)
    # if len(lines) != expected_lines:
    #     print(f"Warning: Expected {expected_lines} lines, got {len(lines)}")
        
    matrix = np.zeros((num_nodes, num_nodes, num_steps), dtype=np.float32)
    
    line_idx = 1
    for i in range(num_nodes):
        for j in range(num_nodes):
            # Parse line
            values = list(map(float, lines[line_idx].strip().split()))
            matrix[i, j, :] = values
            line_idx += 1
        # Skip empty line
        line_idx += 1
        
    return matrix, step_duration

def load_tdtsp_benchmark_instance(instance_path):
    with open(instance_path, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    # 10 0
    header = lines[0].strip().split()
    
    node_indices = [0] # Start with Depot (Global ID 0)
    time_windows = [] # List of (early, late)
    service_times = []
    
    # Default depot TW and service
    time_windows.append((0, float('inf')))
    service_times.append(0)
    
    for line in lines[1:]: # Skip header
        parts = line.strip().split()
        if not parts: continue
        node_id = int(parts[0])
        node_indices.append(node_id)
        
        # Check for TW columns
        # Format: ID X Y Service Ready Due ...
        if len(parts) >= 6:
            service = float(parts[3])
            ready = float(parts[4])
            due = float(parts[5])
            
            service_times.append(service)
            time_windows.append((ready, due))
        else:
            # No TW info, use defaults
            service_times.append(0.0)
            time_windows.append((0.0, float('inf')))
            
    if len(time_windows) <= 1:
        time_windows = None
        service_times = None
        
    return node_indices, time_windows, service_times
