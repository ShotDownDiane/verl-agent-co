import torch
import os
from tensordict import TensorDict
import numpy as np
import scipy.sparse
import scipy.sparse.csgraph

def parse_pmed(file_path):
    """
    Parses OR-Lib pmed file (P-Median Problem / Facility Location).
    Format:
    N E P
    u v cost
    ...
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Skip empty lines
    lines = [l.strip() for l in lines if l.strip()]
    
    header = lines[0].split()
    N = int(header[0])
    E = int(header[1])
    P = int(header[2])
    
    # Construct adjacency matrix
    row = []
    col = []
    data = []
    
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3: continue
        u, v, cost = int(parts[0]), int(parts[1]), float(parts[2])
        # 1-based to 0-based
        u -= 1
        v -= 1
        
        # Undirected graph
        row.append(u)
        col.append(v)
        data.append(cost)
        row.append(v)
        col.append(u)
        data.append(cost)
        
    adj = scipy.sparse.coo_matrix((data, (row, col)), shape=(N, N))
    
    # Compute shortest paths
    # directed=False ensures it treats matrix as undirected
    dist_matrix = scipy.sparse.csgraph.shortest_path(adj, directed=False)
    
    # Convert to torch
    dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32)
    
    # FLPEnv expects locs. We provide dummy locs.
    locs = torch.zeros((N, 2), dtype=torch.float32)
    
    # Initial distances (to chosen facilities) should be infinity
    distances = torch.full((N,), float('inf'), dtype=torch.float32)

    # Wrap in TensorDict
    td = TensorDict({
        "locs": locs.unsqueeze(0), # [1, N, 2]
        "orig_distances": dist_tensor.unsqueeze(0), # [1, N, N]
        "to_choose": torch.tensor([P]),
        "num_loc": torch.tensor([N]),
        "distances": distances.unsqueeze(0) # [1, N]
    }, batch_size=[1])
    
    return [td]

def parse_pmed_mclp(file_path):
    """
    Parses OR-Lib pmed file and adapts it for MCLP.
    Uses shortest path distances as distance matrix.
    Sets coverage radius heuristically.
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    lines = [l.strip() for l in lines if l.strip()]
    header = lines[0].split()
    N = int(header[0])
    E = int(header[1])
    P = int(header[2])
    
    row = []
    col = []
    data = []
    
    for line in lines[1:]:
        parts = line.split()
        if len(parts) < 3: continue
        u, v, cost = int(parts[0]), int(parts[1]), float(parts[2])
        u -= 1
        v -= 1
        row.append(u)
        col.append(v)
        data.append(cost)
        row.append(v)
        col.append(u)
        data.append(cost)
        
    adj = scipy.sparse.coo_matrix((data, (row, col)), shape=(N, N))
    dist_matrix = scipy.sparse.csgraph.shortest_path(adj, directed=False)
    dist_tensor = torch.tensor(dist_matrix, dtype=torch.float32)
    
    # Dummy locs
    locs = torch.zeros((N, 2), dtype=torch.float32)
    
    # Heuristic radius: mean of finite distances
    valid_dists = dist_tensor[torch.isfinite(dist_tensor)]
    if valid_dists.numel() > 0:
        radius = valid_dists.mean() * 0.5
    else:
        radius = 10.0 # Fallback
        
    td = TensorDict({
        "demand_locs": locs.unsqueeze(0),
        "facility_locs": locs.unsqueeze(0),
        "demand_weights": torch.ones(N, dtype=torch.float32).unsqueeze(0),
        "coverage_radius": torch.tensor([radius], dtype=torch.float32),
        "distance_matrix": dist_tensor.unsqueeze(0),
        "num_facilities_to_select": torch.tensor([P]),
        "covered_demand": torch.zeros(N, dtype=torch.float32).unsqueeze(0),
        "chosen": torch.zeros(N, dtype=torch.bool).unsqueeze(0),
        "i": torch.zeros(1, dtype=torch.int64)
    }, batch_size=[1])
    
    return [td]

def parse_estein(file_path):
    """
    Parses OR-Lib estein file (Euclidean Steiner Tree).
    Format typically:
    [Num Problems]
    [Num Points P1]
    x1 y1
    ...
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    problems = []
    current_coords = []
    expected_points = None
    
    for line in lines:
        line = line.strip()
        if not line: continue
        
        try:
            parts = line.split()
            nums = [float(x) for x in parts]
        except ValueError:
            continue
            
        if len(nums) == 1:
            val = int(nums[0])
            if expected_points is None:
                expected_points = val
            elif len(current_coords) == expected_points:
                problems.append(torch.tensor(current_coords, dtype=torch.float32))
                current_coords = []
                expected_points = val
            else:
                expected_points = val
                current_coords = []
                
        elif len(nums) == 2:
            current_coords.append(nums)
            
    if expected_points is not None and len(current_coords) == expected_points:
        problems.append(torch.tensor(current_coords, dtype=torch.float32))
        
    print(f"Parsed {len(problems)} instances from {file_path}")
    
    data_list = []
    for loc in problems:
        num_nodes = loc.shape[0]
        terminals = torch.arange(num_nodes)
        
        x = torch.arange(num_nodes)
        u, v = torch.meshgrid(x, x, indexing='ij')
        mask = u != v
        src = u[mask]
        dst = v[mask]
        edge_list = torch.stack([src, dst], dim=0)
        
        edge_weights_matrix = torch.cdist(loc, loc)
        
        adjacency_mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
        adjacency_mask.fill_diagonal_(False)
        
        td = TensorDict({
            "locs": loc.unsqueeze(0),
            "terminals": terminals.unsqueeze(0),
            "edge_list": edge_list.transpose(0, 1).unsqueeze(0), # [1, E, 2]
            "edge_weights": edge_weights_matrix.unsqueeze(0), # [1, N, N]
            "adjacency": adjacency_mask.unsqueeze(0), # [1, N, N]
            "num_terminals": torch.tensor([num_nodes]),
            "num_loc": torch.tensor([num_nodes]),
            "num_edges": torch.tensor([edge_list.shape[1]])
        }, batch_size=[1])
        
        data_list.append({'td': td})
        
    return data_list

if __name__ == "__main__":
    # Test
    path = "/root/autodl-tmp/or-library/estein1.txt"
    if os.path.exists(path):
        data = parse_estein(path)
        if len(data) > 0:
            print(f"First instance loc shape: {data[0]['td']['locs'].shape}")
