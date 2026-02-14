
import numpy as np
from ortools.sat.python import cp_model

def solve_ortools(matrix, duration, time_windows, service_times, config):
    """
    Solve TDTSPTW using OR-Tools CP-SAT solver.
    Nodes: 0 (Start Depot), 1..N-1 (Customers), N (End Depot).
    
    Args:
        matrix: [N, N, T] travel time matrix
        duration: float, duration of each time step
        time_windows: [N, 2]
        service_times: [N]
        config: dict with 'time_limit' etc.
        
    Returns:
        tour: list of visited nodes (indices 0..N-1)
        cost: float (makespan)
    """
    
    # Scale factor for converting floats to integers (CP-SAT requires integers)
    SCALE = 10
    
    # Unpack config
    time_limit = config.get("time_limit", 60)
    
    num_nodes = matrix.shape[0] # Includes depot 0
    # Create extended node list: 0, 1, ..., N-1, N (copy of 0)
    start_depot = 0
    end_depot = num_nodes
    customers = list(range(1, num_nodes))
    nodes = [start_depot] + customers + [end_depot]
    
    # Ensure inputs are numpy arrays
    matrix = np.array(matrix)
    time_windows = np.array(time_windows)
    service_times = np.array(service_times)
    
    matrix_int = (matrix * SCALE).astype(int)
    tw_int = (time_windows * SCALE).astype(int)
    srv_int = (service_times * SCALE).astype(int)
    bin_duration_int = int(duration * SCALE)
    num_bins = matrix.shape[2]
    
    model = cp_model.CpModel()
    
    # --- Variables ---
    
    # x[i, j]: BoolVar, true if arc (i, j) is used
    x = {}
    for i in nodes:
        for j in nodes:
            if i == j: continue
            if i == end_depot: continue # No outgoing from end depot
            if j == start_depot: continue # No incoming to start depot
            x[i, j] = model.NewBoolVar(f"x_{i}_{j}")
            
    # Time variables
    # Horizon: Ensure it's large enough.
    # If max time window is large, use that.
    max_time = np.max(time_windows) * SCALE + np.sum(service_times) * SCALE + np.max(matrix) * SCALE * num_nodes
    HORIZON = int(max_time * 1.5) # Safety buffer
    
    a = {} # Arrival time
    s = {} # Service start time
    d = {} # Departure time
    
    for i in nodes:
        orig_i = 0 if i == end_depot else i
        
        early = tw_int[orig_i][0]
        late = tw_int[orig_i][1]
        srv = srv_int[orig_i]
        
        # Ensure bounds are within horizon
        early = min(early, HORIZON)
        late = min(late, HORIZON)
        
        a[i] = model.NewIntVar(0, HORIZON, f"a_{i}")
        s[i] = model.NewIntVar(early, late, f"s_{i}") 
        d[i] = model.NewIntVar(early + srv, HORIZON, f"d_{i}")
        
        # s[i] >= a[i]
        model.Add(s[i] >= a[i])
        # d[i] == s[i] + service
        model.Add(d[i] == s[i] + srv)

    # Fix start depot
    model.Add(a[start_depot] == 0)
    model.Add(s[start_depot] == 0) 
    
    # --- Path Constraints ---
    
    # 1. Leave start depot exactly once
    model.Add(sum(x[start_depot, j] for j in customers) == 1)
    
    # 2. Enter end depot exactly once
    model.Add(sum(x[i, end_depot] for i in customers) == 1)
    
    # 3. Flow conservation for customers
    for k in customers:
        model.Add(sum(x[i, k] for i in nodes if (i, k) in x) == 1) # Incoming
        model.Add(sum(x[k, j] for j in nodes if (k, j) in x) == 1) # Outgoing
        
    # --- Time Dependent Constraints ---
    
    for i in nodes:
        if i == end_depot: continue
        
        orig_i = 0 if i == end_depot else i
        
        # bin_idx = d[i] // bin_duration
        bin_idx = model.NewIntVar(0, num_bins - 1, f"bin_{i}")
        model.AddDivisionEquality(bin_idx, d[i], bin_duration_int)
        
        for j in nodes:
            if (i, j) in x:
                orig_j = 0 if j == end_depot else j
                
                profile = matrix_int[orig_i, orig_j].tolist()
                min_t = min(profile)
                max_t = max(profile)
                travel_var = model.NewIntVar(min_t, max_t, f"travel_{i}_{j}")
                
                model.AddElement(bin_idx, profile, travel_var)
                
                # if x[i,j] -> a[j] >= d[i] + travel_var
                model.Add(a[j] >= d[i] + travel_var).OnlyEnforceIf(x[i, j])

    # Objective: Minimize Arrival at End Depot
    model.Minimize(a[end_depot])
    
    # Solver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.log_search_progress = False
    solver.parameters.num_workers = 1 # Avoid excessive threading in parallel execution
    
    status = solver.Solve(model)
    
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        obj_val = solver.ObjectiveValue() / SCALE
        
        # Reconstruct path
        path = [start_depot]
        curr = start_depot
        while True:
            found_next = False
            for j in nodes:
                if (curr, j) in x and solver.BooleanValue(x[curr, j]):
                    path.append(j)
                    curr = j
                    found_next = True
                    break
            if not found_next or curr == end_depot:
                break
        
        # Convert path: remove end_depot if present (it's N), convert to original indices
        # Original nodes are 0..N-1.
        # Path contains 0, ..., N.
        # We need to return a tour that TDTSPEvaluator accepts.
        # Usually evaluator expects [0, c1, c2, ..., cn, 0] or just [0, c1, ..., cn] depending on implementation.
        # Let's check TDTSPEvaluator.calculate_cost.
        # Assuming it expects a list of node indices.
        
        final_tour = []
        for node in path:
            if node == end_depot:
                final_tour.append(0)
            else:
                final_tour.append(node)
                
        return final_tour, obj_val
    else:
        # Return empty tour and inf cost
        return [], float('inf')
