
from ortools.linear_solver import pywraplp
import numpy as np

class MIPSolver:
    def __init__(self, evaluator, time_limit=60, use_static_approx=True):
        """
        MIP Solver for TDVRP.
        
        Args:
            evaluator: TDVRPEvaluator instance
            time_limit: Time limit in seconds
            use_static_approx: If True, uses mean travel times (CVRP-TW with static times).
                               If False, attempts to model time-dependent travel times (Very Expensive).
        """
        self.evaluator = evaluator
        self.time_limit = time_limit
        self.use_static_approx = use_static_approx
        self.num_nodes = evaluator.num_nodes
        self.num_vehicles = self.num_nodes # Worst case
        
    def solve(self):
        # Create the mip solver with the SCIP backend.
        solver = pywraplp.Solver.CreateSolver('SCIP')
        if not solver:
            print("SCIP solver not available, trying CBC")
            solver = pywraplp.Solver.CreateSolver('CBC')
            if not solver:
                return [], float('inf')

        solver.SetTimeLimit(self.time_limit * 1000)

        # Data
        N = self.num_nodes
        # Depot is 0
        customers = list(range(1, N))
        all_nodes = list(range(N))
        
        # Static Matrix for approximation
        if self.use_static_approx:
            dist_matrix = self.evaluator.matrix.mean(axis=2)
        else:
            # For exact TDVRP, we need time-dependent formulation
            # This is extremely heavy for N=100, so we fallback to static or warn
            print("Warning: Exact TDVRP MIP formulation is too large for N=100. Using static approximation.")
            dist_matrix = self.evaluator.matrix.mean(axis=2)

        # Variables
        # x[i, j] = 1 if arc (i, j) is used
        x = {}
        for i in all_nodes:
            for j in all_nodes:
                if i != j:
                    x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')

        # Time variables
        # t[i] = arrival time at node i
        # We need an upper bound for time. 
        # Using horizon from evaluator or estimation
        horizon = 24 * 3600 * 2 # Safe upper bound
        if self.evaluator.time_windows is not None:
             horizon = max(horizon, np.max(self.evaluator.time_windows))

        t = {}
        for i in all_nodes:
            t[i] = solver.NumVar(0, horizon, f't_{i}')

        # Constraints
        
        # 1. Flow conservation
        # Each customer visited exactly once
        for i in customers:
            solver.Add(solver.Sum(x[j, i] for j in all_nodes if j != i) == 1)
            solver.Add(solver.Sum(x[i, j] for j in all_nodes if j != i) == 1)

        # Depot flow: Leave K times, Enter K times (where K is num vehicles used)
        # But here we model as open number of vehicles.
        # Let's say we have N potential vehicles.
        # Actually, standard formulation allows multiple leaves/enters at depot
        # Or we can treat depot as single node and sum(x[0, j]) = sum(x[j, 0]) = K
        # We leave K free variable, but penalize it in objective
        
        # Vehicles used variable
        k_vehicles = solver.IntVar(1, N, 'k_vehicles')
        solver.Add(solver.Sum(x[0, j] for j in customers) == k_vehicles)
        solver.Add(solver.Sum(x[j, 0] for j in customers) == k_vehicles)

        # 2. Time constraints (MTZ-like for TW)
        # t[j] >= t[i] + service[i] + travel[i,j] - M*(1-x[i,j])
        M = horizon
        
        service_times = self.evaluator.service_times
        def get_service(node_idx):
            if isinstance(service_times, (int, float)):
                return service_times
            if service_times is None:
                return 0
            return service_times[node_idx]

        for i in all_nodes:
            for j in all_nodes:
                if i == j: continue
                
                # travel time
                tt = dist_matrix[i, j]
                s_i = get_service(i)
                
                # if x[i,j]=1 => t[j] >= t[i] + s_i + tt
                # t[j] >= t[i] + s_i + tt - M(1 - x[i,j])
                solver.Add(t[j] >= t[i] + s_i + tt - M * (1 - x[i, j]))

        # 3. Time Windows
        if self.evaluator.time_windows is not None:
            for i in all_nodes:
                early, late = self.evaluator.time_windows[i]
                solver.Add(t[i] >= early)
                solver.Add(t[i] <= late)

        # Objective
        # Cost = 200 * Vehicles + 20 * Total Duration (Hours)
        # Total Duration is hard to sum directly in this flow formulation without vehicle tracking.
        # Approximation: Sum of travel times + Waiting times?
        # Actually, Cost is calculated per route.
        # In this formulation, we can minimize total travel time + fixed cost.
        # Total Time = Sum (Arrival at Depot - Start at Depot) for each vehicle.
        # This is hard because "Start at Depot" is not unique variable.
        
        # Simplified Objective: Minimize Total Travel Time + Fixed Vehicle Cost
        # This is a proxy for the actual cost function.
        
        obj_expr = solver.Sum(x[i, j] * dist_matrix[i, j] for i in all_nodes for j in all_nodes if i != j)
        obj_expr += k_vehicles * 36000 # 200 * 180 (scaling factor from ortools_solver)
        
        solver.Minimize(obj_expr)

        # Solve
        status = solver.Solve()

        if status == pywraplp.Solver.OPTIMAL or status == pywraplp.Solver.FEASIBLE:
            # Reconstruct routes
            routes = []
            visited = set()
            
            # Find all outgoing from depot
            next_nodes = {}
            for i in all_nodes:
                for j in all_nodes:
                    if i != j and x[i, j].solution_value() > 0.5:
                        next_nodes[i] = j
            
            # Build routes
            # Start from depot (0)
            # Since depot can have multiple outgoing, we need to handle it.
            # But in this formulation with single node 0, next_nodes[0] can only be one value if we use dict.
            # Wait, if we use single node 0, we can't have multiple routes departing 0 in a simple dict map!
            # Correct: For VRP, we usually duplicate depot or use a list of next nodes.
            
            # Correction: We need to iterate all edges from 0
            depot_nexts = []
            for j in customers:
                if x[0, j].solution_value() > 0.5:
                    depot_nexts.append(j)
            
            for start_node in depot_nexts:
                route = []
                curr = start_node
                while curr != 0:
                    route.append(curr)
                    if curr in next_nodes:
                        curr = next_nodes[curr]
                    else:
                        break # Should not happen
                routes.append(route)
            
            # Calculate actual cost using evaluator
            final_cost = self.evaluator.calculate_cost(routes)
            return routes, final_cost
        else:
            return [], float('inf')
