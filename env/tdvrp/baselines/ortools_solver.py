
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np

def solve_ortools(matrix, duration, time_windows, service_times, config=None):
    """
    Solves TDVRP using OR-Tools with static approximation (mean travel time).
    """
    if config is None:
        config = {}
    
    time_limit = config.get("time_limit", 5)
    
    # 1. Prepare Data
    num_nodes = matrix.shape[0]
    num_vehicles = num_nodes  # Upper bound, let solver minimize used vehicles
    depot = 0
    
    # Static approximation: mean over time
    # matrix shape: (N, N, T)
    static_matrix = matrix.mean(axis=2).astype(int)
    
    # Service times
    if isinstance(service_times, (int, float)):
        service_times_arr = [int(service_times)] * num_nodes
    else:
        service_times_arr = [int(x) for x in service_times]
    
    # Ensure service time at depot is 0
    service_times_arr[depot] = 0
        
    # Time windows
    # time_windows shape: (N, 2)
    time_windows_int = time_windows.astype(int).tolist()
    
    # 2. Create Routing Index Manager
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)
    
    # 3. Create Routing Model
    routing = pywrapcp.RoutingModel(manager)
    
    # 4. Define Transit Callback
    def time_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        # Transit time = Service time at 'from' + Travel time 'from'->'to'
        return static_matrix[from_node][to_node] + service_times_arr[from_node]
        
    transit_callback_index = routing.RegisterTransitCallback(time_callback)
    
    # 5. Add Time Dimension
    # Horizon: Determine max time from time windows or a large number
    # Typically TW for depot is [0, max_time]
    horizon = int(time_windows_int[0][1] * 2) 
    if horizon > 24 * 3600 * 10: # Cap if unreasonably large
        horizon = 24 * 3600 * 10
        
    routing.AddDimension(
        transit_callback_index,
        horizon, # Allow waiting time (slack) up to horizon
        horizon, # Max time per vehicle
        False,   # Don't force start cumul to zero
        "Time"
    )
    time_dimension = routing.GetDimensionOrDie("Time")
    
    # 6. Add Time Window Constraints
    for i in range(num_nodes):
        index = manager.NodeToIndex(i)
        start, end = time_windows_int[i]
        if end > horizon:
            end = horizon
        time_dimension.CumulVar(index).SetRange(start, end)
        
    # 7. Add Cost
    # Objective: Minimize (200 * num_vehicles + 20 * total_hours)
    # Equivalent to: Minimize (36000 * num_vehicles + 1 * total_seconds)
    # (Scaling by 1800 to keep integers: 200*180 = 36000, 20/3600 * 1800 = 10... Wait)
    # 20/3600 = 1/180.
    # So multiply by 180:
    # Cost_V = 200 * 180 = 36000
    # Cost_T = (20/3600) * 180 = 1
    
    routing.SetFixedCostOfAllVehicles(36000)
    time_dimension.SetSpanCostCoefficientForAllVehicles(1)
    
    # 8. Search Parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = time_limit
    # search_parameters.log_search = True
    
    # 9. Solve
    solution = routing.SolveWithParameters(search_parameters)
    
    # 10. Extract Routes
    routes = []
    total_cost = 0
    
    if solution:
        for vehicle_id in range(num_vehicles):
            if routing.IsVehicleUsed(solution, vehicle_id):
                index = routing.Start(vehicle_id)
                route = []
                # Skip the start node (depot)
                index = solution.Value(routing.NextVar(index))
                
                while not routing.IsEnd(index):
                    node_index = manager.IndexToNode(index)
                    route.append(node_index)
                    index = solution.Value(routing.NextVar(index))
                
                if route:
                    routes.append(route)
                    
        total_cost = solution.ObjectiveValue()
    
    return routes, total_cost
