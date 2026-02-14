
import numpy as np

class TDVRPEvaluator:
    def __init__(self, matrix, duration, time_windows=None, service_times=None, start_time=0.0, penalty_value=3.0):
        """
        matrix: [N, N, T] numpy array (travel times in seconds)
        duration: float (duration of one time step in seconds)
        time_windows: [N, 2] numpy array (relative early/late in seconds)
        service_times: [N] numpy array or float
        start_time: float (relative to 0, usually 0.0)
        penalty_value: float
        """
        self.matrix = matrix
        self.num_nodes = matrix.shape[0]
        self.num_steps = matrix.shape[2]
        self.duration = duration
        self.start_time = start_time
        self.max_s = self.num_steps - 1
        self.time_windows = time_windows
        self.service_times = service_times
        self.penalty_value = penalty_value
        self.was_late = False

        # VRP Specific Costs
        self.COST_PER_VEHICLE = 200.0
        self.COST_PER_HOUR = 20.0

    def calculate_cost(self, routes, return_details=False, late_tolerance=1.0):
        """
        routes: List[List[int]] - Each list is a route for one vehicle.
        """
        # Validity Check
        visited_nodes = []
        for r in routes:
            visited_nodes.extend(r)
        
        visited_set = set(visited_nodes)
        # Customers are 1 to N-1 (0 is depot)
        expected_nodes = set(range(1, self.num_nodes))
        
        missing = expected_nodes - visited_set
        unexpected = visited_set - expected_nodes
        
        is_valid_tour = True
        if missing:
            print(f"Warning: Missing customers: {missing}")
            is_valid_tour = False
        if unexpected:
            print(f"Warning: Unexpected nodes in routes (e.g. depot 0 or out of bounds): {unexpected}")
            is_valid_tour = False
        if len(visited_nodes) != len(visited_set):
            print(f"Warning: Duplicate customers visited. Total visits: {len(visited_nodes)}, Unique: {len(visited_set)}")
            is_valid_tour = False

        total_cost = 0.0
        total_violation = 0.0
        self.was_late = False
        
        all_details = []

        for route_idx, route in enumerate(routes):
            if not route: continue
            
            res = self.evaluate_route(route, late_tolerance=late_tolerance)
            route_violation = res["violation_sec"]
            route_history = res["history"]
            current_time = res["end_time"]
            
            # Route Cost = 200 + 20 * total_hours
            route_duration_hours = (current_time - self.start_time) / 3600.0
            route_cost = self.COST_PER_VEHICLE + self.COST_PER_HOUR * route_duration_hours
            
            if route_violation > 0:
                self.was_late = True
                total_violation += route_violation
                penalty = (route_violation / 3600.0) * self.penalty_value
                # Apply clamping similar to training
                penalty = max(penalty, 0.5 * route_cost)
                penalty = min(penalty, 2.0 * route_cost)
                route_cost += penalty
            
            total_cost += route_cost
            if return_details:
                all_details.append({
                    "route_idx": route_idx,
                    "duration_hours": route_duration_hours,
                    "violation_sec": route_violation,
                    "cost": route_cost,
                    "history": route_history
                })

        if return_details:
            return total_cost, {
                "total_violation": total_violation, 
                "routes": all_details,
                "all_visited": is_valid_tour
            }
        return total_cost

    def evaluate_route(self, route, late_tolerance=1.0):
        """
        Evaluate a single route.
        """
        current_time = self.start_time
        current_node = 0
        route_violation = 0.0
        route_history = [{
            "node": 0,
            "arrival": current_time,
            "window": self.time_windows[0] if self.time_windows is not None else (0, float('inf')),
            "late": 0.0
        }]

        if self.time_windows is not None:
            early, late = self.time_windows[0]
            if current_time < early:
                current_time = early

        for next_node in route:
            tt = self._get_travel_time(current_node, next_node, current_time)
            current_time += tt
            
            node_late = 0.0
            if self.time_windows is not None:
                early, late = self.time_windows[next_node]
                if current_time < early:
                    current_time = early
                if current_time > late:
                    v = current_time - late
                    if v > late_tolerance:
                        route_violation += v
                        node_late = v
            
            route_history.append({
                "node": next_node,
                "arrival": current_time,
                "window": (early, late) if self.time_windows is not None else (0, float('inf')),
                "late": node_late
            })

            if isinstance(self.service_times, (int, float)):
                current_time += self.service_times
            elif self.service_times is not None:
                current_time += self.service_times[next_node]
            
            current_node = next_node

        # Return to depot
        tt = self._get_travel_time(current_node, 0, current_time)
        current_time += tt
        depot_late = 0.0
        if self.time_windows is not None:
            _, late = self.time_windows[0]
            if current_time > late:
                v = current_time - late
                if v > late_tolerance:
                    route_violation += v
                    depot_late = v
        
        route_history.append({
            "node": 0,
            "arrival": current_time,
            "window": (0, late) if self.time_windows is not None else (0, float('inf')),
            "late": depot_late
        })

        return {
            "violation_sec": route_violation,
            "history": route_history,
            "end_time": current_time
        }

    def split_tour(self, permutation, max_nodes_per_route=None):
        """
        Smart split: iterate through the permutation and start a new route if 
        adding the next node causes a violation or exceeds capacity.
        """
        routes = []
        if not permutation:
            return routes
            
        current_route = []
        for node in permutation:
            # Check if adding this node makes the current route late
            temp_route = current_route + [node]
            
            # Constraints:
            # 1. Capacity (optional)
            # 2. Time Window Violation
            
            should_split = False
            if max_nodes_per_route is not None and len(current_route) >= max_nodes_per_route:
                should_split = True
            elif len(current_route) > 0:
                res = self.evaluate_route(temp_route)
                if res["violation_sec"] > 0:
                    should_split = True
            
            if should_split:
                routes.append(current_route)
                current_route = [node]
            else:
                current_route.append(node)
        
        if current_route:
            routes.append(current_route)
            
        return routes

    def _get_travel_time(self, u, v, t):
        # t is relative time in seconds
        idx = int(t // self.duration)
        idx = max(0, min(idx, self.max_s))
        return self.matrix[u, v, idx]
