import random
import time
import copy
import numpy as np

class TDVRPGRASPSolver:
    def __init__(self, evaluator, max_iterations=100, alpha=0.1, time_limit=None):
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.time_limit = time_limit
        self.num_nodes = evaluator.num_nodes

    def solve(self):
        best_cost = float('inf')
        best_routes = None
        start_time = time.time()

        for i in range(self.max_iterations):
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break

            # 1. Construction Phase
            routes = self._construct_solution()
            
            # 2. Local Search Phase
            routes = self._local_search(routes)
            
            # Evaluate
            cost = self.evaluator.calculate_cost(routes)
            
            if cost < best_cost:
                best_cost = cost
                best_routes = copy.deepcopy(routes)

        return best_routes, best_cost

    def _construct_solution(self):
        unvisited = set(range(1, self.num_nodes))
        routes = []
        
        print(f"DEBUG: Starting construction. Unvisited: {len(unvisited)}")
        
        while unvisited:
            route = []
            current_node = 0
            current_time = self.evaluator.start_time
            
            # Loop to build one route
            while unvisited:
                candidates = []
                for neighbor in sorted(unvisited):
                    # Calculate arrival time
                    tt = self.evaluator._get_travel_time(current_node, neighbor, current_time)
                    arrival_at_next = current_time + tt
                    
                    # Heuristic Score: arrival time + penalties
                    score = arrival_at_next
                    
                    if self.evaluator.time_windows is not None:
                        early, late = self.evaluator.time_windows[neighbor]
                        wait = max(0, early - arrival_at_next)
                        arrival_at_next = max(arrival_at_next, early)
                        violation = max(0, arrival_at_next - late)
                        
                        # Penalize violation heavily in construction
                        score = arrival_at_next + violation * 1000 + wait * 0.1
                        
                    candidates.append((neighbor, arrival_at_next, score))
                
                if not candidates:
                    print("DEBUG: No candidates!")
                    break
                    
                # Sort by score
                candidates.sort(key=lambda x: x[2])
                
                # RCL
                min_score = candidates[0][2]
                max_score = candidates[-1][2]
                threshold = min_score + self.alpha * (max_score - min_score)
                
                rcl = [c for c in candidates if c[2] <= threshold]
                
                # Select from RCL
                selected = random.choice(rcl)
                next_node, next_arrival, _ = selected
                
                # Feasibility check: check if adding this node makes the WHOLE route (including return to depot) late
                temp_route = route + [next_node]
                res = self.evaluator.evaluate_route(temp_route)
                
                # If adding this node causes significant violation, start new route
                # We use a strict check here to encourage feasible routes
                if res["violation_sec"] > 0 and len(route) > 0:
                    print(f"DEBUG: Violation for node {next_node}. Violation: {res['violation_sec']}. Breaking route len {len(route)}")
                    break
                
                # Add to route
                route.append(next_node)
                unvisited.remove(next_node)
                current_node = next_node
                
                # Update current_time for next step
                if isinstance(self.evaluator.service_times, (int, float)):
                     current_time = next_arrival + self.evaluator.service_times
                elif self.evaluator.service_times is not None:
                     current_time = next_arrival + self.evaluator.service_times[next_node]
                else:
                     current_time = next_arrival
            
            if route:
                routes.append(route)
                print(f"DEBUG: Added route len {len(route)}. Remaining unvisited: {len(unvisited)}")
            else:
                # Force add one node if we couldn't add any (to prevent infinite loop)
                if unvisited:
                    node = list(unvisited)[0]
                    print(f"DEBUG: Force adding node {node}")
                    routes.append([node])
                    unvisited.remove(node)
        
        print(f"DEBUG: Construction finished. Total routes: {len(routes)}")     
        return routes

    def _local_search(self, routes):
        # Intra-route 2-opt
        # Since routes are independent in cost summation, we can optimize each separately
        optimized_routes = []
        for route in routes:
            optimized_route = self._optimize_route_2opt(route)
            optimized_routes.append(optimized_route)
            
        return optimized_routes

    def _optimize_route_2opt(self, route):
        # Simple 2-opt for a single route
        if len(route) < 3:
            return route
            
        best_route = list(route)
        # Calculate initial cost for this single route
        best_cost = self.evaluator.calculate_cost([best_route])
        
        improved = True
        passes = 0
        max_passes = 5 # Limit passes to avoid excessive runtime
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            n = len(best_route)
            
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if i == 0 and j == n - 1: continue 
                    
                    # 2-opt swap: reverse segment from i to j
                    new_route = best_route[:i] + best_route[i:j+1][::-1] + best_route[j+1:]
                    
                    new_cost = self.evaluator.calculate_cost([new_route])
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_route = new_route
                        improved = True
                        break # First improvement
                if improved:
                    break
        
        return best_route
