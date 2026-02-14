
import random

class GreedyRandomized:
    def __init__(self, evaluator, k=3):
        self.evaluator = evaluator
        self.k = k
        self.num_nodes = evaluator.num_nodes

    def solve(self, num_iterations=100):
        best_cost = float('inf')
        best_routes = None
        
        for _ in range(num_iterations):
            routes = self._generate_dynamic_routes()
            cost, details = self.evaluator.calculate_cost(routes, return_details=True)
            
            if cost < best_cost:
                best_cost = cost
                best_routes = routes
    
        return best_routes, best_cost

    def _generate_dynamic_routes(self, max_nodes_per_route=15):
        unvisited = set(range(1, self.num_nodes))
        all_routes = []
        
        while unvisited:
            route = []
            current_node = 0
            current_time = self.evaluator.start_time
            
            # Start building a single route
            while unvisited and len(route) < max_nodes_per_route:
                candidates = []
                for neighbor in unvisited:
                    tt = self.evaluator._get_travel_time(current_node, neighbor, current_time)
                    arrival_at_next = current_time + tt
                    
                    # Heuristic Score: arrival time + potential penalty
                    score = arrival_at_next
                    if self.evaluator.time_windows is not None:
                        early, late = self.evaluator.time_windows[neighbor]
                        if arrival_at_next < early:
                            arrival_at_next = early
                        if arrival_at_next > late:
                            # Heavily penalize late arrival during construction
                            score += (arrival_at_next - late) * 10 

                    candidates.append((neighbor, arrival_at_next, score))
                
                # Sort by score (arrival + penalty)
                candidates.sort(key=lambda x: x[2])
                
                # Top-k selection
                k = min(self.k, len(candidates))
                selected_idx = random.randint(0, k-1)
                next_node, next_arrival, _ = candidates[selected_idx]
                
                # Check if adding this node makes the WHOLE route (including return to depot) late
                temp_route = route + [next_node]
                res = self.evaluator.evaluate_route(temp_route)
                
                if res["violation_sec"] > 0 and len(route) > 0:
                    # If this node causes violation, and we already have some nodes, 
                    # stop this route here and start a new vehicle.
                    break
                
                # Otherwise, commit the node
                route.append(next_node)
                unvisited.remove(next_node)
                
                # Update current time for the next step in THIS route
                if isinstance(self.evaluator.service_times, (int, float)):
                    current_time = next_arrival + self.evaluator.service_times
                elif self.evaluator.service_times is not None:
                    current_time = next_arrival + self.evaluator.service_times[next_node]
                else:
                    current_time = next_arrival
                current_node = next_node
                
                # If even without this node the route was already cutting it close, 
                # we might want to break early, but the violation check above handles it.

            if route:
                all_routes.append(route)
            else:
                # Fallback: if even a single node causes violation, we must take it
                # to ensure progress, otherwise we'll be stuck in an infinite loop.
                if unvisited:
                    node = list(unvisited)[0]
                    all_routes.append([node])
                    unvisited.remove(node)
            
        return all_routes

    def _generate_permutation(self, max_nodes_per_route=15):
        # This is kept for compatibility if needed, but solve now uses _generate_dynamic_routes
        return [node for route in self._generate_dynamic_routes(max_nodes_per_route) for node in route]
