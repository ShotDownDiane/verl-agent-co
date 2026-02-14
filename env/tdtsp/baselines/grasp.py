
import random
import copy
import time

class TDTSPGRASPSolver:
    def __init__(self, evaluator, max_iterations=100, alpha=0.1, time_limit=None):
        self.evaluator = evaluator
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.time_limit = time_limit
        self.num_nodes = evaluator.num_nodes

    def solve(self):
        best_cost = float('inf')
        best_tour = None
        start_time = time.time()

        for i in range(self.max_iterations):
            if self.time_limit and (time.time() - start_time) > self.time_limit:
                break

            # 1. Construction Phase
            tour = self._construct_solution()
            
            # 2. Local Search Phase (2-opt)
            tour = self._local_search(tour)
            
            # Evaluate
            cost = self.evaluator.calculate_cost(tour)
            
            if cost < best_cost:
                best_cost = cost
                best_tour = tour

        return best_tour, best_cost

    def _construct_solution(self):
        unvisited = set(range(1, self.num_nodes))
        current_node = 0
        tour = [0]
        current_time = self.evaluator.start_time
        
        if self.evaluator.time_windows is not None:
            early, late = self.evaluator.time_windows[0]
            current_time = max(current_time, early)
            
        while unvisited:
            candidates = []
            
            # Evaluate all unvisited candidates
            for neighbor in unvisited:
                # Calculate arrival time
                temp_time = current_time
                if self.evaluator.service_times and len(tour) > 0:
                    temp_time += self.evaluator.service_times[current_node]
                
                tt = self.evaluator._get_travel_time(current_node, neighbor, temp_time)
                arrival_time = temp_time + tt
                
                # Check TW
                early, late = self.evaluator.time_windows[neighbor]
                wait_time = max(0, early - arrival_time)
                arrival_time = max(arrival_time, early)
                violation = max(0, arrival_time - late)
                
                # Cost heuristic: Arrival Time + Penalty * Violation
                # We can use the evaluator's penalty logic implicitly or explicitly
                # Here we use a simple greedy criterion: arrival time + penalty
                cost = arrival_time + self.evaluator.penalty_value * violation
                
                candidates.append((neighbor, cost, arrival_time))
            
            if not candidates:
                break
                
            # Sort by cost
            candidates.sort(key=lambda x: x[1])
            
            # RCL (Restricted Candidate List)
            min_cost = candidates[0][1]
            max_cost = candidates[-1][1]
            threshold = min_cost + self.alpha * (max_cost - min_cost)
            
            rcl = [c for c in candidates if c[1] <= threshold]
            
            # Select random from RCL
            selected = random.choice(rcl)
            next_node, _, next_arrival = selected
            
            tour.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
            current_time = next_arrival
            
        return tour

    def _local_search(self, tour):
        # Simple 2-opt implementation for TDTSP
        # Note: In TDTSP, reversing a segment changes travel times downstream,
        # so delta evaluation is expensive (O(N)). Full 2-opt is O(N^3).
        # We'll do a first-improvement or limited best-improvement.
        
        improved = True
        best_tour = list(tour)
        best_cost = self.evaluator.calculate_cost(best_tour)
        
        # Limit local search passes to avoid excessive runtime
        passes = 0
        max_passes = 5 
        
        while improved and passes < max_passes:
            improved = False
            passes += 1
            
            n = len(tour)
            # Only swap inner nodes, keep depot (0) at start
            # tour indices: 0 (depot), 1..n-1 (customers)
            
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    # Create new tour with reversed segment
                    new_tour = best_tour[:i] + best_tour[i:j+1][::-1] + best_tour[j+1:]
                    
                    new_cost = self.evaluator.calculate_cost(new_tour)
                    
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_tour = new_tour
                        improved = True
                        break # First improvement
                if improved:
                    break
                    
        return best_tour
