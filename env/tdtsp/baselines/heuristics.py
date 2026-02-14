import random

class GreedyRandomized:
    def __init__(self, evaluator, k=3):
        self.evaluator = evaluator
        self.k = k
        self.num_nodes = evaluator.num_nodes

    def solve(self, num_iterations=100):
        best_cost = float('inf')
        best_tour = None
        
        for _ in range(num_iterations):
            tour, cost = self._generate_tour()
            if cost < best_cost:
                best_cost = cost
                best_tour = tour
    
        return best_tour, best_cost

    def _generate_tour(self):
        unvisited = set(range(1, self.num_nodes))
        current_node = 0
        tour = [0]
        current_time = self.evaluator.start_time
        
        # Check start node TW
        if self.evaluator.time_windows is not None:
            early, late = self.evaluator.time_windows[0]
            current_time = max(current_time, early)
        
        while unvisited:
            # Calculate costs to all unvisited neighbors
            candidates = []
            for neighbor in unvisited:
                # Add service time if applicable (service at current node)
                arrival_time = current_time
                if self.evaluator.service_times and len(tour) > 0:
                     arrival_time += self.evaluator.service_times[current_node]

                tt = self.evaluator._get_travel_time(current_node, neighbor, arrival_time)
                arrival_at_next = arrival_time + tt
                
                # Check TW
                early, late = self.evaluator.time_windows[neighbor]
                wait_time = 0
                if arrival_at_next < early:
                    wait_time = early - arrival_at_next
                    arrival_at_next = early
                
                # We no longer filter by 'valid' (arrival_at_next > late)
                # because the evaluator handles penalties.
                candidates.append((neighbor, arrival_at_next, wait_time))
            
            if not candidates:
                # This should not happen now since we don't filter
                return tour, float('inf')
                
            # Sort by arrival time (Greedy)
            candidates.sort(key=lambda x: x[1])
            
            # Select from top-k
            k = min(self.k, len(candidates))
            selected_idx = random.randint(0, k-1)
            next_node, next_arrival, _ = candidates[selected_idx]
            
            # Update
            tour.append(next_node)
            unvisited.remove(next_node)
            current_time = next_arrival
            current_node = next_node
            
        # Return to depot
        # Add service at last node
        if self.evaluator.service_times:
            current_time += self.evaluator.service_times[current_node]
            
        tt = self.evaluator._get_travel_time(current_node, 0, current_time)
        current_time += tt
        
        # Check depot return TW (Evaluator will handle penalty, so we don't return inf)
        if self.evaluator.time_windows is not None:
            early, late = self.evaluator.time_windows[0]
            current_time = max(current_time, early)
        
        return tour, current_time
