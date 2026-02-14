import numpy as np

class ACO:
    def __init__(self, evaluator, num_ants=20, alpha=1.0, beta=2.0, rho=0.1, iterations=100):
        self.evaluator = evaluator
        self.num_ants = num_ants
        self.alpha = alpha # Pheromone weight
        self.beta = beta   # Heuristic weight
        self.rho = rho     # Evaporation rate
        self.iterations = iterations
        self.num_nodes = evaluator.num_nodes
        
        # Pheromone matrix
        self.tau = np.ones((self.num_nodes, self.num_nodes)) * 0.1

    def solve(self):
        best_tour = None
        best_cost = float('inf')
        
        for it in range(self.iterations):
            tours = []
            costs = []
            
            for _ in range(self.num_ants):
                tour, cost = self._construct_solution()
                tours.append(tour)
                costs.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_tour = tour
            
            # Evaporation
            self.tau *= (1 - self.rho)
            
            # Update Pheromones (Global Best + Iteration Best?)
            # Standard AS: Update all ants
            for tour, cost in zip(tours, costs):
                if cost == float('inf'): continue
                delta = 1.0 / cost
                for i in range(len(tour) - 1):
                    u, v = tour[i], tour[i+1]
                    self.tau[u, v] += delta
                    # self.tau[v, u] += delta # Directed
                # Return to depot
                u, v = tour[-1], tour[0]
                self.tau[u, v] += delta
                
        return best_tour, best_cost

    def _construct_solution(self):
        unvisited = set(range(1, self.num_nodes))
        current_node = 0
        tour = [0]
        current_time = self.evaluator.start_time
        
        # Start node TW
        if self.evaluator.time_windows is not None:
            early, late = self.evaluator.time_windows[0]
            current_time = max(current_time, early)
        
        while unvisited:
            # Calculate probabilities
            probs = []
            nodes = list(unvisited)
            
            valid_nodes = []
            valid_probs = []
            
            for neighbor in nodes:
                # Heuristic: 1 / travel_time
                tt = self.evaluator._get_travel_time(current_node, neighbor, current_time)
                eta = 1.0 / (tt + 1e-6)
                tau = self.tau[current_node, neighbor]
                
                # We no longer filter by 'valid' (arrival_at_next > late)
                # because the evaluator handles penalties.
                prob = (tau ** self.alpha) * (eta ** self.beta)
                valid_nodes.append(neighbor)
                valid_probs.append(prob)
                
            if not valid_nodes:
                # This should not happen now since we don't filter
                return tour, float('inf')
            
            probs = np.array(valid_probs)
            if probs.sum() == 0:
                probs = np.ones_like(probs)
            probs = probs / probs.sum()
            
            # Select next node
            next_node = np.random.choice(valid_nodes, p=probs)
            
            # Update
            tour.append(next_node)
            unvisited.remove(next_node)
            tt = self.evaluator._get_travel_time(current_node, next_node, current_time)
            
            if self.evaluator.time_windows is not None:
                early, _ = self.evaluator.time_windows[next_node]
                current_time = max(current_time + tt, early)
            else:
                current_time += tt
                
            current_node = next_node
            
        # Return to tour construction result
        return tour, self.evaluator.calculate_cost(tour)
