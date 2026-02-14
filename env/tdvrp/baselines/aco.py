import numpy as np

class ACO:
    def __init__(self, evaluator, num_ants=20, num_iterations=100, alpha=1.0, beta=2.0, rho=0.1, q=100):
        self.evaluator = evaluator
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.num_nodes = evaluator.num_nodes
        self.pheromone = np.ones((self.num_nodes, self.num_nodes)) * 0.1
        self.customers = [i for i in range(1, self.num_nodes)]
        
    def solve(self):
        best_cost = float('inf')
        best_solution = None
        
        for _ in range(self.num_iterations):
            ant_solutions = []
            ant_costs = []
            
            # Generate ant solutions (Giant Tours)
            for _ in range(self.num_ants):
                tour = self._construct_tour()
                
                # Safety check: ensure tour is complete
                if len(tour) != len(self.customers):
                    # This should rarely happen, but if it does, fill missing
                    missing = list(set(self.customers) - set(tour))
                    tour.extend(missing)

                routes = self.evaluator.split_tour(tour)
                cost = self.evaluator.calculate_cost(routes)
                ant_solutions.append(tour)
                ant_costs.append(cost)
                
                if cost < best_cost:
                    best_cost = cost
                    best_solution = routes
            
            # Update Pheromones
            self.pheromone *= (1 - self.rho)
            for tour, cost in zip(ant_solutions, ant_costs):
                delta = self.q / (cost + 1e-6) # Heuristic update
                # Add pheromone to edges in Giant Tour
                # Edge 0 -> tour[0]
                self.pheromone[0, tour[0]] += delta
                self.pheromone[tour[0], 0] += delta
                
                for i in range(len(tour) - 1):
                    u, v = tour[i], tour[i+1]
                    self.pheromone[u, v] += delta
                    self.pheromone[v, u] += delta
                    
        return best_solution, best_cost

    def _construct_tour(self):
        unvisited = set(self.customers)
        current_node = 0
        tour = []
        
        # We need a reference time for heuristic information
        # Just use start time
        ref_time = self.evaluator.start_time
        
        while unvisited:
            candidates = list(unvisited)
            probs = []
            
            for neighbor in candidates:
                tau = self.pheromone[current_node, neighbor] ** self.alpha
                # Heuristic: 1 / travel_time
                tt = self.evaluator._get_travel_time(current_node, neighbor, ref_time)
                eta = (1.0 / (tt + 1e-6)) ** self.beta
                probs.append(tau * eta)
                
            probs = np.array(probs)
            
            # Handle numerical issues
            if np.isnan(probs).any() or probs.sum() == 0:
                probs = np.ones(len(probs)) / len(probs)
            else:
                probs /= probs.sum()
            
            next_node = np.random.choice(candidates, p=probs)
            tour.append(next_node)
            unvisited.remove(next_node)
            current_node = next_node
            
        return tour
