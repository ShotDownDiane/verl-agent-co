
import random
import math

class SimulatedAnnealing:
    def __init__(self, evaluator, initial_solution=None, initial_temp=1000, cooling_rate=0.995, max_iterations=10000):
        self.evaluator = evaluator
        self.initial_solution = initial_solution # permutation
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iterations = max_iterations
        self.customers = [i for i in range(1, evaluator.num_nodes)]
        
    def _calculate_tour_cost(self, tour):
        """Helper to split tour and calculate cost"""
        routes = self.evaluator.split_tour(tour)
        return self.evaluator.calculate_cost(routes)

    def solve(self):
        if self.initial_solution:
            current_tour = list(self.initial_solution)
        else:
            current_tour = list(self.customers)
            random.shuffle(current_tour)
            
        current_cost = self._calculate_tour_cost(current_tour)
        best_tour = list(current_tour)
        best_cost = current_cost
        
        temp = self.initial_temp
        
        for i in range(self.max_iterations):
            # Generate neighbor (Swap, Reverse, Insert)
            neighbor = list(current_tour)
            op = random.choice(['swap', 'reverse', 'insert'])
            idx1, idx2 = sorted(random.sample(range(len(neighbor)), 2))
            
            if op == 'swap':
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
            elif op == 'reverse':
                neighbor[idx1:idx2+1] = list(reversed(neighbor[idx1:idx2+1]))
            elif op == 'insert':
                val = neighbor.pop(idx1)
                neighbor.insert(idx2, val)
                
            neighbor_cost = self._calculate_tour_cost(neighbor)
            
            # Acceptance
            delta = neighbor_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / (temp + 1e-9)):
                current_tour = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = list(current_tour)
                    
            temp *= self.cooling_rate
            
        # Return final routes and cost
        routes = self.evaluator.split_tour(best_tour)
        return routes, best_cost
