import random
import math

class SimulatedAnnealing:
    def __init__(self, evaluator, initial_temp=100, cooling_rate=0.995, max_iter=5000, initial_solution=None):
        self.evaluator = evaluator
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.initial_solution = initial_solution
        
    def solve(self):
        # Initial solution
        if self.initial_solution is not None:
            current_solution = list(self.initial_solution)
        else:
            # Random initialization
            nodes = list(range(1, self.evaluator.num_nodes))
            random.shuffle(nodes)
            current_solution = [0] + nodes
            
        current_cost = self.evaluator.calculate_cost(current_solution)
        
        best_solution = list(current_solution)
        best_cost = current_cost
        
        temp = self.initial_temp
        
        for _ in range(self.max_iter):
            neighbor = self._get_neighbor(current_solution)
            neighbor_cost = self.evaluator.calculate_cost(neighbor)
            
            delta = neighbor_cost - current_cost
            
            if delta < 0 or random.random() < math.exp(-delta / temp):
                current_solution = neighbor
                current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_solution = list(current_solution)
            
            temp *= self.cooling_rate
        return best_solution, best_cost
        
    def _get_neighbor(self, tour):
        neighbor = list(tour)
        n = len(tour)
        if n <= 2: return neighbor
        
        # Operators: Swap, Reverse, Insert
        # Keep depot (0) fixed? The logic below assumes 0 is at index 0 and shouldn't move?
        # The indices 1..n-1 are mutable.
        op = random.choice(['swap', 'reverse', 'insert'])
        i, j = sorted(random.sample(range(1, n), 2))
        
        if op == 'swap':
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        elif op == 'reverse':
            neighbor[i:j+1] = list(reversed(neighbor[i:j+1]))
        elif op == 'insert':
            val = neighbor.pop(i)
            neighbor.insert(j-1, val)
            
        return neighbor
