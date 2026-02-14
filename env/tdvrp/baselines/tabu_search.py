
import random
from collections import deque

class TabuSearch:
    def __init__(self, evaluator, initial_solution=None, max_iterations=1000, tabu_tenure=20, num_neighbors=50):
        self.evaluator = evaluator
        self.initial_solution = initial_solution # permutation
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        self.num_neighbors = num_neighbors
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
        
        # Tabu list stores moves
        tabu_list = deque(maxlen=self.tabu_tenure)
        
        for i in range(self.max_iterations):
            best_neighbor = None
            best_neighbor_cost = float('inf')
            best_move = None
            
            # Generate multiple neighbors and pick the best non-tabu one (or aspiration)
            # To ensure we don't get stuck if all generated neighbors are tabu and non-improving (unlikely with random sampling),
            # we just take the best valid one found in this batch.
            
            for _ in range(self.num_neighbors):
                neighbor = list(current_tour)
                op = random.choice(['swap', 'reverse', 'insert'])
                idx1, idx2 = sorted(random.sample(range(len(neighbor)), 2))
                move = None
                
                if op == 'swap':
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                    # Signature: swap values at indices (independent of index if values are unique)
                    # But simpler to just use values swapped
                    v1, v2 = sorted((neighbor[idx1], neighbor[idx2]))
                    move = ('swap', v1, v2)
                    
                elif op == 'reverse':
                    neighbor[idx1:idx2+1] = list(reversed(neighbor[idx1:idx2+1]))
                    # Signature: reverse range
                    # This is harder to track by values, so we use indices approx or start/end values
                    move = ('reverse', neighbor[idx1], neighbor[idx2]) 
                    
                elif op == 'insert':
                    val = neighbor.pop(idx1)
                    neighbor.insert(idx2, val)
                    move = ('insert', val, idx2) # Insert val at idx2
                
                neighbor_cost = self._calculate_tour_cost(neighbor)
                
                # Check Tabu status
                is_tabu = move in tabu_list
                is_aspiration = neighbor_cost < best_cost
                
                if (not is_tabu) or is_aspiration:
                    if neighbor_cost < best_neighbor_cost:
                        best_neighbor = neighbor
                        best_neighbor_cost = neighbor_cost
                        best_move = move
            
            if best_neighbor is not None:
                current_tour = best_neighbor
                current_cost = best_neighbor_cost
                if best_move:
                    tabu_list.append(best_move)
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_tour = list(current_tour)
            
        # Return final routes and cost
        routes = self.evaluator.split_tour(best_tour)
        return routes, best_cost
