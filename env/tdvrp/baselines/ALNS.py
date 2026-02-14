import random
import math
import copy
import numpy as np

class ALNS:
    def __init__(self, evaluator, max_iter=1000, initial_temp=100.0, cooling_rate=0.995, 
                 reaction_factor=0.1, segment_iter=100, min_destroy=None, max_destroy=None):
        self.evaluator = evaluator
        self.max_iter = max_iter
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.reaction_factor = reaction_factor # rho for weight smoothing
        self.segment_iter = segment_iter
        
        self.num_nodes = evaluator.num_nodes
        # Default destroy limits
        if min_destroy is None:
            self.min_destroy = 2
        else:
            self.min_destroy = min_destroy
            
        if max_destroy is None:
            self.max_destroy = max(4, int(0.3 * (self.num_nodes - 1))) # Up to 30% of customers
        else:
            self.max_destroy = max_destroy
            
        # Ensure bounds (customers are 1..N-1)
        num_customers = self.num_nodes - 1
        # Handle small instance case
        if num_customers < 2:
            self.min_destroy = 0
            self.max_destroy = 0
        else:
            self.min_destroy = min(self.min_destroy, num_customers)
            self.max_destroy = min(self.max_destroy, num_customers)
            if self.min_destroy < 1: self.min_destroy = 1
            if self.max_destroy < self.min_destroy: self.max_destroy = self.min_destroy

        # Operators
        self.destroy_ops = [self._random_removal, self._worst_removal]
        self.repair_ops = [self._greedy_insertion, self._regret_insertion]
        
        # Weights and Probabilities
        self.d_weights = [1.0] * len(self.destroy_ops)
        self.r_weights = [1.0] * len(self.repair_ops)
        self.d_probs = self._calc_probs(self.d_weights)
        self.r_probs = self._calc_probs(self.r_weights)
        
        # Scores (accumulated within segment)
        self.d_scores = [0.0] * len(self.destroy_ops)
        self.r_scores = [0.0] * len(self.repair_ops)
        self.d_counts = [0] * len(self.destroy_ops)
        self.r_counts = [0] * len(self.repair_ops)

        # Rewards
        self.sigma1 = 10 # New global best
        self.sigma2 = 5  # Better than current
        self.sigma3 = 2  # Accepted
        
    def solve(self, initial_solution=None):
        # 1. Initial Solution
        if initial_solution is None:
            # Simple initialization: one route per customer
            current_solution = [[0, i, 0] for i in range(1, self.num_nodes)]
        else:
            current_solution = copy.deepcopy(initial_solution)
            
        current_cost = self.evaluator.calculate_cost(current_solution)
        
        best_solution = copy.deepcopy(current_solution)
        best_cost = current_cost
        
        temp = self.initial_temp
        
        for it in range(self.max_iter):
            # Select operators
            d_idx = self._select_op(self.d_probs)
            r_idx = self._select_op(self.r_probs)
            d_op = self.destroy_ops[d_idx]
            r_op = self.repair_ops[r_idx]
            
            # Determine how many to remove
            if self.max_destroy > 0:
                num_remove = random.randint(self.min_destroy, self.max_destroy)
            else:
                num_remove = 0

            if num_remove > 0:
                # Apply Destroy
                partial_solution, removed_nodes = d_op(current_solution, num_remove)
                
                # Apply Repair
                new_solution = r_op(partial_solution, removed_nodes)
            else:
                new_solution = copy.deepcopy(current_solution)
                
            new_cost = self.evaluator.calculate_cost(new_solution)
            
            # Acceptance & Scoring
            delta = new_cost - current_cost
            accepted = False
            score = 0
            
            if delta < 0:
                accepted = True
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_solution = copy.deepcopy(new_solution)
                    score = self.sigma1
                else:
                    score = self.sigma2
            else:
                if random.random() < math.exp(-delta / temp):
                    accepted = True
                    score = self.sigma3
                else:
                    accepted = False
                    score = 0
            
            if accepted:
                current_solution = new_solution
                current_cost = new_cost
                
            # Update scores
            self.d_scores[d_idx] += score
            self.r_scores[r_idx] += score
            self.d_counts[d_idx] += 1
            self.r_counts[r_idx] += 1
            
            # Update Weights at segment end
            if (it + 1) % self.segment_iter == 0:
                self._update_weights()
                
            # Cooling
            temp *= self.cooling_rate
            
        return best_solution, best_cost

    def _calc_probs(self, weights):
        total = sum(weights)
        if total == 0: return [1.0/len(weights)] * len(weights)
        return [w / total for w in weights]

    def _select_op(self, probs):
        r = random.random()
        cum = 0.0
        for i, p in enumerate(probs):
            cum += p
            if r < cum:
                return i
        return len(probs) - 1

    def _update_weights(self):
        for i in range(len(self.d_weights)):
            if self.d_counts[i] > 0:
                avg_score = self.d_scores[i] / self.d_counts[i]
                self.d_weights[i] = (1 - self.reaction_factor) * self.d_weights[i] + self.reaction_factor * avg_score
            
            # Reset
            self.d_scores[i] = 0
            self.d_counts[i] = 0
            
        for i in range(len(self.r_weights)):
            if self.r_counts[i] > 0:
                avg_score = self.r_scores[i] / self.r_counts[i]
                self.r_weights[i] = (1 - self.reaction_factor) * self.r_weights[i] + self.reaction_factor * avg_score
            
            # Reset
            self.r_scores[i] = 0
            self.r_counts[i] = 0
            
        self.d_probs = self._calc_probs(self.d_weights)
        self.r_probs = self._calc_probs(self.r_weights)

    # --- Destroy Operators ---
    
    def _random_removal(self, solution, n):
        # Flatten customers from routes
        # solution is [[0, c1, 0], [0, c2, c3, 0]]
        # Map (route_idx, node_idx) for all customers
        customers_map = []
        for r_idx, route in enumerate(solution):
            for i in range(1, len(route) - 1):
                customers_map.append((r_idx, i, route[i]))
        
        if n > len(customers_map):
            n = len(customers_map)
            
        removed_entries = random.sample(customers_map, n)
        # Sort by route_idx desc, then node_idx desc to pop safely
        removed_entries.sort(key=lambda x: (x[0], x[1]), reverse=True)
        
        new_solution = copy.deepcopy(solution)
        removed_nodes = []
        
        for r_idx, i, node in removed_entries:
            removed_nodes.append(node)
            new_solution[r_idx].pop(i)
        
        # Clean up empty routes (length 2: [0, 0])
        new_solution = [r for r in new_solution if len(r) > 2]
        
        return new_solution, removed_nodes

    def _worst_removal(self, solution, n):
        # Remove nodes that contribute most to cost.
        # We perform full global evaluation (calculate_cost) for each candidate.
        
        current_sol = copy.deepcopy(solution)
        removed_nodes = []
        
        # Optimization: Limit n to avoid infinite loops if solution is small
        total_customers = sum(len(r) - 2 for r in current_sol)
        n = min(n, total_customers)
        
        for _ in range(n):
            max_save = -float('inf')
            target_info = None # (r_idx, node_idx, node)
            
            base_cost = self.evaluator.calculate_cost(current_sol)
            
            # Iterate over all customers in all routes
            for r_idx, route in enumerate(current_sol):
                if len(route) <= 2: continue
                
                for i in range(1, len(route) - 1):
                    # Try removing
                    temp_sol = copy.deepcopy(current_sol)
                    temp_sol[r_idx].pop(i)
                    if len(temp_sol[r_idx]) == 2:
                        temp_sol.pop(r_idx)
                        
                    cost = self.evaluator.calculate_cost(temp_sol)
                    save = base_cost - cost
                    
                    if save > max_save:
                        max_save = save
                        target_info = (r_idx, i, route[i])
            
            if target_info:
                r_idx, i, node = target_info
                removed_nodes.append(node)
                current_sol[r_idx].pop(i)
                if len(current_sol[r_idx]) == 2:
                    current_sol.pop(r_idx)
            else:
                break
                
        return current_sol, removed_nodes

    # --- Repair Operators ---
    
    def _greedy_insertion(self, partial_solution, removed_nodes):
        # Insert removed nodes one by one into best position
        
        curr_sol = copy.deepcopy(partial_solution)
        to_insert = list(removed_nodes)
        random.shuffle(to_insert)
        
        for node in to_insert:
            best_cost = float('inf')
            best_sol = None
            
            # 1. Try existing routes
            for r_idx, route in enumerate(curr_sol):
                for i in range(1, len(route)): # Insert at i (between i-1 and i)
                    temp_sol = copy.deepcopy(curr_sol)
                    temp_sol[r_idx].insert(i, node)
                    
                    cost = self.evaluator.calculate_cost(temp_sol)
                    if cost < best_cost:
                        best_cost = cost
                        best_sol = temp_sol
            
            # 2. Try new route
            temp_sol = copy.deepcopy(curr_sol)
            temp_sol.append([0, node, 0])
            cost = self.evaluator.calculate_cost(temp_sol)
            if cost < best_cost:
                best_cost = cost
                best_sol = temp_sol
            
            if best_sol is not None:
                curr_sol = best_sol
            else:
                curr_sol.append([0, node, 0])
                
        return curr_sol

    def _regret_insertion(self, partial_solution, removed_nodes):
        # Regret-2
        curr_sol = copy.deepcopy(partial_solution)
        to_insert = list(removed_nodes)
        
        while to_insert:
            candidates = []
            
            for node in to_insert:
                costs = []
                # Try all positions in existing routes
                for r_idx, route in enumerate(curr_sol):
                    for i in range(1, len(route)):
                        temp_sol = copy.deepcopy(curr_sol)
                        temp_sol[r_idx].insert(i, node)
                        cost = self.evaluator.calculate_cost(temp_sol)
                        costs.append((cost, temp_sol))
                
                # Try new route
                temp_sol = copy.deepcopy(curr_sol)
                temp_sol.append([0, node, 0])
                cost = self.evaluator.calculate_cost(temp_sol)
                costs.append((cost, temp_sol))
                
                # Sort costs
                costs.sort(key=lambda x: x[0])
                
                best_c, best_s = costs[0]
                if len(costs) > 1:
                    second_c = costs[1][0]
                else:
                    second_c = float('inf')
                    
                regret = second_c - best_c
                candidates.append((regret, node, best_s))
            
            # Pick max regret
            candidates.sort(key=lambda x: x[0], reverse=True)
            if not candidates:
                 break
                 
            _, best_node, best_sol = candidates[0]
            
            curr_sol = best_sol
            to_insert.remove(best_node)
            
        return curr_sol

if __name__ == "__main__":
    pass
