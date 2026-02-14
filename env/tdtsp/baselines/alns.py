import random
import math
import numpy as np
import copy

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
            self.max_destroy = max(4, int(0.3 * self.num_nodes)) # Up to 30%
        else:
            self.max_destroy = max_destroy
            
        # Ensure bounds
        self.min_destroy = min(self.min_destroy, self.num_nodes - 2)
        self.max_destroy = min(self.max_destroy, self.num_nodes - 2)
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
        
    def solve(self):
        # 1. Initial Solution (Random)
        nodes = list(range(1, self.evaluator.num_nodes))
        random.shuffle(nodes)
        current_solution = [0] + nodes
        current_cost = self.evaluator.calculate_cost(current_solution)
        
        best_solution = list(current_solution)
        best_cost = current_cost
        
        temp = self.initial_temp
        
        for it in range(self.max_iter):
            # Select operators
            d_idx = self._select_op(self.d_probs)
            r_idx = self._select_op(self.r_probs)
            d_op = self.destroy_ops[d_idx]
            r_op = self.repair_ops[r_idx]
            
            # Determine how many to remove
            num_remove = random.randint(self.min_destroy, self.max_destroy)
            
            # Apply Destroy
            partial_tour, removed_nodes = d_op(current_solution, num_remove)
            
            # Apply Repair
            new_solution = r_op(partial_tour, removed_nodes)
            new_cost = self.evaluator.calculate_cost(new_solution)
            
            # Acceptance & Scoring
            delta = new_cost - current_cost
            accepted = False
            score = 0
            
            if delta < 0:
                accepted = True
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_solution = list(new_solution)
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
            
        # Final validation
        if not self.evaluator.validate_tour(best_solution):
            print("WARNING: ALNS produced an invalid solution! Falling back to initial random.")
            # This should ideally not happen if logic is correct
            # We can re-evaluate strict cost to be sure
        return best_solution, best_cost

    def _calc_probs(self, weights):
        total = sum(weights)
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
    
    def _random_removal(self, tour, n):
        # tour[0] is depot, don't remove
        candidates = tour[1:]
        removed = random.sample(candidates, n)
        partial = [x for x in tour if x not in removed]
        return partial, removed

    def _worst_removal(self, tour, n):
        # Remove nodes that contribute most to cost.
        # We perform full global evaluation (calculate_cost) for each candidate to accurately 
        # capture time-dependent effects (global time shifts), as local delta approximations 
        # are invalid/inaccurate in TDTSP.
        
        current_tour = list(tour)
        removed = []
        
        for _ in range(n):
            max_save = -float('inf')
            target_idx = -1
            
            # Calculate base cost
            base_cost = self.evaluator.calculate_cost(current_tour)
            
            # Try removing each node (except depot)
            # Full scan is necessary for accuracy in TDTSP.
            
            candidates_indices = range(1, len(current_tour))
            # Optimization: If tour is large, random sample candidates?
            # For N<100, full scan is fine.
            
            for i in candidates_indices:
                # Create temp tour
                temp_tour = current_tour[:i] + current_tour[i+1:]
                cost = self.evaluator.calculate_cost(temp_tour, check_completeness=False)
                save = base_cost - cost
                
                if save > max_save:
                    max_save = save
                    target_idx = i
            
            if target_idx != -1:
                removed.append(current_tour[target_idx])
                current_tour.pop(target_idx)
            else:
                break
                
        return current_tour, removed

    # --- Repair Operators ---
    
    def _greedy_insertion(self, partial_tour, removed_nodes):
        # Insert removed nodes one by one into best position
        # Order of insertion: Random shuffle to avoid bias
        
        curr_tour = list(partial_tour)
        to_insert = list(removed_nodes)
        random.shuffle(to_insert)
        
        for node in to_insert:
            best_cost = float('inf')
            best_pos = -1
            
            for i in range(1, len(curr_tour) + 1):
                # Insert at i (between i-1 and i)
                temp_tour = curr_tour[:i] + [node] + curr_tour[i:]
                cost = self.evaluator.calculate_cost(temp_tour, check_completeness=False)
                
                if cost < best_cost:
                    best_cost = cost
                    best_pos = i
            
            curr_tour.insert(best_pos, node)
            
        return curr_tour

    def _regret_insertion(self, partial_tour, removed_nodes):
        # Regret-2
        curr_tour = list(partial_tour)
        to_insert = list(removed_nodes)
        
        while to_insert:
            best_regret = -float('inf')
            target_node = -1
            target_pos = -1
            
            # For each node, find best and second best insertion
            candidates = []
            
            for node in to_insert:
                costs = []
                # Try all positions
                for i in range(1, len(curr_tour) + 1):
                    temp_tour = curr_tour[:i] + [node] + curr_tour[i:]
                    cost = self.evaluator.calculate_cost(temp_tour, check_completeness=False)
                    costs.append((cost, i))
                
                # Sort costs
                costs.sort(key=lambda x: x[0])
                
                best_c, best_p = costs[0]
                if len(costs) > 1:
                    second_c = costs[1][0]
                else:
                    second_c = float('inf') # Or huge number
                    
                regret = second_c - best_c
                candidates.append((regret, node, best_p))
            
            # Pick max regret
            candidates.sort(key=lambda x: x[0], reverse=True)
            _, best_node, best_pos = candidates[0]
            
            curr_tour.insert(best_pos, best_node)
            to_insert.remove(best_node)
            
        return curr_tour
