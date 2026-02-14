import random
import math
from heuristics import GreedyRandomized

class SAH:
    def __init__(self, evaluator, initial_temp=100, cooling_rate=0.995, max_iter=5000, initial_solution=None):
        self.evaluator = evaluator
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.max_iter = max_iter
        self.initial_solution = initial_solution

    def _violation_amount(self, details):
        """
        Returns (is_feasible: bool, total_late: float).
        total_late is in seconds (or same unit as details provides) if available; otherwise count-like proxy.
        """
        if not details:
            return True, 0.0

        violations = details.get("violations", None)
        if not violations:
            return True, 0.0

        total = details.get("total_violation",0)
        return (total <= 0.0), total

    def _accept(self, cur_cost, cur_feas, cur_late, nb_cost, nb_feas, nb_late, temp):
        """
        Feasibility-first acceptance.
        """
        if cur_feas and not nb_feas:
            return False
        if not cur_feas and nb_feas:
            return True

        if not cur_feas and not nb_feas:
            delta_v = nb_late - cur_late
            if delta_v < 0:
                return True
            if temp <= 0:
                return False
            try:
                return random.random() < math.exp(-delta_v / temp)
            except OverflowError:
                return False

        delta = nb_cost - cur_cost
        if delta < 0:
            return True
        if temp <= 0:
            return False
        try:
            return random.random() < math.exp(-delta / temp)
        except OverflowError:
            return False

    def solve(self):
        # 1) Initial solution: Use provided or Greedy
        if self.initial_solution is not None:
            current_solution = list(self.initial_solution)
        else:
            greedy = GreedyRandomized(self.evaluator, k=3)
            current_solution, _ = greedy.solve()

        current_cost, details = self.evaluator.calculate_cost(current_solution, return_details=True)
        cur_feas, cur_late = self._violation_amount(details)

        # Track best feasible ONLY
        best_feasible_solution = list(current_solution) if cur_feas else None
        best_feasible_cost = current_cost if cur_feas else float("inf")

        temp = self.initial_temp

        for _ in range(self.max_iter):
            neighbor = self._get_neighbor_tw(current_solution, details)
            neighbor_cost, neighbor_details = self.evaluator.calculate_cost(neighbor, return_details=True)
            nb_feas, nb_late = self._violation_amount(neighbor_details)

            # Update best feasible whenever we see one
            if nb_feas and neighbor_cost < best_feasible_cost:
                best_feasible_cost = neighbor_cost
                best_feasible_solution = list(neighbor)

            # Feasibility-first accept
            if self._accept(current_cost, cur_feas, cur_late,
                            neighbor_cost, nb_feas, nb_late,
                            temp):
                current_solution = neighbor
                current_cost = neighbor_cost
                details = neighbor_details
                cur_feas, cur_late = nb_feas, nb_late

            temp *= self.cooling_rate

        # RETURN ONLY BEST FEASIBLE
        if best_feasible_solution is not None:
            return best_feasible_solution, best_feasible_cost

        # Fallback: if never feasible, return the initial solution (or raise)
        # Returning initial_solution is safer for pipelines; raising is stricter.
        # Here I return current_solution as a fallback, but you can raise ValueError instead.
        
        return current_solution, current_cost

    def _get_neighbor_tw(self, tour, details):
        """
        TW-aware neighbor:
        - If there are violations, pick a late *node id* and relocate it earlier.
        - Works whether tour starts with depot 0 or not.
        """
        neighbor = list(tour)
        n = len(neighbor)
        if n <= 2:
            return neighbor

        violations = details.get("violations", [])

        # Determine mutable index range
        has_depot = (neighbor[0] == 0)
        lo = 1 if has_depot else 0  # smallest mutable index
        hi = n - 1                  # largest mutable index

        # Try a repair move with 50% probability
        if violations and random.random() < 0.5:
            # v is like (node_id, late_amount)
            late_nodes = []
            for v in violations:
                if isinstance(v, (tuple, list)) and len(v) >= 1 and isinstance(v[0], int):
                    late_nodes.append(v[0])
                elif isinstance(v, dict):
                    # in case your evaluator uses dict format sometimes
                    for key in ("node", "customer", "cust", "id", "idx"):
                        if isinstance(v.get(key, None), int):
                            late_nodes.append(v[key])
                            break

            # keep only nodes that actually appear in the tour
            late_nodes = [node for node in late_nodes if node in neighbor]

            if late_nodes:
                node = random.choice(late_nodes)
                idx = neighbor.index(node)  # <--关键修复：node_id -> position index

                # Only move if it's in movable range
                if lo <= idx <= hi:
                    val = neighbor.pop(idx)

                    # Move it earlier (toward lo)
                    # new_idx in [lo, idx] (inclusive) => earlier or same
                    new_idx = random.randint(lo, idx)
                    neighbor.insert(new_idx, val)
                    return neighbor

        # Otherwise do a standard move
        return self._get_neighbor_standard(neighbor)


    def _get_neighbor_standard(self, tour):
        """
        Standard operators (swap/reverse/insert) that respect optional depot at index 0.
        """
        neighbor = list(tour)
        n = len(neighbor)
        if n <= 2:
            return neighbor

        has_depot = (neighbor[0] == 0)
        lo = 1 if has_depot else 0
        if n - lo < 2:
            return neighbor

        op = random.choice(['swap', 'reverse', 'insert'])
        i, j = sorted(random.sample(range(lo, n), 2))

        if op == 'swap':
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        elif op == 'reverse':
            neighbor[i:j+1] = reversed(neighbor[i:j+1])
        elif op == 'insert':
            val = neighbor.pop(i)
            # reinsert near j (handle index shift)
            if j > i:
                j -= 1
            neighbor.insert(j, val)

        return neighbor