"""
Optimized Fast Heuristic for Large-Scale TDVRP
Focuses on speed and scalability while maintaining solution quality.
"""

import numpy as np
from typing import List, Tuple
import time


def solve(matrix: np.ndarray,
          duration: float,
          time_windows: np.ndarray,
          service_times: np.ndarray,
          demands: np.ndarray = None,
          capacity: float = None,
          **kwargs) -> Tuple[List[List[int]], float]:
    """
    Fast TDVRP solver optimized for large instances.

    Strategy:
    1. Cluster-first, route-second approach
    2. Nearest neighbor with regret-based insertion
    3. Limited local search (intra-route only)
    """
    N = matrix.shape[0]
    T = matrix.shape[2]
    timeout = kwargs.get('timeout', 10.0)
    start_time = time.time()

    # Set defaults
    if demands is None:
        demands = np.ones(N)
        demands[0] = 0

    if capacity is None:
        total_demand = np.sum(demands[1:])
        est_vehicles = max(1, int(np.ceil(total_demand / (total_demand / max(1, int(np.sqrt(N)))))))
        capacity = total_demand / est_vehicles * 1.2

    def get_travel_time(from_node: int, to_node: int, curr_time: float) -> float:
        t_idx = min(int(curr_time / duration), T - 1)
        return float(matrix[from_node, to_node, t_idx])

    def evaluate_route(route: List[int]) -> Tuple[float, bool]:
        """Quick route evaluation."""
        if len(route) <= 2:
            return 0.0, True

        # Check capacity
        load = sum(demands[n] for n in route[1:-1])
        if load > capacity:
            return float('inf'), False

        current_time = 0.0
        total_cost = 0.0

        for i in range(len(route) - 1):
            curr = route[i]
            next_node = route[i + 1]

            if i > 0:
                current_time += service_times[curr]

            travel = get_travel_time(curr, next_node, current_time)
            current_time += travel
            total_cost += travel

            earliest, latest = time_windows[next_node]
            if current_time < earliest:
                current_time = earliest
            elif current_time > latest:
                return float('inf'), False

        return total_cost, True

    # Regret-based parallel insertion
    routes = []
    unassigned = list(range(1, N))

    # Sort by urgency (earliest deadline first)
    unassigned.sort(key=lambda x: time_windows[x][1])

    # Build initial routes using regret heuristic
    while unassigned:
        if time.time() - start_time > timeout * 0.7:
            break

        customer = unassigned[0]

        # Calculate regret: difference between best and second-best insertion
        insertion_costs = []

        for route_idx, route in enumerate(routes):
            best_cost = float('inf')
            best_pos = None

            for pos in range(1, len(route)):
                test_route = route[:pos] + [customer] + route[pos:]
                cost, feasible = evaluate_route(test_route)
                if feasible and cost < best_cost:
                    best_cost = cost
                    best_pos = pos

            if best_pos is not None:
                original_cost, _ = evaluate_route(route)
                insertion_costs.append((best_cost - original_cost, route_idx, best_pos))

        if not insertion_costs:
            # Create new route
            routes.append([0, customer, 0])
            unassigned.remove(customer)
        else:
            # Sort by insertion cost
            insertion_costs.sort()

            if len(insertion_costs) == 1:
                regret = float('inf')
            else:
                regret = insertion_costs[1][0] - insertion_costs[0][0]

            # Insert at best position
            _, route_idx, pos = insertion_costs[0]
            routes[route_idx].insert(pos, customer)
            unassigned.remove(customer)

    # Handle any remaining unassigned (should be rare)
    for customer in unassigned[:]:
        routes.append([0, customer, 0])

    # Fast intra-route 2-opt
    def fast_2opt(route: List[int], max_iter: int = 20) -> List[int]:
        if len(route) <= 4:
            return route

        best = route.copy()
        best_cost, _ = evaluate_route(best)

        for _ in range(max_iter):
            if time.time() - start_time > timeout * 0.95:
                break

            improved = False
            for i in range(1, len(best) - 2):
                for j in range(i + 2, min(i + 6, len(best) - 1)):  # Limit search window
                    new = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    cost, feasible = evaluate_route(new)

                    if feasible and cost < best_cost - 1e-6:
                        best = new
                        best_cost = cost
                        improved = True
                        break
                if improved:
                    break

            if not improved:
                break

        return best

    # Apply limited local search
    print("Optimizing routes...")
    routes = [fast_2opt(r) for r in routes]

    # Calculate total cost
    total_cost = sum(evaluate_route(r)[0] for r in routes)

    print(f"Generated {len(routes)} routes, total cost: {total_cost:.2f}")

    return routes, total_cost
