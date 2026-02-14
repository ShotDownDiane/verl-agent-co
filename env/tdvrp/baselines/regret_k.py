import math
import random

class RegretInsertion:
    """
    Regret-k insertion for TDVRPTW (no capacity/demand).
    Output: routes = [[0, ..., 0], [0, ..., 0], ...]
    """

    def __init__(
        self,
        evaluator,
        k=2,                               # regret-k
        seed_strategy="earliest_due",      # "earliest_due" / "random"
        rcl=None,                          # top-rcl regret selection
        max_routes=None                    # limit number of vehicles
    ):
        self.evaluator = evaluator
        self.k = k
        self.start_time = float(evaluator.start_time)
        self.seed_strategy = seed_strategy
        self.rcl = rcl
        self.max_routes = max_routes

        self.customers = [i for i in range(1, evaluator.num_nodes)]

    # -------------------- public --------------------

    def solve(self):
        unserved = set(self.customers)
        routes = []

        # 开一条初始路线
        seed = self._choose_seed(unserved)
        routes.append([0, seed, 0])
        unserved.remove(seed)

        while unserved:
            move = self._best_regret_move(unserved, routes)

            if move is None:
                # 没有任何可行插入：开新路线（如果允许）
                if self.max_routes is not None and len(routes) >= self.max_routes:
                    break
                seed = self._choose_seed(unserved)
                routes.append([0, seed, 0])
                unserved.remove(seed)
                continue

            # 执行插入
            node, r_idx, pos, _delta = move
            routes[r_idx].insert(pos, node)
            unserved.remove(node)

        # 你可以用 evaluator.calculate_cost(routes) 得到统一口径 cost
        # Convert [[0, 1, 0], [0, 2, 0]] to [[1], [2]] for evaluator
        eval_routes = [r[1:-1] for r in routes]
        cost = self.evaluator.calculate_cost(eval_routes)
        return eval_routes, cost

    # -------------------- core logic --------------------

    def _best_regret_move(self, unserved, routes):
        """
        For each unserved node:
          - compute best insertion deltas into all routes & positions (feasible only)
          - regret = sum_{i=2..k} (delta_i - delta_1)  (or use delta_k - delta_1)
        Choose node with max regret (tie-break: smallest best delta).
        Return (node, route_index, insert_pos, best_delta) or None if nothing feasible.
        """
        candidates = []

        for node in unserved:
            insertions = self._all_feasible_insertions(node, routes)
            if not insertions:
                continue

            # sort by delta ascending
            insertions.sort(key=lambda x: x[0])
            best_delta, best_r_idx, best_pos = insertions[0]

            # compute regret
            # use up to k best insertion deltas
            m = min(self.k, len(insertions))
            base = insertions[0][0]
            regret = 0.0
            for i in range(1, m):
                regret += (insertions[i][0] - base)

            candidates.append((regret, best_delta, node, best_r_idx, best_pos))

        if not candidates:
            return None

        # want max regret; tie-break by smaller best_delta
        candidates.sort(key=lambda x: (-x[0], x[1]))

        if self.rcl is not None and self.rcl > 1:
            r = min(self.rcl, len(candidates))
            pick = random.randint(0, r - 1)
            regret, best_delta, node, best_r_idx, best_pos = candidates[pick]
        else:
            regret, best_delta, node, best_r_idx, best_pos = candidates[0]

        return (node, best_r_idx, best_pos, best_delta)

    def _all_feasible_insertions(self, node, routes):
        """
        Returns list of (delta_cost, route_index, insert_pos) for feasible insertions only.
        """
        out = []
        for r_idx, route in enumerate(routes):
            # route looks like [0, ..., 0], insert positions are between them: 1..len(route)-1
            for pos in range(1, len(route)):
                new_route_nodes = route[:pos] + [node] + route[pos:]
                # Remove start/end 0s for evaluate_route if it expects only customer nodes
                # or adjust based on evaluate_route implementation.
                # Looking at evaluator.py, evaluate_route expects a list of customer nodes.
                customer_nodes = new_route_nodes[1:-1]
                
                res = self.evaluator.evaluate_route(customer_nodes)
                if res["violation_sec"] > 0:
                    continue

                # Use end_time as delta basis
                old_customer_nodes = route[1:-1]
                res_old = self.evaluator.evaluate_route(old_customer_nodes)
                
                delta = res["end_time"] - res_old["end_time"]
                out.append((delta, r_idx, pos))
        return out

    # -------------------- simulation (TD + TW) --------------------
    # (Removed as we use evaluator.evaluate_route)

    # -------------------- data access --------------------
    # (Removed as we use evaluator properties)

    def _choose_seed(self, unserved):
        """
        Choose a seed to start a new route.
        - earliest_due: smallest tw_close
        - random: random pick
        """
        if not unserved:
            raise ValueError("No unserved customers.")

        if self.seed_strategy == "random":
            return random.choice(list(unserved))

        # default: earliest_due
        best = None
        best_due = float("inf")
        for node in unserved:
            tw = self.evaluator.time_windows[node]
            due = tw[1] if tw is not None else float("inf")
            if due < best_due:
                best_due = due
                best = node

        return best if best is not None else random.choice(list(unserved))
