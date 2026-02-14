import numpy as np

class TDTSPEvaluator:
    def __init__(self, matrix, duration, time_windows=None, service_times=None, start_time=0.0, penalty_value=3.0):
        """
        matrix: [N, N, T] numpy array (travel times)
        duration: float (duration of one time step)
        time_windows: list of (early, late) tuples. None if no TW.
        service_times: list of service times. None if 0.
        start_time: float
        penalty_value: float (used for late penalty)
        """
        self.matrix = matrix
        self.num_nodes = matrix.shape[0]
        self.num_steps = matrix.shape[2]
        self.duration = duration
        self.start_time = start_time
        self.max_s = self.num_steps - 1
        self.time_windows = time_windows
        self.service_times = service_times
        self.penalty_value = penalty_value
        self.was_late = False

    def calculate_cost(self, tour, return_details=False, late_tolerance=180.0, check_completeness=True):
        """
        tour: list of node indices (0 to N-1). 
        Assumes tour starts at tour[0] (usually depot=0).
        late_tolerance: float (ignore violation if violation <= late_tolerance)
        check_completeness: bool (if True, enforces tour has all nodes)
        """
        current_time = self.start_time
        current_node = tour[0]
        total_violation = 0.0
        self.was_late = False
        
        # Check if all customers are visited exactly once (except depot which is visited start/end implicitly)
        # The tour input should be a permutation of all nodes
        unique_nodes = set(tour)
        
        # Check for duplicates (INVALID)
        if len(unique_nodes) != len(tour):
            print(f"Warning: Tour has duplicates. Length: {len(tour)}, Unique: {len(unique_nodes)}")
            return float('inf')
            
        # Check completeness if requested
        if check_completeness:
            if len(tour) != self.num_nodes or len(unique_nodes) != self.num_nodes:
                 # We return inf for incomplete tours when strict check is enabled
                 return float('inf')
 
        violations = [] # (node_idx, amount)
        arrival_times = [current_time]
        
        # Check start node TW
        if self.time_windows is not None:
            early, late = self.time_windows[current_node]
            if current_time < early:
                current_time = early
            if current_time > late:
                v = current_time - late
                if v > late_tolerance:
                    total_violation += v
                    violations.append((0, v))
        
        # Traverse tour
        for i in range(1, len(tour)):
            next_node = tour[i]
            
            # Add service time at current node
            if self.service_times:
                current_time += self.service_times[tour[i-1]] # FIXED: was i-1, should be tour[i-1]
            
            tt = self._get_travel_time(current_node, next_node, current_time)
            current_time += tt
            arrival_times.append(current_time)
            
            # TW check at next_node
            if self.time_windows is not None:
                early, late = self.time_windows[next_node]
                if current_time < early:
                    current_time = early
                if current_time > late:
                    v = current_time - late
                    if v > late_tolerance:
                        total_violation += v
                        violations.append((i, v))
            
            current_node = next_node
            
        # Return to start
        if self.service_times:
             current_time += self.service_times[tour[-1]]

        tt = self._get_travel_time(current_node, tour[0], current_time)
        current_time += tt
        arrival_times.append(current_time)
        
        # Depot return TW? Usually depot has long TW.
        if self.time_windows is not None:
            early, late = self.time_windows[0]
            if current_time > late:
                v = current_time - late
                if v > late_tolerance:
                    total_violation += v
                    violations.append((len(tour), v))
        
        final_time = current_time
        makespan_hour = final_time / 3600.0
        
        if total_violation > 0:
            self.was_late = True
            penalty = (total_violation / 3600.0) * self.penalty_value
            penalty = max(penalty, 0.5 * makespan_hour)
            penalty = min(penalty, 2.0 * makespan_hour)
            total_cost = (makespan_hour + penalty) * 3600.0
        else:
            total_cost = final_time

        if return_details:
            return total_cost, {
                "violations": violations,
                "arrival_times": arrival_times,
                "total_violation": total_violation,
                "all_visited": (len(tour) == self.num_nodes) and (len(set(tour)) == self.num_nodes)
            }
        
        return total_cost

    def _get_travel_time(self, u, v, current_time):
        s = int(current_time // self.duration)
        if s > self.max_s:
            s = self.max_s
        return float(self.matrix[u, v, s])

    def validate_tour(self, tour):
        """
        Validates if the tour is a valid permutation of all nodes.
        Returns True if valid, False otherwise.
        """
        if len(tour) != self.num_nodes:
            return False
        if len(set(tour)) != self.num_nodes:
            return False
        return True
