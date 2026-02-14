
import os
import json
import bz2
import numpy as np
import torch
from scipy.spatial import cKDTree
from tensordict.tensordict import TensorDict
from rl4co.envs.common.utils import Generator
from .env import TDTSPMatrixEnv
from rl4co.utils.pylogger import get_pylogger
# Simple console logger
class SimpleLogger:
    def info(self, msg): print(f"INFO: {msg}")
    def warning(self, msg): print(f"WARNING: {msg}")
    def error(self, msg): print(f"ERROR: {msg}")

log = get_pylogger(__name__)
# log = SimpleLogger()

class TDTSPTWGenerator(Generator):
    def __init__(self, 
                 data_path: str, 
                 base_data_path: str, 
                 matrix_path: str, 
                 num_matrix_steps: int = 37,
                 force_rebuild_matrix: bool = False,
                 service_time: float = 180.0, # 3 minutes delivery time
                 phase: str = "train",
                 random_sample: bool = True,
                 target_base_file: str = None,
                 **kwargs):
        self.service_time = service_time
        self.random_sample = random_sample
        self.target_base_file = target_base_file
        """
        Generator for TDTSPTWEnv.
        Args:
            data_path: Path to the .npz file containing the dataset (e.g. berlin_20_1000.npz)
            base_data_path: Path to the base instance JSON (e.g. berlin_2000.json)
            matrix_path: Path to the time-dependent matrix (e.g. berlin_2000_tt.json.bz2)
            num_matrix_steps: Number of discrete time steps for the matrix approximation
            phase: Current phase (train, val, or test) for 7:1:2 dataset splitting
        """
        self.data_path = data_path
        self.base_data_path = base_data_path
        self.matrix_path = matrix_path
        self.num_matrix_steps = num_matrix_steps
        self.force_rebuild_matrix = force_rebuild_matrix
        self.phase = phase
        self.min_loc = 0
        self.max_loc = 1
        self.current_instance_id = 0
        # data path 中的第一个 N_nodes 是 depot
        try:
            self.num_loc = int(os.path.basename(data_path).split("_")[1])
        except (IndexError, ValueError):
            log.warning(f"Could not parse num_loc from filename {data_path}, attempting to use labels shape")
            self.num_loc = None 
        
        self._load_data()

    def _load_data(self):
        # 1. Load NPZ Data
        if not hasattr(self, "_all_data_dict"):
            log.info(f"Loading dataset from {self.data_path}")
            with np.load(self.data_path) as data:
                # Load all arrays into memory as dict to avoid keeping file handle open
                self._all_data_dict = {k: data[k] for k in data.files}
        
        data = self._all_data_dict
        locs_idx = data['locs_idx'].astype(int)
        labels = data['labels'].astype(int)
        base_files = data['base_file']
        
        # Identify unique base files and choose one (e.g., the one with '2000' or the first one)
        unique_bases = np.unique(base_files)
        log.info(f"Dataset contains samples from: {unique_bases}")
        
        # Selection logic: prioritize '2000' as suggested by user
        target_base = self.target_base_file
        if target_base is None:
            for b in unique_bases:
                if "2000" in b:
                    target_base = b
                    break
            if target_base is None:
                target_base = unique_bases[0]
            
        log.info(f"Filtering dataset to only use samples from: {target_base}")
        mask = (base_files == target_base)
        all_locs_idx = locs_idx[mask]
        all_labels = labels[mask]
        all_base_files = base_files[mask]
        total_samples = all_locs_idx.shape[0]
        
        if total_samples == 0:
            raise ValueError(f"No samples found for target base file: {target_base}")

        # 7:1:2 Split logic
        train_end = int(0.7 * total_samples)
        val_end = int(0.8 * total_samples)
        
        if self.phase == "train":
            start_idx, end_idx = 0, train_end
        elif self.phase == "val":
            start_idx, end_idx = train_end, val_end
        elif self.phase == "test":
            start_idx, end_idx = val_end, total_samples
        else:
            log.warning(f"Unknown phase {self.phase}, defaulting to all samples")
            start_idx, end_idx = 0, total_samples
            
        self.locs_idx = all_locs_idx[start_idx:end_idx]
        self.labels = all_labels[start_idx:end_idx]
        self.base_files = all_base_files[start_idx:end_idx]
        self.num_samples = self.locs_idx.shape[0]
        
        log.info(f"Phase {self.phase}: using samples {start_idx} to {end_idx} (Total: {self.num_samples})")
            
        base_file_name = target_base
        
        # 2. Load Base Instance (Coordinates and Time Windows)
        if not hasattr(self, "base_coords"):
            if os.path.isdir(self.base_data_path):
                actual_base_data_path = os.path.join(self.base_data_path, base_file_name)
            else:
                actual_base_data_path = self.base_data_path
                
            log.info(f"Loading base instance from {actual_base_data_path}")
            with open(actual_base_data_path, 'r') as f:
                base_json = json.load(f)
                
            base_coords = []
            base_tws = []
            depot = base_json['depot']
            base_coords.append([depot['latitude'], depot['longitude']])
            start = depot.get('earliest_delivery', 54000000.0)
            end = depot.get('latest_delivery', 86400000.0)
            base_tws.append([start / 1000.0, end / 1000.0])
            
            self.num_base_nodes = len(base_json['items']) + 1
            for i in range(1, self.num_base_nodes):
                if str(i) in base_json['items']:
                    item = base_json['items'][str(i)]
                    base_coords.append([item['latitude'], item['longitude']])
                    start = item.get('earliest_delivery', 54000000.0)
                    end = item.get('latest_delivery', 86400000.0)
                    base_tws.append([start / 1000.0, end / 1000.0])
                else:
                    log.warning(f"Item {i} not found in base instance items.")
                    base_coords.append([0.0, 0.0])
                    base_tws.append([0.0, 0.0])
                
            self.base_coords = np.array(base_coords)
            self._all_base_tws = np.array(base_tws)
            
            # 3. Load or Build Matrix
            if os.path.isdir(self.matrix_path):
                matrix_file_name = base_file_name.replace(".json", "_tt.json.bz2")
                actual_matrix_path = os.path.join(self.matrix_path, matrix_file_name)
            else:
                actual_matrix_path = self.matrix_path

            cache_path = actual_matrix_path + f".steps{self.num_matrix_steps}.pt"
            if os.path.exists(cache_path) and not self.force_rebuild_matrix:
                log.info(f"Loading cached matrix from {cache_path}")
                cached_data = torch.load(cache_path)
                self.matrix = cached_data["matrix"]
                self.time_step_duration = cached_data["duration"]
                self.min_time = cached_data["min_time"]
            else:
                log.info(f"Building matrix from {actual_matrix_path} (this may take a while)...")
                self._build_and_cache_matrix(actual_matrix_path, cache_path)

            # 4. Shift Time Windows by min_time
            log.info(f"Shifting time windows by min_time: {self.min_time}")
            self.base_tws = self._all_base_tws - self.min_time
        else:
            # Still need to ensure base_tws is set correctly if min_time changed (though it shouldn't)
            self.base_tws = self._all_base_tws - self.min_time

    def _build_and_cache_matrix(self, matrix_path, cache_path):
        # Load JSON Matrix
        with bz2.open(matrix_path, "rt") as f:
            raw_matrix = json.load(f)
            
        # Determine time horizon from the data
        # We scan the first few entries to find min/max time
        # Or use the depot TW from base_json
        # Usually VRPTDT instances are within a day.
        # Let's assume a fixed range based on the data we saw: 
        # Start: 0 (or min leave time), End: 86400 (or max arrive time).
        # We'll use 0 to 86400 (24 hours) as safe bounds, or 
        # tighter bounds if we want better resolution.
        # Berlin 2000 depot open 0-79200000 (22h).
        # We use 54000 (15h) as min_time to match the depot TW. 
        # use 86400 (24h) as max_time to match the depot TW.
        min_time = 54000.0
        max_time = 86400.0 # Default 24h
        
        # Parse into a dense tensor [N, N, T]
        # Initialize with 0 or infinity? Travel time is usually > 0.
        matrix_tensor = torch.zeros((self.num_base_nodes, self.num_base_nodes, self.num_matrix_steps), dtype=torch.float32)
        
        # Time steps
        times = np.linspace(min_time, max_time, self.num_matrix_steps)
        self.time_step_duration = (max_time - min_time) / (self.num_matrix_steps - 1)
        self.min_time = min_time
        
        # Helper to map "depot" to 0
        def get_idx(s):
            return 0 if s == "depot" else int(s)

        log.info("Parsing ATF profiles and interpolating...")
        count = 0
        for entry in raw_matrix:
            u = get_idx(entry["from"])
            v = get_idx(entry["to"])
            
            atf = entry["atf"]
            # atf_leave_time (departure), atf_arrive_time (arrival)
            # Units: ms. Convert to seconds.
            leave_times = np.array(atf["atf_leave_time"]) / 1000.0
            arrive_times = np.array(atf["atf_arrive_time"]) / 1000.0
            
            # travel_time = arrive - leave
            # We want travel_time at time t.
            # We interpolate arrive_time at t, then subtract t.
            # Note: numpy interp requires sorted x. leave_times should be sorted.
            
            # Interpolate arrival times at query times
            # extrapolate: if t < min(leave), assume first travel time?
            # if t > max(leave), assume last travel time?
            # np.interp does constant extrapolation by default (left/right).
            
            interp_arrivals = np.interp(times, leave_times, arrive_times)
            travel_times = interp_arrivals - times
            
            # Ensure non-negative
            travel_times = np.maximum(travel_times, 0.0)
            
            matrix_tensor[u, v] = torch.from_numpy(travel_times).float()
            
            count += 1
            if count % 100000 == 0:
                log.info(f"Processed {count} edges...")
                
        log.info("Matrix build complete. Saving cache...")
        torch.save({
            "matrix": matrix_tensor,
            "duration": self.time_step_duration,
            "min_time": self.min_time
        }, cache_path)
        
        self.matrix = matrix_tensor

    def _generate(self, batch_size) -> TensorDict:
        if isinstance(batch_size, int):
            batch_size = [batch_size]
            
        batch_size = batch_size[0]
        
        # Sample indices from the loaded dataset
        if self.random_sample:
            idxs = np.random.randint(0, self.num_samples, size=batch_size)
        else:
            # Sequential sampling
            if batch_size < self.num_samples:
                # 按顺序返回且不重复（通过维护 current_instance_id）
                idxs = np.arange(self.current_instance_id, self.current_instance_id + batch_size) % self.num_samples
                self.current_instance_id = (self.current_instance_id + batch_size) % self.num_samples
                log.warning(f"Sequential sampling: {idxs}")
            else:
                if batch_size > self.num_samples:
                    log.warning(f"batch_size ({batch_size}) > num_samples ({self.num_samples}). Wrapping around.")
                idxs = np.arange(batch_size) % self.num_samples
        
        # Get location indices for this batch: [B, N]
        batch_locs_idx = self.locs_idx[idxs]
        
        # Gather coordinates and time windows from base data: [B, N, 2] and [B, N, 2]
        locs = torch.from_numpy(self.base_coords[batch_locs_idx]).float()
        time_windows = torch.from_numpy(self.base_tws[batch_locs_idx]).float()
        
        # Normalize coordinates to [0, 1] for the model
        # We use global min/max from base_coords for consistency
        c_min = torch.from_numpy(self.base_coords.min(axis=0)).float()
        c_max = torch.from_numpy(self.base_coords.max(axis=0)).float()
        locs = (locs - c_min) / (c_max - c_min + 1e-6)
        
        locs_idx = torch.from_numpy(batch_locs_idx).long()
        
        # Create TensorDict
        time_step_duration = [self.time_step_duration]*batch_size
        min_time = [self.min_time]*batch_size
        
        # 4. Extract sub-matrix for the current batch: [B, N, N, T]
        # self.matrix: [N_base, N_base, T], batch_locs_idx: [B, N]
        # Optimization: Use advanced indexing instead of large gather
        T = self.matrix.shape[-1]
        N = locs.shape[1]
        
        # Advanced indexing: [B, N, N, T]
        # This is more memory-efficient for sub-matrix extraction
        sub_matrix = self.matrix[batch_locs_idx[:, :, None], batch_locs_idx[:, None, :]]

        # Create TensorDict
        return TensorDict({
            "locs": locs,
            "time_windows": time_windows,
            "locs_idx": locs_idx,
            "current_node": torch.zeros(batch_size, dtype=torch.long),
            "current_time": torch.zeros(batch_size, dtype=torch.float32), # Start at 0 (shifted min_time)
            "i": torch.zeros(batch_size, dtype=torch.long),
            "action_mask": torch.ones(batch_size, locs.shape[1], dtype=torch.bool),
            "first_node": torch.zeros(batch_size, dtype=torch.long), 
            "time_step_duration": torch.tensor([self.time_step_duration]*batch_size),
            "min_time": torch.tensor([self.min_time]*batch_size),
            "travel_time_matrix": sub_matrix,
        }, batch_size=batch_size)


class TDTSPTWEnv(TDTSPMatrixEnv):
    name = "tdtsp_tw"

    def __init__(self, 
                 data_file_path: str = None, 
                 base_data_path: str = None,
                 matrix_path: str = None,
                 service_time: float = 180.0,
                 penalty_value: float = 0.0,
                 **kwargs):
        self.service_time = service_time
        self.penalty_value = penalty_value
        # Default paths if not provided
        if data_file_path is None:
            # Try to find any npz in the dataset directory
            data_dir = "/root/autodl-tmp/tdtsp_dataset"
            if os.path.exists(data_dir):
                npz_files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
                if npz_files:
                    data_file_path = os.path.join(data_dir, npz_files[0])
                else:
                    data_file_path = os.path.join(data_dir, "berlin_20_1000.npz")
            else:
                data_file_path = "/root/autodl-tmp/tdtsp_dataset/berlin_20_1000.npz"

        if base_data_path is None:
            # Now we prefer the directory containing the base JSONs
            base_data_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
        if matrix_path is None:
            # Now we prefer the directory containing the matrix BZ2s
            matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
        
        if "generator" not in kwargs:
            generator = TDTSPTWGenerator(
                data_path=data_file_path,
                base_data_path=base_data_path,
                matrix_path=matrix_path,
                **kwargs
            )
        else:
            generator = kwargs.pop("generator")
        super().__init__(generator=generator, **kwargs)
    
    def dataset(self, batch_size=[], phase="train", filename=None):
        """Override dataset to handle phase-specific data splitting in the generator"""
        if hasattr(self.generator, "phase"):
            old_phase = self.generator.phase
            self.generator.phase = phase
            self.generator._load_data() # Update indices for the new phase
            res = super().dataset(batch_size, phase, filename)
            self.generator.phase = old_phase # Restore
            self.generator._load_data()
            return res
        return super().dataset(batch_size, phase, filename)

    def _reset(self, td: TensorDict, **kwargs) -> TensorDict:
        device = td.device
        batch_size = kwargs.get("batch_size", td.batch_size)
        
        # Action mask: start with all True
        num_loc = td["locs"].shape[-2]
        action_mask = torch.ones((*batch_size, num_loc), dtype=torch.bool, device=device)
        
        # Set depot (index 0) to False as it's the starting point
        action_mask[..., 0] = False
        visited = torch.zeros((*batch_size, num_loc), dtype=torch.bool, device=device)
        visited[..., 0] = True
        
        return TensorDict({
            "locs": td["locs"],
            "time_windows": td["time_windows"],
            "locs_idx": td["locs_idx"] if "locs_idx" in td else None,
            "travel_time_matrix": td["travel_time_matrix"],
            "current_node": torch.zeros(batch_size, dtype=torch.long, device=device),
            "current_time": torch.zeros(batch_size, dtype=torch.float32, device=device),
            "i": torch.zeros(batch_size, dtype=torch.long, device=device),
            "action_mask": action_mask,
            "visited": visited,
            "first_node": torch.zeros(batch_size, dtype=torch.long, device=device),
            "time_step_duration": td["time_step_duration"],
            "min_time": td["min_time"],
            "reward": torch.zeros(batch_size, dtype=torch.float32, device=device),
        }, batch_size=batch_size)

    def _get_travel_time(self, td, prev_node_idx, current_node_idx, is_first_step=None, current_time=None):
        if current_time is None:
            current_time = td["current_time"]
            
        # 1. Look up Matrix from TensorDict (now it's [B, N, N, T])
        matrix = td["travel_time_matrix"]
        
        # 2. Calculate Time Step s
        duration = td["time_step_duration"]
        if duration.dim() == 0:
            duration = duration.unsqueeze(0).expand_as(current_time)
        elif duration.dim() == 1 and duration.shape[0] == 1:
            duration = duration.expand_as(current_time)
            
        s = (current_time // duration).long()
        s = s.clamp(min=0)
        max_s = matrix.shape[-1] - 1
        s = s.clamp(max=max_s)
        
        # 3. Use local indices directly on batch matrix
        batch_idx = torch.arange(td.batch_size[0], device=td.device)
        travel_time = matrix[batch_idx, prev_node_idx, current_node_idx, s]
            
        # Mask first step
        if is_first_step is not None:
            travel_time[is_first_step] = 0.0
        return travel_time.float()

    def _step(self, td: TensorDict) -> TensorDict:
        current_node_idx = td["action"]
        prev_node_idx = td["current_node"]
        visited = td["visited"]
        
        for i in range(td.batch_size[0]):
            visited[i, current_node_idx[i]] = True
        
        # Handle first step logic
        is_first_step = (td["i"] == 0)
        
        # 1. Calculate Travel Time
        travel_time = self._get_travel_time(td, prev_node_idx, current_node_idx, is_first_step)
        
        # 2. Update Time and Check TW
        current_time = td["current_time"]
        arrival_time = current_time + travel_time
        
        # Get TW for current node
        # time_windows: (B, N, 2)
        # gather: (B, 1, 2)
        node_tw = td["time_windows"].gather(1, current_node_idx.view(-1, 1, 1).expand(-1, -1, 2)).squeeze(1)
        early_tw = node_tw[:, 0]
        late_tw = node_tw[:, 1]
        
        # Wait if early
        ready_time = torch.max(arrival_time, early_tw)
        
        # 2.1 Add Service Time (Delivery Time)
        # Service time is only added for delivery nodes (not depot at start, but maybe at end?)
        # For simplicity, we add it to all nodes except depot at start.
        # However, is_first_step handles the trip from depot to first customer.
        # So arrival_time is arrival at customer i. 
        # service starts at ready_time, finishes at ready_time + service_time.
        departure_time = ready_time + self.service_time
        
        # Update masked nodes
        # Mask current node
        available = td["action_mask"].scatter(
            -1, current_node_idx.unsqueeze(-1).expand_as(td["action_mask"]), 0
        )
        
        # 3. Lookahead Masking (TW Feasibility)
        # For all available nodes k, check if departure_time + travel(current, k) > late_tw(k)
        
        # Time step s
        duration = td["time_step_duration"]
        if duration.dim() == 0:
            duration = duration.unsqueeze(0).expand_as(departure_time)
            
        s = (departure_time // duration).long()
        s = s.clamp(min=0)
        
        matrix = td["travel_time_matrix"]
        max_s = matrix.shape[-1] - 1
        s = s.clamp(max=max_s)
        
        # Look up row from local matrix: [B, N, N, T]
        batch_idx = torch.arange(td.batch_size[0], device=td.device)
        tt_to_subset = matrix[batch_idx, current_node_idx, :, s] # (B, N)
        
        # Check feasibility
        arrival_at_next = departure_time.unsqueeze(-1) + tt_to_subset
        # next_late_tw: (B, N_subset)
        next_late_tw = td["time_windows"][..., 1]
        
        feasible = arrival_at_next <= next_late_tw
        
        # Update mask: available AND feasible
        # Only apply feasibility mask if penalty_value is 0
        if self.penalty_value == 0:
            actual_mask = available & feasible
            # Fallback: if no feasible nodes but still unvisited ones, use available mask to avoid crash
            has_unvisited = (available.sum(-1) > 0)
            none_feasible = (actual_mask.sum(-1) == 0) & has_unvisited
            actual_mask[none_feasible] = available[none_feasible]
        else:
            # If penalty is enabled, we don't mask based on TW feasibility
            actual_mask = available
            
        # Check if truly done: all nodes except node 0 (depot) are visited
        done = torch.sum(available[..., 1:], dim=-1) == 0
        
        # Update first node
        first_node_idx = td["first_node"]
        first_node_idx[is_first_step] = current_node_idx[is_first_step]
        
        # Reward
        reward = torch.zeros_like(done, dtype=torch.float32)
        
        td.update({
            "first_node": first_node_idx,
            "current_node": current_node_idx,
            "current_time": departure_time, # Updated to departure_time (waiting + service included)
            "i": td["i"] + 1,
            "action_mask": actual_mask,
            "visited": visited,
            "reward": reward,
            "done": done,
        })
        return td

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """
        Recalculate reward (negative makespan) with TW penalties.
        We assume actions contains customers (1 to N-1) and we start/end at depot (0).
        """
        batch_size = actions.shape[0]
        device = actions.device
        
        current_time = torch.zeros(batch_size, device=device)
        violations = torch.zeros(batch_size, device=device)
        
        # 1. Start at depot
        prev_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # time_windows: (B, N, 2)
        tws = td["time_windows"]
        
        # 2. Travel through all customers in actions
        for i in range(actions.shape[1]):
            curr_node = actions[:, i]
            
            # Travel time from prev_node to curr_node
            # If i=0, it's travel from depot to first customer
            tt = self._get_travel_time(td, prev_node, curr_node, current_time=current_time)
            arrival_time = current_time + tt
            
            # Check TW
            node_tw = tws.gather(1, curr_node.view(-1, 1, 1).expand(-1, -1, 2)).squeeze(1)
            early = node_tw[:, 0]
            late = node_tw[:, 1]
            
            # Penalty if late
            late_violation = torch.clamp(arrival_time - late, min=0)

            violations += late_violation
            
            # Wait if early
            ready_time = torch.max(arrival_time, early)
            current_time = ready_time + self.service_time
            prev_node = curr_node
            
        # 3. Return to depot (0)
        depot_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        tt_return = self._get_travel_time(td, prev_node, depot_node, current_time=current_time)
        final_time = current_time + tt_return
        
        # Return trip also has a TW (usually for depot)
        depot_tw = tws[:, 0, :]
        depot_late = depot_tw[:, 1]
        late_violation_return = torch.clamp(final_time - depot_late, min=0)
        violations += late_violation_return
        
        # 4. Calculate Final Reward
        makespan_hour = final_time / 3600.0
        
        # Store violations in td for reporting
        td.set("violations", violations)
        
        # Makespan + Penalty
        penalty = torch.zeros_like(makespan_hour, device=device)
        if violations.any() > 0:
            for b in range(batch_size):
                if violations[b] > 0:
                    penalty[b] += (violations[b] / 3600.0) * self.penalty_value
                    penalty[b] = max(penalty[b], 0.5*makespan_hour[b])
                    penalty[b] = min(penalty[b], 2*makespan_hour[b])
        final_reward = -(makespan_hour + penalty)
        td.set("cumulative_reward", final_reward)
        td.set("total_penalty", penalty)

        return final_reward
  

    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        """Check that solution is valid: all customers are visited exactly once"""
        batch_size, num_cust = actions.shape
        
        num_loc = td["locs"].shape[-2] # Includes depot
        
        # actions should contain indices 1 to num_loc-1 or 0 if num_cust == num_loc
        if num_cust == num_loc-1:
            expected = torch.arange(1, num_loc, device=actions.device).view(1, -1).expand_as(actions)
        else:
            expected = torch.arange(0, num_loc, device=actions.device).view(1, -1).expand_as(actions)

        sorted_actions, _ = actions.sort(1)
        
        assert (sorted_actions == expected).all(), "Invalid tour: not all customers visited exactly once"
        

