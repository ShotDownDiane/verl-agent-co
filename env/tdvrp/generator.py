import os
import json
import bz2
import numpy as np
import torch
from tensordict.tensordict import TensorDict
from rl4co.envs.common.utils import Generator
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class TDVRPGenerator(Generator):
    """Data generator for the Time-Dependent Vehicle Routing Problem with Time Windows (TDVRP-TW).
    Loads data from real-world instances (e.g., berlin_10.json) and travel time matrices.
    """
    def __init__(self, 
                 instance_path: str = "/root/autodl-tmp/vrptdt-benchmark/instances/berlin_10.json",
                 matrix_path: str = "/root/autodl-tmp/vrptdt-benchmark/instances/berlin_10_tt.json.bz2",
                 num_matrix_steps: int = 50,
                 force_rebuild_matrix: bool = False,
                 service_time: float = 180.0,
                 phase: str = "train",
                 num_nodes: int = 20,
                 data_path: str = None,
                 base_data_path: str = None,
                 random_sample: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.instance_path = instance_path
        self.matrix_path = matrix_path
        self.num_matrix_steps = num_matrix_steps
        self.force_rebuild_matrix = force_rebuild_matrix
        self.service_time = service_time
        self.phase = phase
        self.data_path = data_path
        self.base_data_path = base_data_path
        self.num_nodes = num_nodes
        self.random_sample = random_sample
        self.min_loc = 0
        self.max_loc = 1
        self.current_instance_id = 0
        
        self._load_data()

    def _load_data(self):
        # 1. Load NPZ Data if data_path is provided (following TDTSPTW logic)
        if self.data_path:
            if not hasattr(self, "_all_data_dict"):
                log.info(f"Loading dataset from {self.data_path}")
                with np.load(self.data_path) as data:
                    self._all_data_dict = {k: data[k] for k in data.files}
            
            data = self._all_data_dict
            locs_idx = data['locs_idx'].astype(int)
            base_files = data['base_file']
            
            unique_bases = np.unique(base_files)
            target_base = None
            for b in unique_bases:
                if "2000" in b:
                    target_base = b
                    break
            if target_base is None:
                target_base = unique_bases[0]
                
            mask = (base_files == target_base)
            all_locs_idx = locs_idx[mask]
            total_samples = all_locs_idx.shape[0]
            
            # 7:1:2 Split
            train_end = int(0.7 * total_samples)
            val_end = int(0.8 * total_samples)
            
            if self.phase == "train":
                start_idx, end_idx = 0, train_end
            elif self.phase == "val":
                start_idx, end_idx = train_end, val_end
            elif self.phase == "test":
                start_idx, end_idx = val_end, total_samples
            else:
                start_idx, end_idx = 0, total_samples
                
            self.locs_idx = all_locs_idx[start_idx:end_idx]
            self.num_samples = self.locs_idx.shape[0]
            log.info(f"Phase {self.phase}: using {self.num_samples} samples from {target_base}")

            # Use base_data_path or matrix_path if they are directories
            if self.base_data_path and os.path.isdir(self.base_data_path):
                self.instance_path = os.path.join(self.base_data_path, target_base)
            if self.matrix_path and os.path.isdir(self.matrix_path):
                matrix_file_name = target_base.replace(".json", "_tt.json.bz2")
                self.matrix_path = os.path.join(self.matrix_path, matrix_file_name)

        # 2. Load Base Instance
        
        log.info(f"Loading instance from {self.instance_path}")
        with open(self.instance_path, 'r') as f:
            instance_json = json.load(f)
            
        coords = []
        tws = []
        depot = instance_json['depot']
        coords.append([depot['latitude'], depot['longitude']])
        start = depot.get('earliest_delivery', 54000000.0) 
        end = depot.get('latest_delivery', 86400000.0) 
        tws.append([start / 1000.0, end / 1000.0])
        item_keys = sorted([int(k) for k in instance_json['items'].keys()])
        self.num_base_nodes = len(item_keys) + 1
        for k in item_keys:
            item = instance_json['items'][str(k)]
            coords.append([item['latitude'], item['longitude']])
            start = item.get('earliest_delivery', 54000000.0)
            end = item.get('latest_delivery', 86400000.0)
            tws.append([start / 1000.0, end / 1000.0])
            
        self.base_coords = np.array(coords)
        self.base_tws_raw = np.array(tws)
        
        # 3. Matrix
        cache_path = self.matrix_path + f".steps{self.num_matrix_steps}.pt"
        if os.path.exists(cache_path) and not self.force_rebuild_matrix:
            log.info(f"Loading cached matrix from {cache_path}")
            cached_data = torch.load(cache_path)
            self.matrix = cached_data["matrix"]
            self.time_step_duration = cached_data["duration"]
            self.min_time = cached_data["min_time"]
        else:
            log.info(f"Building matrix from {self.matrix_path}...")
            self._build_and_cache_matrix(cache_path)

        self.base_tws = self.base_tws_raw - self.min_time
        
        # If not using NPZ, initialize dummy locs_idx for num_nodes compatibility
        if not self.data_path:
            # self.num_nodes is already set in __init__
            pass
        else:
            self.num_nodes = self.locs_idx.shape[1]

    def _build_and_cache_matrix(self, cache_path):
        with bz2.open(self.matrix_path, "rt") as f:
            raw_matrix = json.load(f)
            
        min_time = 54000.0 
        max_time = 86400.0 
        
        matrix_tensor = torch.zeros((self.num_base_nodes, self.num_base_nodes, self.num_matrix_steps), dtype=torch.float32)
        times = np.linspace(min_time, max_time, self.num_matrix_steps)
        self.time_step_duration = (max_time - min_time) / (self.num_matrix_steps - 1)
        self.min_time = min_time
        
        def get_idx(s):
            return 0 if s == "depot" else int(s)

        for entry in raw_matrix:
            u = get_idx(entry["from"])
            v = get_idx(entry["to"])
            atf = entry["atf"]
            leave_times = np.array(atf["atf_leave_time"]) / 1000.0
            arrive_times = np.array(atf["atf_arrive_time"]) / 1000.0
            
            interp_arrivals = np.interp(times, leave_times, arrive_times)
            travel_times = interp_arrivals - times
            travel_times = np.maximum(travel_times, 0.0)
            matrix_tensor[u, v] = torch.from_numpy(travel_times).float()
            
        torch.save({
            "matrix": matrix_tensor,
            "duration": self.time_step_duration,
            "min_time": self.min_time
        }, cache_path)
        self.matrix = matrix_tensor

    def _generate(self, batch_size) -> TensorDict:

        if isinstance(batch_size, int):
            batch_size = [batch_size]
        b = batch_size[0]
        
        if self.data_path:
            # Sample indices from the loaded dataset
            if self.random_sample:
                idxs = np.random.randint(0, self.num_samples, size=b)
                log.warning(f"Sequential sampling: {idxs}")
            else:
                # Sequential sampling
                if b < self.num_samples:
                    # 按顺序返回且不重复（通过维护 current_instance_id）
                    idxs = np.arange(self.current_instance_id, self.current_instance_id + b) % self.num_samples
                    self.current_instance_id = (self.current_instance_id + b) % self.num_samples 
                    log.warning(f"Sequential sampling: {idxs}")
            locs_idx = self.locs_idx[idxs]
            locs = torch.from_numpy(self.base_coords[locs_idx]).float()
            time_windows = torch.from_numpy(self.base_tws[locs_idx]).float()  
            
            c_min = torch.from_numpy(self.base_coords.min(axis=0)).float()
            c_max = torch.from_numpy(self.base_coords.max(axis=0)).float()
            locs_norm = (locs - c_min) / (c_max - c_min + 1e-6)
            
            T = self.matrix.shape[-1]
            sub_matrix = self.matrix[locs_idx[:, :, None], locs_idx[:, None, :]]
        else:
            # Sample num_nodes from base_nodes (index 0 is depot, 1+ are customers)
            if self.num_nodes > self.num_base_nodes:
                log.warning(f"num_nodes ({self.num_nodes}) > available base nodes ({self.num_base_nodes}). Using all base nodes.")
                num_to_sample = self.num_base_nodes
            else:
                num_to_sample = self.num_nodes
            
            # Create indices for each instance in batch
            customer_indices = np.arange(1, self.num_base_nodes)
            
            batch_locs_idx = []
            for i in range(b):
                if self.random_sample:
                    sampled_customers = np.random.choice(customer_indices, num_to_sample - 1, replace=False)
                else:
                    # Sequential selection of customers for each instance in batch
                    # This is a bit arbitrary, but let's say we shift the selection by i
                    start_idx = (i * (num_to_sample - 1)) % len(customer_indices)
                    # For simplicity, if random_sample is False, just take the first set of nodes
                    # or some deterministic subset. Let's just take the first N-1 nodes.
                    sampled_customers = customer_indices[:num_to_sample - 1]
                
                full_idx = np.concatenate([[0], sorted(sampled_customers)])
                batch_locs_idx.append(full_idx)
            
            batch_locs_idx = np.array(batch_locs_idx) # [b, num_to_sample]
            
            locs = torch.from_numpy(self.base_coords[batch_locs_idx]).float()
            time_windows = torch.from_numpy(self.base_tws[batch_locs_idx]).float()
            
            c_min = locs.min(dim=1, keepdim=True)[0]
            c_max = locs.max(dim=1, keepdim=True)[0]
            locs_norm = (locs - c_min) / (c_max - c_min + 1e-6)
            
            T = self.matrix.shape[-1]
            sub_matrix = self.matrix[batch_locs_idx[:, :, None], batch_locs_idx[:, None, :]]
            # gather_idx = torch.from_numpy(batch_locs_idx).long().unsqueeze(1).unsqueeze(-1).expand(-1, num_to_sample, -1, T)
            # sub_matrix = sub_matrix.gather(2, gather_idx)
        
        return TensorDict({
            "locs": locs_norm,
            "time_windows": time_windows,
            "travel_time_matrix": sub_matrix,
            "time_step_duration": torch.tensor([self.time_step_duration] * b),
            "min_time": torch.tensor([self.min_time] * b),
        }, batch_size=batch_size)
