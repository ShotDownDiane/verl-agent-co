from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class TDTSPGenerator(Generator):
    """Data generator for the Time-Dependent Travelling Salesman Problem (TDTSP).
    Generates locations and speed profile parameters.

    The speed profile is modeled as:
        speed(t) = base_speed + speed_amplitude * sin(2 * pi * t / period)
    Travel time is approximated by: dist / speed(start_time)

    Args:
        num_loc: number of locations (customers) in the TSP
        min_loc: minimum value for the location coordinates
        max_loc: maximum value for the location coordinates
        base_speed: average speed
        min_speed_amplitude: minimum amplitude of speed variation
        max_speed_amplitude: maximum amplitude of speed variation
        min_period: minimum period of the speed cycle
        max_period: maximum period of the speed cycle
        loc_distribution: distribution for the location coordinates
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        base_speed: float = 1.0,
        min_speed_amplitude: float = 0.2,
        max_speed_amplitude: float = 0.5,
        min_period: float = 1.0,
        max_period: float = 5.0,
        loc_distribution: int | float | str | type | Callable = Uniform,
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        
        self.base_speed = base_speed
        self.min_speed_amplitude = min_speed_amplitude
        self.max_speed_amplitude = max_speed_amplitude
        self.min_period = min_period
        self.max_period = max_period

        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **kwargs
            )

    def _generate(self, batch_size) -> TensorDict:
        # Sample locations [batch_size, num_loc, 2]
        locs = self.loc_sampler.sample((*batch_size, self.num_loc, 2))
        
        # Sample speed profile parameters per instance
        # speed_amplitude: [batch_size]
        speed_amplitude = torch.empty((*batch_size, 1)).uniform_(
            self.min_speed_amplitude, self.max_speed_amplitude
        )
        
        # period: [batch_size]
        period = torch.empty((*batch_size, 1)).uniform_(
            self.min_period, self.max_period
        )
        
        # phase: [batch_size] - random start of the cycle
        phase = torch.empty((*batch_size, 1)).uniform_(0, 2 * 3.14159)

        return TensorDict(
            {
                "locs": locs,
                "speed_amplitude": speed_amplitude,
                "period": period,
                "phase": phase,
                "base_speed": torch.full((*batch_size, 1), self.base_speed),
            },
            batch_size=batch_size,
        )


class TDTSPBenchmarkGenerator(Generator):
    """
    Generator for loading TDTSP benchmark instances (BonnTour / VRP-TDT).
    Similar to DVRPBenchmarkGenerator but adapted for TDTSP matrix environment.
    
    Loads matrix data if available (e.g. berlin_10_tt.json.bz2) or calculates it?
    Actually TDTSPEnv supports speed profile OR matrix.
    TDTSPMatrixEnv supports matrix.
    
    For benchmark, we usually have a travel time matrix.
    Let's implement loading the matrix.
    """
    def __init__(self, data_dir: str, instance_name: str = "berlin_10", **kwargs):
        self.data_dir = data_dir
        self.instance_name = instance_name
        self.num_loc = 0 # Will be updated
        self.min_loc = 0.0
        self.max_loc = 1.0
        
        # Pre-load to get metadata
        # We need to read the JSON file to get locations and matrix
        # Or just locations if we use TDTSPEnv (non-matrix) with fitted params?
        # But benchmark data is usually real-world data which is best represented by matrix.
        # So we should use TDTSPMatrixEnv and this generator should output matrix.
        
        super().__init__(**kwargs)

    def _generate(self, batch_size) -> TensorDict:
        import os
        import json
        import bz2
        import numpy as np
        
        # 1. Load Locations (from _pdp.json or .json)
        pdp_path = os.path.join(self.data_dir, "instances", f"{self.instance_name}_pdp.json")
        if not os.path.exists(pdp_path):
             pdp_path = os.path.join(self.data_dir, "instances", f"{self.instance_name}.json")
             
        with open(pdp_path, 'r') as f:
            data = json.load(f)
            
        # Parse Depots and Items
        depot_lat = data['depot']['latitude']
        depot_lon = data['depot']['longitude']
        
        # ID Mapping
        id_to_idx = {"depot": 0}
        
        # Collect all locations
        addresses = {}
        if 'addresses' in data:
            for k, v in data['addresses'].items():
                addresses[k] = (v['latitude'], v['longitude'])
                
        items = []
        if 'items' in data:
            # Sort keys to ensure deterministic order (1, 2, ..., N)
            item_keys = sorted(data['items'].keys(), key=lambda x: int(x))
            for k in item_keys:
                v = data['items'][k]
                id_to_idx[k] = len(items) + 1 # 1-based index for items
                
                if 'pickup_address' in v:
                    addr_id = v['pickup_address']
                    lat, lon = addresses[addr_id]
                else:
                    lat = v['latitude']
                    lon = v['longitude']
                items.append((lat, lon))
        
        # Total nodes = 1 (depot) + N (items)
        all_locs = [(depot_lat, depot_lon)] + items
        num_loc = len(all_locs)
        self.num_loc = num_loc # Update
        
        # 2. Load Travel Time Matrix
        # berlin_10_tt.json.bz2
        tt_path = os.path.join(self.data_dir, "instances", f"{self.instance_name}_tt.json.bz2")
        
        if os.path.exists(tt_path):
            with bz2.open(tt_path, "rt") as f:
                tt_data = json.load(f) # List of edges
            
            # Construct Matrix
            # We need to define time resolution
            # Max time ~ 24h = 86400000 ms
            # Time step = 10 mins = 600000 ms
            # Num steps = 144
            
            time_step_ms = 600000.0
            max_time_ms = 86400000.0
            num_steps = int(max_time_ms / time_step_ms) + 1
            time_grid = np.linspace(0, max_time_ms, num_steps)
            
            matrix = torch.zeros((num_loc, num_loc, num_steps), dtype=torch.float32)
            
            # Helper to parse edges
            # tt_data is a list of dicts: {'from': 'depot', 'to': 'depot', 'atf': ...}
            
            for edge in tt_data:
                u_id = edge['from']
                v_id = edge['to']
                
                # Check if nodes are in our instance
                if u_id not in id_to_idx or v_id not in id_to_idx:
                    continue
                    
                u = id_to_idx[u_id]
                v = id_to_idx[v_id]
                
                atf = edge['atf']
                L = np.array(atf['atf_leave_time'], dtype=np.float64)
                A = np.array(atf['atf_arrive_time'], dtype=np.float64)
                
                # Interpolate Arrive Time at time_grid
                A_interp = np.interp(time_grid, L, A)
                
                # Travel Time = Arrive - Leave
                travel_time = A_interp - time_grid
                
                # Assign to matrix
                matrix[u, v] = torch.from_numpy(travel_time).float()
            
            # Normalize
            # Scale time to hours? Or [0, 1]?
            # Let's use hours for interpretability.
            scale = 3600000.0 # 1 hour
            
            matrix = matrix / scale
            time_step_duration = torch.tensor(time_step_ms / scale, dtype=torch.float32)
            
            # Normalize Locations (for embedding)
            lats = [x[0] for x in all_locs]
            lons = [x[1] for x in all_locs]
            min_lat, max_lat = min(lats), max(lats)
            min_lon, max_lon = min(lons), max(lons)
            
            def norm_lat(l): return (l - min_lat) / (max_lat - min_lat + 1e-6)
            def norm_lon(l): return (l - min_lon) / (max_lon - min_lon + 1e-6)
            
            locs_norm = [[norm_lat(x[0]), norm_lon(x[1])] for x in all_locs]
            locs_tensor = torch.tensor(locs_norm, dtype=torch.float32)
            
            # Batchify
            def expand_to_batch(tensor, batch_size):
                 for _ in range(len(batch_size)):
                     tensor = tensor.unsqueeze(0)
                 return tensor.expand(*batch_size, *tensor.shape[len(batch_size):]).clone()

            # Map instance nodes to matrix indices
            # Here we assume 1:1 mapping 0..N-1
            locs_idx = torch.arange(num_loc).unsqueeze(0).expand(*batch_size, -1)
            
            return TensorDict(
                {
                    "locs": expand_to_batch(locs_tensor, batch_size),
                    "locs_idx": locs_idx,
                    "travel_time_matrix": expand_to_batch(matrix, batch_size), # Shared matrix expanded
                    "time_step_duration": expand_to_batch(time_step_duration, batch_size),
                },
                batch_size=batch_size
            )
            
        else:
            # Fallback or Error
            raise FileNotFoundError(f"Travel time matrix not found: {tt_path}")
