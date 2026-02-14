from typing import Callable, Union, Optional

import torch
import numpy as np
from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class TDTSPMatrixGenerator(Generator):
    """
    Data generator for the Time-Dependent TSP (TDTSP).
    Focuses on generating dynamic Travel Time Matrices without hard time window constraints.
    
    Output:
        - locs: [B, N, 2] (Includes depot at index 0)
        - travel_time_matrix: [B, N, N, T]
        - time_windows: [B, N, 2] (Dummy windows: 0 to Horizon)
    """

    def __init__(
        self,
        num_loc: int = 20,
        min_loc: float = 0.0,
        max_loc: float = 1.0,
        
        # Distribution Params
        loc_distribution: int | float | str | type | Callable = "mixed", # 'uniform', 'cluster', 'mixed'
        num_clusters: int = 3,
        cluster_std: float = 0.08,
        
        # Matrix / Physics Params
        num_time_steps: int = 100,
        time_step_duration: float = 600.0, # 10 mins
        distance_scale: float = 100000.0,  # Map scale (meters)
        base_speed: float = 15.0,          # Avg speed (m/s)
        
        # Speed Profile Params
        min_speed_amplitude: float = 0.2,
        max_speed_amplitude: float = 0.5,
        min_period: float = 0.8, # Factor of horizon
        max_period: float = 1.5,
        
        # Caching Settings
        use_cached_matrix: bool = True,
        
        **kwargs,
    ):
        self.num_loc = num_loc
        self.min_loc = min_loc
        self.max_loc = max_loc
        
        # Location distribution
        if kwargs.get("loc_sampler", None) is not None:
            self.loc_sampler = kwargs["loc_sampler"]
        else:
            # Map num_clusters to what get_sampler expects for different distributions
            sampler_kwargs = kwargs.copy()
            if "n_cluster" not in sampler_kwargs:
                sampler_kwargs["n_cluster"] = num_clusters
            if "n_cluster_mix" not in sampler_kwargs:
                sampler_kwargs["n_cluster_mix"] = num_clusters
            
            self.loc_sampler = get_sampler(
                "loc", loc_distribution, min_loc, max_loc, **sampler_kwargs
            )
        
        # Matrix settings
        self.num_time_steps = num_time_steps
        self.time_step_duration = time_step_duration
        self.horizon = num_time_steps * time_step_duration
        self.distance_scale = distance_scale
        
        # Physics settings
        self.base_speed = base_speed
        self.min_speed_amplitude = min_speed_amplitude
        self.max_speed_amplitude = max_speed_amplitude
        self.min_period = min_period
        self.max_period = max_period

        # Caching state
        self.use_cached_matrix = use_cached_matrix
        self._cached_data = None

    def _sample_locations(self, batch_size, device):
        """Generates locations using the configured sampler."""
        # RL4CO samplers return [bs, num_loc, 2]
        return self.loc_sampler.sample((*batch_size, self.num_loc, 2)).to(device)

    def _generate_matrix(self, locs, batch_size, device):
        """Constructs 4D Travel Time Matrix efficiently using broadcasting and vectorization."""
        bs = batch_size[0]
        N = self.num_loc
        T = self.num_time_steps
        
        # 1. Physics Parameters (Vectorized sampling)
        # amp: [bs, 1]
        amp = torch.rand((bs, 1), device=device) * \
              (self.max_speed_amplitude - self.min_speed_amplitude) + self.min_speed_amplitude
        
        # period: [bs, 1]
        period = (torch.rand((bs, 1), device=device) * \
                 (self.max_period - self.min_period) + self.min_period) * self.horizon
        
        # phase: [bs, 1]
        phase = torch.rand((bs, 1), device=device) * (2 * np.pi)
        
        # 2. Distance Matrix (Meters)
        # Use torch.cdist for optimized Euclidean distance calculation
        # locs: [bs, N, 2] -> dists: [bs, N, N]
        dists = torch.cdist(locs, locs, p=2) * self.distance_scale
        
        # 3. Speed Matrix over Time [bs, T]
        t_steps = torch.arange(T, device=device).float()
        t_sec = t_steps * self.time_step_duration # [T]
        
        # sine_wave: [bs, T]
        # (2 * pi * t / period + phase)
        # period and phase are [bs, 1], t_sec is [T]
        arg = (2 * np.pi * t_sec.unsqueeze(0) / period) + phase
        speed_profile = self.base_speed * (1.0 + amp * torch.sin(arg))
        speed_profile = torch.clamp(speed_profile, min=0.1) # [bs, T]
        
        # 4. Travel Time Matrix [bs, N, N, T]
        # We want: dists[bs, N, N] / speed_profile[bs, T]
        # Broadcast dists to [bs, N, N, 1] and speed_profile to [bs, 1, 1, T]
        tt_matrix = dists.unsqueeze(-1) / speed_profile.view(bs, 1, 1, T)
        
        # Mask diagonal (set distance to 0, travel time becomes 0)
        # This is faster than masked_fill on the full 4D matrix
        return tt_matrix

    def _generate(self, batch_size) -> TensorDict:
        # Use CPU for large matrix generation to prevent GPU OOM
        device = torch.device("cpu")
        bs = batch_size[0]
        
        # Check if we can reuse cached matrix
        if self.use_cached_matrix and self._cached_data is not None:
            cached_bs = self._cached_data["travel_time_matrix"].shape[0]
            if cached_bs >= bs:
                log.info(f"Reusing cached travel_time_matrix for batch size {bs} (Cached: {cached_bs})")
                return self._cached_data[:bs].clone()
            else:
                log.info(f"Batch size changed from {cached_bs} to {bs}. Regenerating matrix.")

        locs = self._sample_locations(batch_size, device)
        tt_matrix = self._generate_matrix(locs, batch_size, device)
        
        # Dummy Time Windows (0 to Horizon)
        start_times = torch.zeros((bs, self.num_loc), device=device)
        end_times = torch.full((bs, self.num_loc), self.horizon, device=device)
        time_windows = torch.stack([start_times, end_times], dim=-1)
        
        res = TensorDict({
            "locs": locs,
            "travel_time_matrix": tt_matrix,
            "time_windows": time_windows,
            "time_step_duration": torch.full((bs,), self.time_step_duration, device=device),
            "min_time": torch.zeros((bs,), device=device),
        }, batch_size=batch_size)

        # Cache the result if enabled
        if self.use_cached_matrix:
            self._cached_data = res.clone()
            log.info(f"Cached travel_time_matrix for batch size {bs}")
            
        return res


class TDTSPTWGenerator(TDTSPMatrixGenerator):
    """
    Data generator for the Time-Dependent TSP with Time Windows (TDTSPTW).
    Inherits matrix generation from TDTSPMatrixGenerator but adds
    Simulation-Based Time Window Generation to ensure feasibility.
    """

    def __init__(
        self,
        tw_width_mean: float = 3600.0, # 1 hour
        service_time: float = 180.0,   # 3 mins
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tw_width_mean = tw_width_mean
        self.service_time = service_time

    def _generate(self, batch_size) -> TensorDict:
        device = torch.device("cpu")
        bs = batch_size[0]
        N = self.num_loc
        
        # Check if we can reuse cached matrix
        if self.use_cached_matrix and self._cached_data is not None:
            cached_bs = self._cached_data["travel_time_matrix"].shape[0]
            if cached_bs >= bs:
                log.info(f"Reusing cached matrix for TDTSPTW (Batch: {bs} from {cached_bs})")
                res = self._cached_data[:bs].clone()
                
                # Regenerate Time Windows for diversity
                perm = torch.argsort(torch.rand((bs, N-1), device=device), dim=-1) + 1
                curr_time = torch.zeros(bs, device=device)
                prev_node = torch.zeros(bs, dtype=torch.long, device=device)
                arrival_times = torch.zeros((bs, N), device=device)
                tt_matrix = res["travel_time_matrix"]
                
                for i in range(N-1):
                    curr_node = perm[:, i]
                    s = (curr_time / self.time_step_duration).long().clamp(0, self.num_time_steps - 1)
                    batch_idx = torch.arange(bs, device=device)
                    tt = tt_matrix[batch_idx, prev_node, curr_node, s]
                    arr = curr_time + tt
                    arrival_times[batch_idx, curr_node] = arr
                    curr_time = arr + self.service_time
                    prev_node = curr_node
                
                margin = torch.rand((bs, N), device=device) * self.tw_width_mean
                start_times = torch.max(torch.zeros_like(arrival_times), arrival_times - margin)
                end_times = arrival_times + margin
                start_times[:, 0] = 0.0
                end_times[:, 0] = self.horizon
                res["time_windows"] = torch.stack([start_times, end_times], dim=-1)
                
                return res
            else:
                log.info(f"Batch size changed from {cached_bs} to {bs}. Regenerating everything.")

        # 1. Generate Locs & Matrix (Reuse base logic)
        locs = self._sample_locations(batch_size, device)
        tt_matrix = self._generate_matrix(locs, batch_size, device)
        
        # 2. Simulation to ensure Feasibility
        # Generate a random valid tour: 0 -> p1 -> p2 ... -> pn -> 0
        # Indices 1 to N-1 are customers
        perm = torch.argsort(torch.rand((bs, N-1), device=device), dim=-1) + 1
        
        curr_time = torch.zeros(bs, device=device)
        prev_node = torch.zeros(bs, dtype=torch.long, device=device) # Start at Depot (0)
        
        arrival_times = torch.zeros((bs, N), device=device)
        
        # Simulate step-by-step
        for i in range(N-1):
            curr_node = perm[:, i]
            
            # Get time step index s
            s = (curr_time / self.time_step_duration).long().clamp(0, self.num_time_steps - 1)
            
            # Lookup travel time: matrix[b, prev, curr, s]
            batch_idx = torch.arange(bs, device=device)
            tt = tt_matrix[batch_idx, prev_node, curr_node, s]
            
            arr = curr_time + tt
            arrival_times[batch_idx, curr_node] = arr
            
            # Service & Move on
            curr_time = arr + self.service_time
            prev_node = curr_node
            
        # 3. Back-calculate Time Windows based on Arrival Times
        # E = Arrival - random, L = Arrival + random
        margin = torch.rand((bs, N), device=device) * self.tw_width_mean
        
        # Ensure non-negative start
        start_times = torch.max(torch.zeros_like(arrival_times), arrival_times - margin)
        end_times = arrival_times + margin
        
        # Fix Depot TW (Full Horizon)
        start_times[:, 0] = 0.0
        end_times[:, 0] = self.horizon
        
        time_windows = torch.stack([start_times, end_times], dim=-1)
        
        res = TensorDict({
            "locs": locs,
            "travel_time_matrix": tt_matrix,
            "time_windows": time_windows,
            "time_step_duration": torch.full((bs,), self.time_step_duration, device=device),
            "min_time": torch.zeros((bs,), device=device),
            "service_time": torch.full((bs,), self.service_time, device=device),
        }, batch_size=batch_size)

        # Cache the result if enabled
        if self.use_cached_matrix:
            self._cached_data = res.clone()
            log.info(f"Cached matrix for TDTSPTW (Batch: {bs})")

        return res