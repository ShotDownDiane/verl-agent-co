from typing import Optional

import torch
import math

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger

from .generator import TDTSPGenerator
from .render import render

log = get_pylogger(__name__)


class TDTSPEnv(RL4COEnvBase):
    """Time-Dependent Traveling Salesman Problem (TDTSP) environment.
    The travel time between cities depends on the time of day (current cumulative time).
    
    The speed model is defined in the generator as:
        v(t) = base_speed + speed_amplitude * sin(2*pi*t/period + phase)
    
    The travel time T for a distance d starting at time t_start is approximated by:
        T = d / v(t_start)
    
    Objective: Minimize total travel time (makespan).
    
    Observations:
        - locations of each customer.
        - current location.
        - current time.
        - visited mask.
        - speed profile parameters (implicit in environment dynamics, but potentially observable).

    Args:
        generator: TDTSPGenerator instance
        generator_params: parameters for the generator
    """

    name = "tdtsp"

    def __init__(
        self,
        generator: TDTSPGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ):
        super().__init__(**kwargs)
        if generator is None:
            generator = TDTSPGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _calculate_travel_time_integral(self, dist, start_time, base_speed, amp, period, phase):
        """
        Calculate travel time using integral of velocity to satisfy FIFO property.
        Solve: Integral_{t_start}^{t_end} v(t) dt = distance
        v(t) = base + amp * sin(omega * t + phi)
        F(t) = base * t - (amp/omega) * cos(omega * t + phi)
        F(t_end) - F(t_start) = distance
        """
        omega = 2 * math.pi / period
        
        # F(t) function
        def get_dist_integral(t):
            return base_speed * t - (amp / omega) * torch.cos(omega * t + phase)
            
        # v(t) function
        def get_velocity(t):
            v = base_speed + amp * torch.sin(omega * t + phase)
            return torch.clamp(v, min=0.1) # Ensure positive speed
            
        # Target value for F(t_end)
        F_start = get_dist_integral(start_time)
        target_F = F_start + dist
        
        # Newton's method to solve for t_end
        # Initial guess: t_end = t_start + dist / v(t_start)
        t_curr = start_time + dist / get_velocity(start_time)
        
        # 3 iterations is usually enough for high precision
        for _ in range(5):
            F_curr = get_dist_integral(t_curr)
            v_curr = get_velocity(t_curr)
            t_curr = t_curr - (F_curr - target_F) / v_curr
            
        return t_curr - start_time

    def _get_travel_time(self, td, prev_node_idx, current_node_idx, is_first_step=None, current_time=None):
        locs = td["locs"]
        
        # Get previous node location
        prev_loc = locs.gather(1, prev_node_idx.view(-1, 1, 1).expand(-1, -1, 2)).squeeze(1)
        
        # Get current node location (target)
        curr_loc = locs.gather(1, current_node_idx.view(-1, 1, 1).expand(-1, -1, 2)).squeeze(1)
        
        # Calculate Euclidean distance
        dist = (curr_loc - prev_loc).norm(p=2, dim=-1)
        
        if is_first_step is not None:
            dist[is_first_step] = 0.0
            
        # Calculate Speed params
        if current_time is None:
            current_time = td["current_time"]
            
        base_speed = td["base_speed"].squeeze(-1)
        amp = td["speed_amplitude"].squeeze(-1)
        period = td["period"].squeeze(-1)
        phase = td["phase"].squeeze(-1)
        
        # Calculate travel time using integral method (FIFO consistent)
        return self._calculate_travel_time_integral(
            dist, current_time, base_speed, amp, period, phase
        )

    def _step(self, td: TensorDict) -> TensorDict:
        current_node_idx = td["action"]
        prev_node_idx = td["current_node"]
        
        # Handle first step logic for distance/travel time
        is_first_step = (td["i"] == 0)
        
        travel_time = self._get_travel_time(td, prev_node_idx, current_node_idx, is_first_step)
        
        # Update masked nodes
        if is_first_step.any():
            # After the first step (picking node 0), all other nodes become available
            available = torch.ones_like(td["action_mask"])
            available = available.scatter(
                -1, current_node_idx.unsqueeze(-1).expand_as(td["action_mask"]), 0
            )
        else:
            # Set not visited to 0 (i.e., we visited the node)
            available = td["action_mask"].scatter(
                -1, current_node_idx.unsqueeze(-1).expand_as(td["action_mask"]), 0
            )
        
        # Check if done: all nodes except node 0 (depot) are visited
        done = torch.sum(available[..., 1:], dim=-1) == 0

        # Update first node if it is the first step
        first_node_idx = td["first_node"]
        first_node_idx[is_first_step] = current_node_idx[is_first_step]
        
        # Update time
        current_time = td["current_time"]
        new_time = current_time + travel_time
        
        # If done, terminal reward will be calculated by _get_reward
        reward = torch.zeros_like(done, dtype=torch.float32)

        td.update(
            {
                "first_node": first_node_idx,
                "current_node": current_node_idx,
                "current_time": new_time,
                "i": td["i"] + 1,
                "action_mask": available,
                "reward": reward,
                "done": done,
            },
        )
        return td

    def _reset(self, td: Optional[TensorDict] = None, **kwargs) -> TensorDict:
        # Initialize locations
        batch_size = td.batch_size if td is not None else kwargs.get("batch_size")
        device = td.device if td is not None else "cpu"
        
        # Create action mask (force start at 0)
        num_loc = td["locs"].shape[-2]
        available = torch.zeros(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )
        available[..., 0] = True # Only node 0 (depot) is available at first step
        
        # Initial node (can be random or fixed to 0). 
        # Usually we let the agent pick the first node, or we start at 0.
        # RL4CO TSPEnv typically allows picking the first node.
        # But here, we have "current_node". 
        # If the agent picks the first node at step 0, we are "at" that node.
        # Wait, standard TSPEnv logic:
        # Step 0: Agent picks Node X. This is the start of the tour.
        # Reward: 0.
        # Step 1: Agent picks Node Y. Distance X->Y added.
        # ...
        
        # To make it consistent with _step logic which calculates distance from `current_node` to `action`,
        # we need a placeholder `current_node` for step 0.
        # Actually, in standard TSPEnv, the first action sets the `first_node` and `current_node`.
        # No distance is accumulated for the first step (it's the start point).
        
        current_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        
        if td is not None and "current_time" in td.keys():
            current_time = td["current_time"]
        else:
            current_time = torch.zeros((*batch_size,), dtype=torch.float32, device=device)
            
        i = torch.zeros((*batch_size,), dtype=torch.int64, device=device)

        return TensorDict(
            {
                "locs": td["locs"],
                "speed_amplitude": td["speed_amplitude"],
                "period": td["period"],
                "phase": td["phase"],
                "base_speed": td["base_speed"],
                "first_node": current_node, # Will be updated at first step
                "current_node": current_node,
                "current_time": current_time,
                "i": i,
                "action_mask": available,
            },
            batch_size=batch_size,
        )

    def _make_spec(self, generator: TDTSPGenerator):
        self.observation_spec = Composite(
            locs=Bounded(
                low=generator.min_loc,
                high=generator.max_loc,
                shape=(generator.num_loc, 2),
                dtype=torch.float32,
            ),
            first_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            current_node=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            current_time=Unbounded(
                shape=(1),
                dtype=torch.float32,
            ),
            i=Unbounded(
                shape=(1),
                dtype=torch.int64,
            ),
            action_mask=Unbounded(
                shape=(generator.num_loc),
                dtype=torch.bool,
            ),
            # Generator params might be part of obs if we want agent to see them
            base_speed=Unbounded(shape=(1), dtype=torch.float32),
            speed_amplitude=Unbounded(shape=(1), dtype=torch.float32),
            period=Unbounded(shape=(1), dtype=torch.float32),
            phase=Unbounded(shape=(1), dtype=torch.float32),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1),
            dtype=torch.int64,
            low=0,
            high=generator.num_loc,
        )
        self.reward_spec = Unbounded(shape=(1))
        self.done_spec = Unbounded(shape=(1), dtype=torch.bool)
    
    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """
        Recalculate reward (negative makespan) for the given sequence of actions.
        We must re-simulate the path because travel time is time-dependent.
        """
        locs = td["locs"]
        batch_size = locs.shape[0]
        
        # Generator params
        base_speed = td["base_speed"].squeeze(-1)
        amp = td["speed_amplitude"].squeeze(-1)
        period = td["period"].squeeze(-1)
        phase = td["phase"].squeeze(-1)
        
        # Start at time 0
        current_time = torch.zeros(batch_size, device=locs.device)
        
        # We assume the first action is the start node (spawn point).
        # So travel to actions[:, 0] takes 0 time (or we are already there).
        # Then we travel to actions[:, 1], etc.
        
        # Get locations ordered by action
        # actions: [batch_size, num_loc]
        # locs_ordered: [batch_size, num_loc, 2]
        locs_ordered = locs.gather(1, actions.unsqueeze(-1).expand(-1, -1, 2))
        
        # Iterate through the tour
        # From node i to node i+1
        for i in range(actions.shape[1] - 1):
            curr_loc = locs_ordered[:, i]
            next_loc = locs_ordered[:, i+1]
            
            dist = (next_loc - curr_loc).norm(p=2, dim=-1)
            
            travel_time = self._calculate_travel_time_integral(
                dist, current_time, base_speed, amp, period, phase
            )
            current_time = current_time + travel_time
            
        # Return to start (actions[:, -1] -> actions[:, 0])
        curr_loc = locs_ordered[:, -1]
        next_loc = locs_ordered[:, 0]
        dist = (next_loc - curr_loc).norm(p=2, dim=-1)
        
        travel_time = self._calculate_travel_time_integral(
            dist, current_time, base_speed, amp, period, phase
        )
        current_time = current_time + travel_time
        
        # Convert current_time to hours
        final_makespan = current_time / 3600
        
        return -final_makespan

    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor) -> None:
        """Check that solution is valid: nodes are visited exactly once"""
        assert (
            torch.arange(actions.size(1), out=actions.data.new())
            .view(1, -1)
            .expand_as(actions)
            == actions.data.sort(1)[0]
        ).all(), "Invalid tour"

    def render(self, td: TensorDict, actions: torch.Tensor = None, ax=None):
        return render(td, actions, ax)


class TDTSPMatrixEnv(TDTSPEnv):
    """
    TDTSP Environment using a pre-computed travel time matrix.
    The matrix is of shape [N_matrix, N_matrix, T_steps].
    The environment handles mapping from instance node indices to matrix location indices.
    """
    name = "tdtsp_matrix"
    
    def _get_travel_time(self, td, prev_node_idx, current_node_idx, is_first_step=None, current_time=None):
        if current_time is None:
            current_time = td["current_time"]
        
        u_physical = prev_node_idx
        v_physical = current_node_idx
        
        # 2. Calculate Time Step s
        # s = int(current_time // duration)
        duration = td["time_step_duration"] # Scalar or [batch_size]
        
        # Ensure duration is shaped correctly
        if duration.dim() == 0:
            duration = duration.unsqueeze(0).expand_as(current_time)
        elif duration.dim() == 1 and duration.shape[0] == 1:
            duration = duration.expand_as(current_time)
            
        # Add safety for NaN current_time
        safe_time = torch.nan_to_num(current_time, nan=0.0)
        s = (safe_time // duration).long()
        s = s.clamp(min=0) # Ensure s is at least 0
        
        # 3. Look up Matrix
        # matrix: [batch_size, N_matrix, N_matrix, T_steps] (if broadcasted)
        # or [N_matrix, N_matrix, T_steps] (if shared)
        
        matrix = td["travel_time_matrix"]
        
        # Handle matrix shape
        # Clamp s to max time steps
        max_s = matrix.shape[-1] - 1
        s = s.clamp(max=max_s)
        
        # Check if matrix has batch dim corresponding to current_time
        if matrix.dim() == 4:
            batch_size = current_time.shape[0]
            batch_indices = torch.arange(batch_size, device=current_time.device)
            travel_time = matrix[batch_indices, u_physical, v_physical, s]
        else:
            # Shared matrix [N, N, M]
            travel_time = matrix[u_physical, v_physical, s]
            
        # Mask first step
        if is_first_step is not None:
            travel_time[is_first_step] = 0.0
            
        return travel_time.float()

    def _reset(self, td: Optional[TensorDict] = None, batch_size=None) -> TensorDict:
        # Initialize locations
        device = td.device
        
        # Create action mask (force start at 0)
        # td["locs_idx"] is [batch, N]
        num_loc = td["locs_idx"].shape[-1] if "locs_idx" in td.keys() else td["locs"].shape[-2]
        
        available = torch.zeros(
            (*batch_size, num_loc), dtype=torch.bool, device=device
        )
        available[..., 0] = True # Only node 0 (depot) is available at first step
        
        current_node = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        
        if td is not None and "current_time" in td.keys():
            current_time = td["current_time"]
        else:
            current_time = torch.zeros((*batch_size,), dtype=torch.float32, device=device)
            
        i = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        
        return TensorDict(
            {
                "locs": td["locs"],
                "locs_idx": td["locs_idx"] if "locs_idx" in td.keys() else None,
                "travel_time_matrix": td["travel_time_matrix"],
                "time_step_duration": td["time_step_duration"],
                "first_node": current_node,
                "current_node": current_node,
                "current_time": current_time,
                "i": i,
                "action_mask": available,
                "reward": torch.zeros((*batch_size,), dtype=torch.float32, device=device),
            },
            batch_size=batch_size,
        )

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        """
        Recalculate reward (negative makespan) using matrix lookup.
        """
        
        # Check validity
        self.check_solution_validity(td, actions)
        
        batch_size = actions.shape[0]
        device = actions.device
        
        current_time = torch.zeros(batch_size, device=device)
        
        # Iterate through the tour
        # actions: [batch_size, num_loc]
        # We assume start at depot (0) at time 0.
        
        prev_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        for i in range(actions.shape[1]):
            curr_node = actions[:, i]
            # Travel time from prev to curr
            tt = self._get_travel_time(td, prev_node, curr_node, current_time=current_time)
            current_time = current_time + tt
            prev_node = curr_node
            
        # Return to start (depot 0)
        curr_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        tt = self._get_travel_time(td, prev_node, curr_node, current_time=current_time)
        current_time = current_time + tt
        
        return -current_time / 3600.0
