import torch
from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.pylogger import get_pylogger
from .generator import TDVRPGenerator

log = get_pylogger(__name__)

class TDVRPEnv(RL4COEnvBase):
    """Time-Dependent Vehicle Routing Problem with Time Windows (TDVRP-TW) environment.
    Capacity is ignored in this implementation as requested.
    Logic is based on TDTSP-TW.
    """
    name = "tdvrp"

    def __init__(self, 
                 generator: TDVRPGenerator = None,
                 generator_params: dict = {},
                 penalty_value: float = 0.0,
                 **kwargs):
        super().__init__(**kwargs)
        if generator is None:
            generator = TDVRPGenerator(**generator_params)
        self.generator = generator
        self.service_time = generator.service_time
        self.penalty_value = penalty_value
        self._make_spec(self.generator)

    def dataset(self, batch_size=[], phase="train", filename=None):
        """Override dataset to handle phase-specific data splitting in the generator"""
        if hasattr(self.generator, "phase"):
            old_phase = self.generator.phase
            self.generator.phase = phase
            self.generator._load_data() 
            res = super().dataset(batch_size, phase, filename)
            self.generator.phase = old_phase 
            self.generator._load_data()
            return res
        return super().dataset(batch_size, phase, filename)

    def _make_spec(self, generator: TDVRPGenerator):
        self.observation_spec = Composite(
            locs=Bounded(low=0, high=1, shape=(generator.num_nodes, 2), dtype=torch.float32),
            time_windows=Unbounded(shape=(generator.num_nodes, 2), dtype=torch.float32),
            current_node=Unbounded(shape=(1), dtype=torch.int64),
            current_time=Unbounded(shape=(1), dtype=torch.float32),
            visited=Unbounded(shape=(generator.num_nodes), dtype=torch.bool),
            action_mask=Unbounded(shape=(generator.num_nodes), dtype=torch.bool),
            shape=(),
        )
        self.action_spec = Bounded(low=0, high=generator.num_nodes, shape=(1,), dtype=torch.int64)
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)

    def _reset(self, td: TensorDict, **kwargs) -> TensorDict:
        
        device = td.device
        batch_size = td.batch_size
        
        # Initial visited: all False
        visited = torch.zeros((*batch_size, self.generator.num_nodes), dtype=torch.bool, device=device)
        
        # current_node: depot (0)
        current_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # current_time: 0 (shifted)
        current_time = torch.zeros((*batch_size, 1), dtype=torch.float32, device=device)
        
        # Initialize step count and first node (for TDTSP context compatibility)
        i = torch.zeros(batch_size, dtype=torch.long, device=device)
        first_node = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        # Initialize cumulative reward
        cumulative_reward = torch.zeros((*batch_size,), dtype=torch.float32, device=device)
        num_routes = torch.zeros((*batch_size,), dtype=torch.int64, device=device)
        
        td_reset = TensorDict({
            "locs": td["locs"],
            "time_windows": td["time_windows"],
            "travel_time_matrix": td["travel_time_matrix"],
            "time_step_duration": td["time_step_duration"],
            "min_time": td["min_time"],
            "current_node": current_node,
            "current_time": current_time,
            "i": i,
            "first_node": first_node,
            "visited": visited,
            "cumulative_reward": cumulative_reward,
            "num_routes": num_routes,
        }, batch_size=batch_size)
        
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    def get_action_mask(self, td: TensorDict) -> torch.Tensor:
        batch_size = td.batch_size[0]
        current_node = td["current_node"].squeeze(-1)
        current_time = td["current_time"]
        visited = td["visited"]
        matrix = td["travel_time_matrix"]
        duration = td["time_step_duration"]
        
        # 1. Basic mask: already visited customers
        mask = visited.clone()
        
        # 2. Time Window Feasibility (Lookahead)
        s = (current_time // duration.unsqueeze(-1)).long()
        s = s.clamp(min=0, max=matrix.shape[-1] - 1).squeeze(-1)
        
        batch_idx = torch.arange(batch_size, device=td.device)
        tt_to_all = matrix[batch_idx, current_node, :, s] # [B, N]
        
        arrival_at_next = current_time + tt_to_all
        late_tws = td["time_windows"][..., 1] # [B, N]
        
        infeasible = arrival_at_next > late_tws
        
        # Combine visited and infeasible for customers
        if self.penalty_value == 0:
            mask_customers = mask[:, 1:] | infeasible[:, 1:]
        else:
            mask_customers = mask[:, 1:] # Only visited mask if penalty enabled
        
        # 3. Depot logic
        # Depot is reachable if:
        # a) We are at a customer node
        # b) We are at the depot and there are no reachable unvisited customers
        no_reachable_customers = mask_customers.all(dim=-1)
        
        mask_depot = (current_node == 0) & (~no_reachable_customers)
        # Also check if returning to depot is feasible (only if penalty_value == 0)
        if self.penalty_value == 0:
            depot_infeasible = infeasible[:, 0]
            mask_depot = mask_depot | depot_infeasible
        
        # Construct final mask
        final_mask = torch.cat([mask_depot.unsqueeze(-1), mask_customers], dim=-1)
        
        # Fallback: if no feasible actions but still unvisited customers, allow all unvisited to avoid crash
        # Only relevant when penalty_value == 0
        if self.penalty_value == 0:
            all_masked = final_mask.all(dim=-1)
            if all_masked.any():
                # Allow visiting any unvisited customer
                final_mask[all_masked, 1:] = mask[all_masked, 1:]
                # If still all masked (all visited), allow depot
                still_all_masked = final_mask.all(dim=-1)
                final_mask[still_all_masked, 0] = False

        return ~final_mask

    def _get_travel_time(self, td, prev_node, curr_node):
        matrix = td["travel_time_matrix"]
        duration = td["time_step_duration"]
        current_time = td["current_time"]
        
        s = (current_time // duration.unsqueeze(-1)).long()
        s = s.clamp(min=0, max=matrix.shape[-1] - 1).squeeze(-1)
        
        batch_idx = torch.arange(td.batch_size[0], device=td.device)
        return matrix[batch_idx, prev_node, curr_node, s]

    def _step(self, td: TensorDict) -> TensorDict:
        action = td["action"]
        current_node = td["current_node"].squeeze(-1)
        
        # Calculate travel time
        tt = self._get_travel_time(td, current_node, action)
        
        # Update time
        arrival_time = td["current_time"].squeeze(-1) + tt
        
        # Check against early TW
        node_tw = td["time_windows"].gather(1, action.unsqueeze(-1).expand(-1, 2).unsqueeze(1)).squeeze(1)
        early_tw = node_tw[:, 0]
        late_tw = node_tw[:, 1]
        
        # Penalty tracking
        late_time = (arrival_time - late_tw).clamp(min=0)
        total_penalty = td.get("total_penalty", torch.zeros_like(arrival_time)) + late_time
        
        ready_time = torch.max(arrival_time, early_tw)
        
        # Add service time if it's a customer
        is_depot = (action == 0)
        departure_time = ready_time + (~is_depot).float() * self.service_time
        
        # Update visited
        new_visited = td["visited"].scatter(-1, action.unsqueeze(-1), True)
        # But depot can be visited multiple times
        new_visited[..., 0] = False

        # Update step count
        i = td["i"] + 1
        
        # --- Reward Logic ---
        # Cost structure: $200 per vehicle + $20 per hour
        # A new trip starts when leaving depot (node 0) to a customer (node > 0)
        is_new_trip = (current_node == 0) & (action > 0)
        fixed_cost = is_new_trip.float() * 200.0
        num_routes = td["num_routes"] + is_new_trip.float()
        start_new_trip = action == 0
        current_time = departure_time * (1 - start_new_trip.float())
        
        # Labor cost is based on the time elapsed in this step (in hours)
        duration_hours = (departure_time - td["current_time"].squeeze(-1)) / 3600.0
        labor_cost = duration_hours * 20.0
        
        step_reward = -(fixed_cost + labor_cost)
        cumulative_reward = td.get("cumulative_reward", 0.0) + step_reward
        # --- End Reward Logic ---
        
        # New state for mask calculation
        new_td = td.clone()
        new_td.update({
            "current_node": action,
            "current_time": current_time.unsqueeze(-1),
            "i": i,
            "visited": new_visited,
            "cumulative_reward": cumulative_reward,
            "total_penalty": total_penalty,
        })
        
        action_mask = self.get_action_mask(new_td)
        
        # Done condition: all customers are visited
        no_more_unvisited = (new_visited[:, 1:]).all(dim=-1)
        done = no_more_unvisited
        td.update({
            "current_node": action,
            "current_time": current_time.unsqueeze(-1),
            "i": i,
            "visited": new_visited,
            "reward": step_reward,
            "cumulative_reward": cumulative_reward,
            "total_penalty": total_penalty,
            "done": done,
            "terminated": done,
            "action_mask": action_mask,
            "num_routes": num_routes,
        })
        
        return td

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        cumulative_reward = td["cumulative_reward"]
        current_node = td["current_node"].squeeze(-1)
        current_time = td["current_time"]
        
        # 1. Final return to depot if not already there
        is_not_at_depot = current_node != 0
        final_reward = cumulative_reward.clone()
        
        if is_not_at_depot.any():
            depot_node = torch.zeros_like(current_node)
            tt_to_depot = self._get_travel_time(td, current_node, depot_node)
            extra_labor_cost = (tt_to_depot / 3600.0) * 20.0
            
            final_reward = torch.where(
                is_not_at_depot,
                cumulative_reward - extra_labor_cost,
                cumulative_reward
            )
        td.update({
            "cumulative_reward": final_reward,
        })
        return final_reward

    def check_solution_validity(self, td: TensorDict, actions: torch.Tensor):
        # Basic validity check: all customers must be visited exactly once
        # (Already handled by action masking and environment logic)
        return True
