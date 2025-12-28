import ray
import gymnasium as gym
import random
import os
import torch
from torch import Size
import logging
import numpy as np

from rl4co.envs.routing.tsp.env import TSPEnv
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.op.env import OPEnv

import matplotlib.pyplot as plt
from tensordict.tensordict import TensorDict
from typing import Any, Dict, List, Optional, Tuple

# Import Base Classes
from .base_env import BaseCOWorker, BaseCOEnvs, _to_numpy

class RouteWorker(BaseCOWorker):
    """Wrapper for RL4CO routing environments (TSP / CVRP / OP).
    
    Inherits from BaseCOWorker to reuse common logic.
    """

    def __init__(
        self,
        env_name: str = "tsp",
        seed: int = 0,
        env_num: int = 1,
        device: str = "cpu",
        num_loc: int = 10,
        loc_distribution: str = "uniform",
        return_topk_options: int = 0,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        # Store routing-specific params
        self.num_loc = num_loc
        self.loc_distribution = loc_distribution
        
        # Call base init
        super().__init__(
            env_name=env_name,
            seed=seed,
            env_num=env_num,
            device=device,
            return_topk_options=return_topk_options
        )

    def _init_env(self, seed: int, **kwargs):
        generator_params = {
            "num_loc": self.num_loc,
            "loc_distribution": self.loc_distribution,
        }

        if self.env_name == "tsp":
            return TSPEnv(
                generator=None,
                generator_params=generator_params,
                seed=seed,
                device=self.device
            )
        elif self.env_name == "cvrp":
            return CVRPEnv(
                generator=None,
                generator_params=generator_params,
                seed=seed,
                device=self.device
            )
        elif self.env_name == "op":
            return OPEnv(
                generator=None,
                generator_params=generator_params,
                seed=seed,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported RL4CO routing env: {self.env_name}")

    def _sync_instances(self, td: TensorDict) -> TensorDict:
        """Force synchronization of locations across the batch."""
        # 强制同步: 让所有环境实例共享相同的 locs (假设这是期望的行为)
        for i in range(1, self.env_num):
            td['locs'][i] = td['locs'][0]
        return td

    def _post_reset_hook(self, td: TensorDict) -> TensorDict:
        """Handle TopK pre-calculation step."""
        if self.return_topk_options:
            actions = [0] * self.env_num
            self._td.set("action", torch.tensor(actions, device=self.device))
            # Step once to get costs/next state
            self._td = self.base_env.step(self._td)['next']
            self.actions.append(actions)
        return self._td

    def build_obs(self, td: TensorDict) -> List[str]:
        """Convert TensorDict numeric state into textual observations."""
        batch_size = td.batch_size[0] if td.batch_size else 1
        obs_list: List[str] = []

        # --- Pre-calculate Top-K if needed ---
        topk_data = None
        if self.return_topk_options and len(self.actions) != 0:
            topk_acts_tensor, topk_costs_tensor = self._get_greedy_topk_helper(td)
            
            topk_acts_list = topk_acts_tensor.tolist()
            topk_costs_list = topk_costs_tensor.tolist()
            
            td["topk_acts"] = topk_acts_tensor
            td["topk_costs"] = topk_costs_tensor

        for i in range(batch_size):
            # Extract locs
            locs = td["locs"][i]
            
            # Extract first_node and current_node
            first_node = None
            current_node = None
            if self.actions != []:
                fn = _to_numpy(td["first_node"][i])
                first_node = int(fn) if hasattr(fn, "__int__") else int(fn[0])
                cn = _to_numpy(td["current_node"][i])
                current_node = int(cn) if hasattr(cn, "__int__") else int(cn[0])
            
            if "locs_mask" in td.keys():
                mask = td["locs_mask"][i]
                if mask.numel() > 0:
                    valid_n = int(mask.sum().item())
                    locs = locs[:valid_n]

            # Scale coordinates
            locs_np = _to_numpy(locs)
            try:
                locs_scaled = (locs_np * 1000).astype(int)
            except Exception:
                locs_scaled = np.array(locs_np, dtype=int)

            # --- Build Metadata ---
            meta_parts: List[str] = []
            if first_node is not None:
                meta_parts.append(f"Start node: {first_node};")
            else:
                meta_parts.append("Choose an arbitrary node as the starting node.")
            if current_node is not None:
                meta_parts.append(f"Current node: {current_node};")
            if hasattr(self, "actions") and len(self.actions) > 0:
                action_str = ",".join(str(action[i]) for action in self.actions) 
                meta_parts.append(f"Trajectory: {action_str};")
            meta_prefix = " ".join(meta_parts) + " " if meta_parts else ""

            # --- Build Problem Specific Context ---
            lines = []
            if self.env_name == "tsp":
                for node_idx, (x, y) in enumerate(locs_scaled.tolist()):
                    lines.append(f"Node {node_idx}, coordinates: [{x}, {y}];")
                base_info = " ".join(lines) + "\n"
            
            elif self.env_name == "cvrp":
                demands = td.get("demand", None)
                d_np = _to_numpy(demands[i]) if demands is not None else None
                
                cap_tensor = td.get("capacity", td.get("vehicle_capacity", None))
                capacity = float(_to_numpy(cap_tensor)[0]) if cap_tensor is not None else None

                for node_idx, (x, y) in enumerate(locs_scaled.tolist()):
                    demand_val = int(d_np[node_idx]) if (d_np is not None and node_idx < len(d_np)) else 0
                    lines.append(f"Node {node_idx}, coordinates: [{x}, {y}], demand: {demand_val};")
                cap_str = f" Vehicle capacity: {int(capacity)}." if capacity is not None else ""
                base_info = " ".join(lines) + cap_str + "\n"

            elif self.env_name == "op":
                prize = td.get("prize", None)
                p_np = _to_numpy(prize[i]) if prize is not None else None
                
                max_len_tensor = td.get("max_length", td.get("max_route_length", None))
                max_route_length = None
                if max_len_tensor is not None:
                    try:
                        max_route_length = float(_to_numpy(max_len_tensor[i]).item())
                    except:
                        pass

                for node_idx, (x, y) in enumerate(locs_scaled.tolist()):
                    prize_val = int(p_np[node_idx]) if (p_np is not None and node_idx < len(p_np)) else 0
                    lines.append(f"Node {node_idx}, coordinates: [{x}, {y}], prize: {prize_val};")
                max_len_str = f" Max route length: {max_route_length}." if max_route_length is not None else ""
                base_info = " ".join(lines) + max_len_str + "\n"
            else:
                base_info = ""

            obs_str = base_info + meta_prefix

            # --- Append Top-K Options ---
            if self.return_topk_options and len(self.actions) != 0:
                options_str = "\nTop candidates based on distance:\n"
                opts_labels = ["A", "B", "C", "D", "E", "F", "G", "H"]
                
                b_acts = topk_acts_list[i]
                b_costs = topk_costs_list[i]
                
                valid_opts = []
                for idx, (act, cost) in enumerate(zip(b_acts, b_costs)):
                    if cost == float('inf'):
                        continue
                    
                    label = opts_labels[idx] if idx < len(opts_labels) else str(idx+1)
                    valid_opts.append(f"{label}. Node {act} (Distance: {cost:.3f})")
                
                if not valid_opts:
                    options_str += "No valid moves available."
                else:
                    options_str += "; ".join(valid_opts)
                
                obs_str += options_str

            obs_list.append(obs_str)

        return obs_list

    def _get_greedy_topk_helper(self, td):
        """Internal helper to calculate topk."""
        batch_size = td.batch_size[0] if td.batch_size else 1
        locs = td["locs"]
        mask = td["action_mask"]
        device = locs.device
        num_loc = locs.shape[1]
        k = min(self.topk_k, num_loc)
        
        curr_node = td.get("current_node", None)
        dists = torch.full((batch_size, num_loc), float('inf'), device=device)

        is_tsp_start = (self.env_name == 'tsp' and (curr_node is None or curr_node[0] is None))
        
        if is_tsp_start:
            dists = torch.zeros((batch_size, num_loc), device=device)
        else:
            curr_indices = curr_node.view(-1, 1, 1).expand(-1, 1, 2)
            current_pos = locs.gather(1, curr_indices.long())
            dists = torch.norm(locs - current_pos, p=2, dim=-1)

        inf_tensor = torch.tensor(float('inf'), device=device)
        dists_masked = torch.where(mask.bool(), dists, inf_tensor)
        
        topk_costs, topk_actions = torch.topk(dists_masked, self.topk_k, dim=1, largest=False)
        return topk_actions, topk_costs


class RouteEnvs(BaseCOEnvs):
    """Ray-based Wrapper for Routing Environments."""
    
    def __init__(self, env_name, seed, env_num, group_n, device, resources_per_worker, is_train=True, return_topk_options=True, env_kwargs=None):
        
        # Prepare params to be stored for _get_worker_args
        self.num_loc_list = None
        self.loc_distribution_list = None
        
        generator_params = env_kwargs.get("generator_params", {}) if isinstance(env_kwargs, dict) else {}
        num_loc = generator_params.get("num_loc", 10)
        loc_distribution = generator_params.get("loc_distribution", "uniform")
        
        # Handle list expansion for per-worker configuration
        if not isinstance(num_loc, list):
            self.num_loc_list = [num_loc] * env_num
        else:
            self.num_loc_list = num_loc
            
        if not isinstance(loc_distribution, list):
            self.loc_distribution_list = [loc_distribution] * env_num
        else:
            self.loc_distribution_list = loc_distribution

        # Call Base init
        super().__init__(
            worker_cls=RouteWorker,
            env_name=env_name,
            seed=seed,
            env_num=env_num,
            group_n=group_n,
            device=device,
            resources_per_worker=resources_per_worker,
            return_topk_options=return_topk_options,
            env_kwargs=env_kwargs
        )

    def _get_worker_args(self, worker_idx, env_name, seed, group_n, device, return_topk_options, env_kwargs):
        """Prepare specific arguments for RouteWorker, handling num_loc/distribution lists."""
        
        # Args expected by RouteWorker.__init__:
        # env_name, seed, env_num, device, num_loc, loc_distribution, return_topk_options, env_kwargs
        
        current_num_loc = self.num_loc_list[worker_idx]
        current_loc_dist = self.loc_distribution_list[worker_idx]
        
        # worker.remote args match RouteWorker.__init__ signature
        args = (
            env_name, 
            seed + worker_idx, 
            group_n, 
            device, 
            current_num_loc, 
            current_loc_dist, 
            return_topk_options, 
            env_kwargs
        )
        return args, {}
