import ray
import gymnasium as gym
import torch
import numpy as np
import os
import cv2
import base64
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple
from tensordict.tensordict import TensorDict

from rl4co.envs.routing.tsp.env import TSPEnv
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.op.env import OPEnv
from rl4co.envs.routing.tdtsp.env import TDTSPMatrixEnv
from rl4co.envs.routing.tdtsp.env_tw import TDTSPTWEnv, TDTSPTWGenerator
from rl4co.envs.routing.tdvrp.env import TDVRPEnv
from rl4co.envs.routing.lrp.env import LRPEnv

from .base_env import BaseCOWorker, BaseCOEnvs
from .route_obs import (
    build_obs_tsp,
    build_obs_cvrp,
    build_obs_op,
    build_obs_tdtsp,
    build_obs_tdtsp_tw,
    build_obs_tdvrp,
    build_obs_lrp
)

class RouteWorker(BaseCOWorker):
    """Wrapper for RL4CO routing environments (TSP / CVRP / OP / LRP).
    
    Inherits from BaseCOWorker to reuse common logic.
    Refactored to separate observation building logic.
    """
    
    ENV_CONFIG = {
        'tsp': {'cls': TSPEnv, 'builder': build_obs_tsp},
        'cvrp': {'cls': CVRPEnv, 'builder': build_obs_cvrp},
        'op': {'cls': OPEnv, 'builder': build_obs_op},
        'tdtsp': {'cls': TDTSPMatrixEnv, 'builder': build_obs_tdtsp},
        'tdtsp_tw': {'cls': TDTSPTWEnv, 'builder': build_obs_tdtsp_tw},
        'tdvrp': {'cls': TDVRPEnv, 'builder': build_obs_tdvrp},
        'lrp': {'cls': LRPEnv, 'builder': build_obs_lrp},
    }

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
        image_obs: str = "rgb", # or "base64", "path"
    ):
        # Store routing-specific params
        self.num_loc = num_loc
        self.loc_distribution = loc_distribution
        self.env_kwargs = env_kwargs
        self.image_obs = image_obs
        self.batch_size = env_kwargs.get("batch_size", 1)
        synchronous = True
        if "synchronous" in self.env_kwargs:
            synchronous = self.env_kwargs["synchronous"]
        
        # Call base init
        super().__init__(
            env_name=env_name,
            seed=seed,
            env_num=env_num,
            device=device,
            return_topk_options=return_topk_options,
            synchronous=synchronous,
            batch_size=self.batch_size
        )

    def _init_env(self, seed: int, **kwargs):
        env_key = self.env_name.lower()
        if env_key not in self.ENV_CONFIG:
            raise ValueError(f"Unsupported RL4CO routing env: {self.env_name}")
            
        env_cls = self.ENV_CONFIG[env_key]['cls']
        
        # Check if a generator instance is provided in env_kwargs
        generator = self.env_kwargs.get("generator", None) if self.env_kwargs else None
        
        # 1. Handle specialized environments that use data paths instead of uniform distribution
        if env_key in ['tdtsp', 'tdtsp_tw', 'tdvrp'] and generator is None:
            # These envs usually require specific paths for matrix and instance
            data_path = self.env_kwargs.get("data_path")
            matrix_path = self.env_kwargs.get("matrix_path")
            base_data_path = self.env_kwargs.get("base_data_path")
            random_sample = self.env_kwargs.get("random_sample", True)
            
            if env_key == 'tdtsp':
                generator = TDTSPTWGenerator(
                    data_path=data_path,
                    base_data_path=base_data_path,
                    matrix_path=matrix_path,
                )
                return env_cls(
                    generator=generator,
                    device=self.device
                )
            elif env_key == 'tdtsp_tw':
                # TDTSPTW uses a generator internally
                params = {
                    "data_file_path": data_path,
                    "base_data_path": base_data_path,
                    "matrix_path": matrix_path,
                    "penalty_value": 3.0, # Default to 0 for easier cost reporting
                    "device": self.device
                }
                if self.env_kwargs:
                    # Filter out keys that might conflict or are already handled
                    extra_params = {k: v for k, v in self.env_kwargs.items() if k not in ["data_path", "base_data_path", "matrix_path"]}
                    params.update(extra_params)
                return env_cls(**params)
            elif env_key == 'tdvrp':
                # TDVRP also needs specific generator setup or path-based init
                from rl4co.envs.routing.tdvrp.generator import TDVRPGenerator
                gen = TDVRPGenerator(
                    data_path=data_path,
                    base_data_path=base_data_path,
                    matrix_path=matrix_path,
                    service_time=180.0, # Hardcoded as per requirement
                    random_sample=random_sample,
                )
                return env_cls(
                    generator=gen,
                    device=self.device
                )
        
        # 2. Standard uniform distribution environments (TSP, CVRP, OP)
        generator_params = {
            "num_loc": self.num_loc,
            "loc_distribution": self.loc_distribution,
        }
        
        # Merge with any extra generator params from env_kwargs
        if self.env_kwargs and "generator_params" in self.env_kwargs:
            generator_params.update(self.env_kwargs["generator_params"])
        
        # Also check inside generator_params for "_generator_obj" (hack to pass via config)
        if generator is None and "generator" in generator_params:
            generator = generator_params.pop("generator")

        return env_cls(
            generator=generator,
            generator_params=generator_params,
            seed=seed,
            device=self.device
        )

    def step(self, actions: List[int]) -> Tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
        obs_list, rewards, dones, infos = super().step(actions)
              
        # Add cumulative reward to info if available in TensorDict
        if "cumulative_reward" in self._td.keys():
            cum_rewards = self._td["cumulative_reward"].tolist()
            total_violations = self._td["total_penalty"].tolist()
            
            for i in range(self.env_num):
                infos[i]["cumulative_reward"] = cum_rewards[i]
                infos[i]["total_penalty"] = total_violations[i]
        
        return obs_list, rewards, dones, infos

    def _sync_instances(self, td: TensorDict) -> TensorDict:
        """Force synchronization of locations across the batch."""
        temp_td = None
        for i in range(0, self.env_num):
            if i % self.group_size == 0:
                temp_td = td[i].clone()
            else:
                td[i] = temp_td.clone()
        return td

    def _post_reset_hook(self, td: TensorDict) -> TensorDict:
        """Handle TopK pre-calculation step."""
        if self.return_topk_options:
            # Initialize with dummy action (0) to step and get initial costs
            actions = [0] * self.env_num
            # Ensure action tensor is on the correct device
            self._td.set("action", torch.tensor(actions, device=self.device))
            
            # Step once to get costs/next state
            # Note: The base_env.step might return a dict or TensorDict
            step_res = self.base_env.step(self._td)
            self._td = step_res['next']
            
            self.actions.append(actions)
        return self._td

    def build_obs(self, td: TensorDict) -> List[str]:
        """Delegate observation building to specific functions defined in route_obs.py"""
        env_key = self.env_name.lower()
        builder = self.ENV_CONFIG[env_key]['builder']
        if builder:
            # Pass necessary context to builder
            return builder(
                td=td, 
                env_num=self.env_num, 
                trajectory=self.actions,
                top_k=self.topk_k,
                image_obs=self.image_obs,
            )
        else:
            return [f"No observation builder defined for {self.env_name}"] * self.env_num


class RouteEnvs(BaseCOEnvs):
    """Ray-based Wrapper for Routing Environments."""
    
    def __init__(self, env_name, seed, env_num, group_n, device, resources_per_worker, is_train=True, return_topk_options=True, env_kwargs=None):
        
        # Prepare params to be stored for _get_worker_args
        self.num_loc_list = None
        self.loc_distribution_list = None
        
        generator_params = env_kwargs.get("generator_params", {}) if isinstance(env_kwargs, dict) else {}
        num_loc = generator_params.get("num_loc", 20)
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
        
        current_num_loc = self.num_loc_list[worker_idx]
        current_loc_dist = self.loc_distribution_list[worker_idx]
        
        # Create a worker-specific env_kwargs to ensure scalars are passed
        worker_env_kwargs = env_kwargs.copy()
        
        # Handle list-based generator in env_kwargs (e.g. for batch processing with different data per worker)
        if "generator" in env_kwargs and isinstance(env_kwargs["generator"], list):
            worker_env_kwargs["generator"] = env_kwargs["generator"][worker_idx]
            
        if "generator_params" in worker_env_kwargs:
            worker_env_kwargs["generator_params"] = worker_env_kwargs["generator_params"].copy()
            worker_env_kwargs["generator_params"]["num_loc"] = current_num_loc
            worker_env_kwargs["generator_params"]["loc_distribution"] = current_loc_dist

        # args matching RouteWorker.__init__ signature
        image_obs = env_kwargs.get("image_obs", "rgb")
        args = (
            env_name, 
            seed + worker_idx, 
            group_n, 
            device, 
            current_num_loc, 
            current_loc_dist, 
            return_topk_options, 
            worker_env_kwargs,
            image_obs
        )
        return args, {}

def build_route_envs(
    env_name: str = "tsp",
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 3,
    device: str = "cpu",
    generator_params: Optional[Dict[str, Any]] = None,
    rl4co_kwargs: Optional[Dict[str, Any]] = None,
    return_topk_options: int = 0,
    batch_size: int = 1,
    mode: str = "train"
):
    # Package generator / rl4co kwargs into env_kwargs for RouteEnvs.
    env_kwargs: Dict[str, Any] = {}
    env_kwargs["batch_size"] = batch_size
    if generator_params is not None:
        env_kwargs["generator_params"] = generator_params
    if rl4co_kwargs is not None:
        env_kwargs["rl4co_kwargs"] = rl4co_kwargs
    if env_name in ["tdvrp", "tdtsp_tw", "tdtsp"]:
        train_data_path = "/root/autodl-tmp/tdtsp_dataset_random/berlin_50_random_train.npz"
        test_data_path = "/root/autodl-tmp/tdtsp_dataset_random/berlin_50_random_test.npz"
        base_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
        matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
        env_kwargs["data_path"] = train_data_path if mode == "train" else test_data_path
        env_kwargs["base_data_path"] = base_path
        env_kwargs["matrix_path"] = matrix_path
        env_kwargs["random_sample"] = True if mode == "train" else False
        env_kwargs["synchronous"] = True if mode == "train" else False

    # resources_per_worker is optional here; use empty dict so RouteEnvs will
    # create actors with default resources unless the caller extends this builder.
    resources_per_worker: Dict[str, Any] = {}

    return RouteEnvs(
        env_name=env_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        device=device,
        resources_per_worker=resources_per_worker,
        is_train=True,
        return_topk_options=return_topk_options,
        env_kwargs=env_kwargs,
    )
