import ray
import gymnasium as gym
import torch
from torch import Size
import numpy as np
from tensordict.tensordict import TensorDict
from typing import Any, Dict, List, Optional, Tuple

def _to_numpy(x: Any):
    """Utility: convert torch.Tensor to numpy, leave others unchanged."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

class BaseCOWorker:
    """Base Worker for Combinatorial Optimization environments in RL4CO."""

    def __init__(
        self,
        env_name: str,
        seed: int,
        env_num: int,
        device: str,
        return_topk_options: int = 0,
        **kwargs
    ):
        if "load_from_path" in kwargs:
            self.load_from_path = kwargs["load_from_path"]
        else:
            self.load_from_path = False

        self.env_name = env_name.lower()
        self.env_num = env_num
        self.device = torch.device(device)
        self.actions: List[Any] = []
        self.return_topk_options = return_topk_options > 0
        self.topk_k = return_topk_options
        
        # Initialize the specific RL4CO environment
        # kwargs usually contains env_kwargs or specific params like num_loc
        self.base_env = self._init_env(seed, **kwargs)
        self._td: Optional[TensorDict] = None
        self.done = False

    def _init_env(self, seed: int, **kwargs):
        """Initialize the specific RL4CO environment. Must be implemented by subclasses."""
        raise NotImplementedError

    def reset(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        """Reset all sub-environments."""
        self.done = False
        batch_size = Size([self.env_num])
        if self.load_from_path:
            _td = load_from_file(self.load_from_path, batch_size=batch_size)
            td = self.base_env.reset(_td)
        else:
            td = self.base_env.reset(batch_size=batch_size)
        
        # Sync instances across the batch if necessary
        td = self._sync_instances(td)
        
        self._td = td
        self.actions = []
        infos = [{}] * self.env_num

        # If topk mode is enabled, we might need an initial step or setup
        self._td = self._post_reset_hook(self._td)

        obs = self.build_obs(self._td)
        return obs, infos

    def _sync_instances(self, td: TensorDict) -> TensorDict:
        """Sync instance data (e.g. locations, graphs) across the batch.
        Default implementation does nothing. Override if needed.
        """
        return td

    def _post_reset_hook(self, td: TensorDict) -> TensorDict:
        """Optional hook to run after reset, e.g. for TopK initialization."""
        return td

    def build_obs(self, td: TensorDict) -> List[str]:
        """Convert TensorDict state into textual observations."""
        raise NotImplementedError

    def action_projection(self, env_idx, action):
        """Project action to valid space (handle TopK mapping and Masking)."""

        # --- 1. Top-K Option Mode ---
        if self.return_topk_options:
            try:
                candidates = self._td["topk_acts"][env_idx]
                try:
                    sel_idx = int(action) if not isinstance(action, torch.Tensor) else int(action.item())
                except Exception:
                    sel_idx = 0 

                if 0 <= sel_idx < len(candidates):
                    real_action = candidates[sel_idx].item()
                    # Double check mask
                    mask = self._td.get("action_mask")[env_idx]
                    if mask[real_action]:
                        return real_action
                    else:
                        # Fallback to Top-1 if mapped action is invalid
                        return candidates[0].item()
                else:
                    # Fallback to Top-1 if index out of bounds
                    return candidates[0].item()
            except Exception as e:
                pass

        # --- 2. Raw Action Mode (Mask Check) ---
        try:
            if not isinstance(self._td, TensorDict):
                return action
            if "action_mask" not in self._td.keys():
                return action

            mask = self._td.get("action_mask")[env_idx]
            # Handle batched mask
            if isinstance(mask, torch.Tensor):
                if mask.dim() == 2 and mask.shape[0] >= 1:
                    mask0 = mask[0]
                else:
                    mask0 = mask
                mask_bool = mask0.bool()
                
                if mask_bool.sum().item() == 0:
                    return action

                try:
                    act_int = int(action) if not isinstance(action, torch.Tensor) else int(action.item())
                except Exception:
                    act_int = None

                if act_int is not None and 0 <= act_int < mask_bool.shape[0] and mask_bool[act_int]:
                    return action

                # Random fallback
                idx = torch.multinomial(mask_bool.float(), 1).item()
                return int(idx)
        except Exception:
            return action
        return action

    def actions_projection(self, actions):
        projected_actions = []
        for i, action in enumerate(actions):
            action = self.action_projection(i, action)
            projected_actions.append(action)
        return projected_actions

    def step(self, action) -> Tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
        if self.done:
            return ["The environment has closed."] * self.env_num, [0] * self.env_num, [True] * self.env_num, [{}] * self.env_num

        projected_action = self.actions_projection(action)
        self.actions.append(projected_action)
        
        action_tensor = torch.as_tensor(
            projected_action, device=self.device, dtype=torch.int64
        )
        
        self._td.set("action", action_tensor)
        out = self.base_env.step(self._td)
        next_td = out["next"] if isinstance(out, dict) and "next" in out else out
        
        if not isinstance(next_td, TensorDict):
            raise TypeError(f"Expected TensorDict from step, got {type(next_td)}")
        self._td = next_td

        rewards = [0] * self.env_num
        dones = next_td["done"]
        
        if dones.all():
            actions_hist = torch.as_tensor(self.actions, device=self.device, dtype=torch.int64) # (steps, batch)
            actions_hist = actions_hist.transpose(1, 0) # (batch, steps)
            rewards = self.base_env.get_reward(self._td, actions_hist)
            if isinstance(rewards, torch.Tensor):
                rewards = rewards.tolist()
            self.done = True
        
        infos = [{}] * self.env_num
        obs = self.build_obs(self._td)
        
        dones_list = dones.tolist() if isinstance(dones, torch.Tensor) else list(dones)
        
        return obs, rewards, dones_list, infos

class BaseCOEnvs(gym.Env):
    """Base Ray-based Wrapper for CO Environments."""
    def __init__(
        self, 
        worker_cls, 
        env_name, 
        seed, 
        env_num, 
        group_n, 
        device, 
        resources_per_worker, 
        return_topk_options=True, 
        env_kwargs=None
    ):
        super().__init__()
        self.env_name = env_name
        self.num_processes = env_num * group_n
        self.env_num = env_num
        self.group_n = group_n
        
        if not ray.is_initialized():
            ray.init()

        if resources_per_worker:
            env_worker = ray.remote(**resources_per_worker)(worker_cls)
        else:
            env_worker = ray.remote(worker_cls)
            
        self.workers = []
        
        # Instantiate workers
        # NOTE: This assumes worker_cls accepts these arguments.
        # Subclasses can override this loop if they need to pass different args.
        # We handle the 'common' case where we might pass extra kwargs.
        
        # We need to handle potential list-based args in kwargs if they exist (like num_loc=[...])
        # But BaseCOEnvs doesn't know about num_loc.
        # So we'll iterate and let the caller/subclass handle arg preparation or pass kwargs as is.
        # Here we assume env_kwargs is a dict common to all, OR we might need a better way to distribute args.
        
        # In the original RouteEnvs, num_loc and loc_distribution could be lists.
        # To keep BaseCOEnvs generic, we will rely on a helper method to get args for worker g.
        
        for g in range(env_num):
            worker_args, worker_kwargs = self._get_worker_args(g, env_name, seed, group_n, device, return_topk_options, env_kwargs)
            worker = env_worker.remote(*worker_args, **worker_kwargs)
            self.workers.append(worker)

    def _get_worker_args(self, worker_idx, env_name, seed, group_n, device, return_topk_options, env_kwargs):
        """Prepare arguments for the worker. Override this for specific parameter distribution."""
        # Default behavior: pass everything as is
        # args: env_name, seed, env_num, device, return_topk_options
        args = (env_name, seed + worker_idx, group_n, device, return_topk_options)
        return args, (env_kwargs if env_kwargs else {})

    def step(self, actions):
        assert len(actions) == self.num_processes, "The num of actions must be equal to the num of processes"
        actions = np.array(actions)
        actions = actions.reshape((self.env_num, self.group_n))
        
        futures = [worker.step.remote(actions[i]) for i, worker in enumerate(self.workers)]
        results = ray.get(futures)

        text_obs_list = []
        rewards_list = []
        dones_list = []
        info_list = []

        for i in range(self.env_num):
            for j in range(self.group_n):
                text_obs_list.append(results[i][0][j])
                rewards_list.append(results[i][1][j])
                dones_list.append(results[i][2][j])
                info_list.append(results[i][3][j])

        return text_obs_list, rewards_list, dones_list, info_list

    def reset(self):
        futures = [worker.reset.remote() for worker in self.workers]
        results = ray.get(futures)
        
        text_obs_list = []
        for i in range(self.env_num):
            for j in range(self.group_n):
                text_obs_list.append(results[i][0][j])
        
        return text_obs_list
