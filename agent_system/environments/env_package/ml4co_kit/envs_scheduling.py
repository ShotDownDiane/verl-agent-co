from typing import Any, Dict, List, Optional, Tuple

import torch
from tensordict.tensordict import TensorDict

# Scheduling still leverages RL4CO until ML4CO-Kit scheduling is wired
from agent_system.environments.env_package.rl4co_scheduling import (
    RL4COSchedulingEnvs,
)

class ML4COKitSchedulingEnvs:
    """One-shot wrapper for scheduling problems using RL4CO scheduling envs."""

    def __init__(
        self,
        env_name: str = "jssp",
        seed: int = 0,
        env_num: int = 1,
        group_n: int = 1,
        device: str = "cpu",
        generator_params: Optional[Dict[str, Any]] = None,
        rl4co_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.env_name = env_name.lower()
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.device = torch.device(device)

        self.inner_env = RL4COSchedulingEnvs(
            env_name="ffsp" if self.env_name == "pfsp" else self.env_name,
            seed=seed,
            env_num=env_num,
            group_n=group_n,
            device=device,
            generator_params=generator_params,
            rl4co_kwargs=rl4co_kwargs,
        )
        self._td: Optional[TensorDict] = None

    def reset(self) -> Tuple[TensorDict, List[Dict[str, Any]]]:
        td, infos = self.inner_env.reset()
        if not isinstance(td, TensorDict):
            raise TypeError(f"Expected TensorDict from reset, got {type(td)}")
        self._td = td
        return td, infos

    def step(
        self, schedules: List[List[int]]
    ) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        if self._td is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        batch_size = self._td.batch_size[0]
        if len(schedules) != batch_size:
            raise ValueError(f"Expected {batch_size} schedules, got {len(schedules)}")

        base_env = self.inner_env.base_env
        rewards: List[torch.Tensor] = []
        dones: List[torch.Tensor] = []
        final_tds: List[TensorDict] = []
        infos: List[Dict[str, Any]] = []

        for b in range(batch_size):
            td_b = self._td[b : b + 1].clone()
            seq = schedules[b] or []
            # Get valid job range from TensorDict
            num_jobs = None
            if "next_op" in td_b.keys():
                num_jobs = td_b["next_op"].shape[-1]
            elif "job_location" in td_b.keys():
                num_jobs = td_b["job_location"].shape[-1] - 1
            # Filter out invalid job indices
            if num_jobs is not None:
                seq = [j for j in seq if 0 <= j < num_jobs]
            
            # Execute schedule respecting action_mask constraints
            max_steps = len(seq) * 10  # Safety limit to avoid infinite loops
            step_count = 0
            seq_idx = 0
            executed_jobs = []
            
            while step_count < max_steps:
                done_b = td_b.get(
                    "done", torch.zeros(1, dtype=torch.bool, device=td_b.device)
                )
                if done_b.all():
                    break
                
                # Get action mask
                action_mask = td_b.get("action_mask", None)
                if action_mask is not None:
                    # Handle different tensor shapes
                    if action_mask.dim() > 1:
                        action_mask_np = action_mask[0].cpu().numpy()
                    else:
                        action_mask_np = action_mask.cpu().numpy()
                    # Convert to boolean and get valid indices
                    if action_mask_np.dtype == bool:
                        valid_jobs = [j for j in range(len(action_mask_np)) if action_mask_np[j]]
                    else:
                        valid_jobs = [j for j in range(len(action_mask_np)) if action_mask_np[j] > 0]
                else:
                    valid_jobs = list(range(num_jobs)) if num_jobs is not None else []
                
                # Try to execute next job from schedule if it's valid
                job_idx = None
                if seq_idx < len(seq):
                    candidate = seq[seq_idx]
                    if candidate in valid_jobs:
                        job_idx = candidate
                        seq_idx += 1
                        executed_jobs.append(job_idx)
                    elif valid_jobs:
                        # If candidate is not valid, try to find next valid job in schedule
                        for i in range(seq_idx, len(seq)):
                            if seq[i] in valid_jobs:
                                job_idx = seq[i]
                                seq_idx = i + 1
                                executed_jobs.append(job_idx)
                                break
                
                # If no job from schedule is valid, try any valid job
                if job_idx is None and valid_jobs:
                    job_idx = valid_jobs[0]
                    executed_jobs.append(job_idx)
                
                # If still no valid job, break
                if job_idx is None:
                    break
                
                # Execute the action
                td_b.set(
                    "action",
                    torch.tensor([job_idx], device=td_b.device, dtype=torch.int64),
                )
                out = base_env.step(td_b)
                td_b = out["next"] if isinstance(out, dict) else out
                step_count += 1
            
            r = td_b.get("reward", torch.zeros(1, 1, device=td_b.device))
            # If reward is inf or nan, set to a large negative value
            if torch.isinf(r) or torch.isnan(r):
                r = torch.tensor([[-1000.0]], device=r.device)
            rewards.append(r.reshape(-1))
            dones.append(torch.ones_like(r.reshape(-1), dtype=torch.bool))
            final_tds.append(td_b)
            infos.append({"schedule": executed_jobs, "original_schedule": seq})

        next_td = TensorDict.cat(final_tds, dim=0)
        reward_tensor = torch.cat(rewards, dim=0)
        done_tensor = torch.cat(dones, dim=0)
        self._td = next_td
        return next_td, reward_tensor, done_tensor, infos

    def close(self) -> None:
        return


def build_ml4cokit_scheduling_envs(
    env_name: str = "jssp",
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    device: str = "cpu",
    generator_params: Optional[Dict[str, Any]] = None,
    rl4co_kwargs: Optional[Dict[str, Any]] = None,
):
    return ML4COKitSchedulingEnvs(
        env_name=env_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        device=device,
        generator_params=generator_params,
        rl4co_kwargs=rl4co_kwargs,
    )
