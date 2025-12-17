from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
from tensordict.tensordict import TensorDict

# Native ML4CO-Kit routing components
from ml4co_kit.generator.routing.tsp import TSPGenerator
from ml4co_kit.generator.routing.cvrp import CVRPGenerator
from ml4co_kit.generator.routing.op import OPGenerator
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.task.routing.cvrp import CVRPTask
from ml4co_kit.task.routing.op import OPTask

# Scheduling still leverages RL4CO until ML4CO-Kit scheduling is wired
from agent_system.environments.env_package.rl4co_scheduling import (
    RL4COSchedulingEnvs,
)


def _to_numpy(x: Any):
    """Utility: convert torch.Tensor to numpy, leave others unchanged."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


def _tsp_cost(locs: np.ndarray, route: List[int]) -> float:
    if len(route) == 0:
        return 0.0
    if route[0] != 0:
        route = [0] + route
    if route[-1] != 0:
        route = route + [0]
    total = 0.0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        diff = locs[a] - locs[b]
        total += float(np.sqrt((diff * diff).sum()))
    return total


class ML4COKitRoutingEnvs:
    """One-shot wrapper for routing problems using ML4CO-Kit generators/tasks."""

    def __init__(
        self,
        env_name: str = "tsp",
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

        gen_params = generator_params or {}
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.env_name == "tsp":
            self.generator = TSPGenerator(
                nodes_num=gen_params.get("num_loc", 20),
                distribution_type=gen_params.get("distribution_type", TSPGenerator.__init__.__defaults__[0]),
            )
        elif self.env_name == "cvrp":
            self.generator = CVRPGenerator(
                nodes_num=gen_params.get("num_loc", 20),
                min_demand=gen_params.get("min_demand", 1),
                max_demand=gen_params.get("max_demand", 9),
                min_capacity=gen_params.get("min_capacity", 40),
                max_capacity=gen_params.get("max_capacity", 40),
                distribution_type=gen_params.get("distribution_type", CVRPGenerator.__init__.__defaults__[0]),
            )
        elif self.env_name == "op":
            self.generator = OPGenerator(
                nodes_num=gen_params.get("num_loc", 20),
                max_length=gen_params.get("max_length", 3.0),
                distribution_type=gen_params.get("distribution_type", OPGenerator.__init__.__defaults__[0]),
            )
        else:
            raise ValueError(f"Unsupported ml4co-kit routing env: {self.env_name}")

        self._tasks: List[Any] = []
        self._td: Optional[TensorDict] = None

    def reset(self) -> Tuple[TensorDict, List[Dict[str, Any]]]:
        tasks: List[Any] = []
        locs_list: List[torch.Tensor] = []
        demands_list: List[torch.Tensor] = []
        prizes_list: List[torch.Tensor] = []
        capacities: List[float] = []
        max_lengths: List[float] = []

        for _ in range(self.env_num):
            task = self.generator.generate()
            tasks.append(task)
            if isinstance(task, TSPTask):
                locs_list.append(torch.tensor(task.points, dtype=torch.float32))
            elif isinstance(task, CVRPTask):
                coords = np.concatenate([task.depots[None, :], task.points], axis=0)
                locs_list.append(torch.tensor(coords, dtype=torch.float32))
                demands_list.append(torch.tensor(task.demands, dtype=torch.float32))
                capacities.append(float(task.capacity))
            elif isinstance(task, OPTask):
                coords = np.concatenate([task.depots[None, :], task.points], axis=0)
                locs_list.append(torch.tensor(coords, dtype=torch.float32))
                prizes_list.append(torch.tensor(task.prizes, dtype=torch.float32))
                max_lengths.append(float(task.max_length))

        locs = torch.stack(locs_list, dim=0)
        data: Dict[str, torch.Tensor] = {"locs": locs}
        if demands_list:
            data["demand"] = torch.stack(demands_list, dim=0)
        if capacities:
            data["capacity"] = torch.tensor(capacities, dtype=torch.float32)
        if prizes_list:
            data["prize"] = torch.stack(prizes_list, dim=0)
        if max_lengths:
            data["max_length"] = torch.tensor(max_lengths, dtype=torch.float32)

        td = TensorDict(data, batch_size=[self.env_num])
        self._tasks = tasks
        self._td = td
        infos = [{} for _ in range(self.env_num)]
        return td, infos

    def step(
        self, actions: List[Any]
    ) -> Tuple[TensorDict, torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
        """One-shot evaluation: takes full routes and returns final reward."""
        if self._td is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        batch_size = self._td.batch_size[0]
        if len(actions) != batch_size:
            raise ValueError(f"Expected {batch_size} actions, got {len(actions)}")

        rewards: List[float] = []
        infos: List[Dict[str, Any]] = []

        for i in range(batch_size):
            task = self._tasks[i]
            act = actions[i] or []
            try:
                if isinstance(task, TSPTask):
                    route = np.array(act, dtype=int)
                    if route.size == 0:
                        raise ValueError("Empty route")
                    cost = float(task.evaluate(route))
                    rewards.append(-cost)
                    infos.append({"route": act})
                elif isinstance(task, CVRPTask):
                    routes = act
                    if not routes:
                        raise ValueError("Empty routes")
                    flat: List[int] = []
                    for r in routes:
                        if not r:
                            continue
                        r_seq = list(r)
                        if r_seq[0] != 0:
                            r_seq = [0] + r_seq
                        if r_seq[-1] != 0:
                            r_seq = r_seq + [0]
                        flat.extend(r_seq[1:] + [0])
                    sol = np.array([0] + flat, dtype=int)
                    cost = float(task.evaluate(sol))
                    rewards.append(-cost)
                    infos.append({"routes": routes})
                elif isinstance(task, OPTask):
                    route = np.array(act, dtype=int)
                    if route.size == 0:
                        raise ValueError("Empty route")
                    if route[0] != 0:
                        route = np.concatenate([[0], route])
                    if route[-1] != 0:
                        route = np.concatenate([route, [0]])
                    prize = float(task.evaluate(route))
                    rewards.append(prize)
                    infos.append({"route": act})
                else:
                    raise ValueError(f"Unknown task type {type(task)}")
            except Exception as e:
                rewards.append(-1e3)
                infos.append({"error": str(e), "route": act})

        reward_tensor = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        done_tensor = torch.ones_like(reward_tensor, dtype=torch.bool)
        return self._td, reward_tensor, done_tensor, infos

    def close(self) -> None:
        return


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


def build_ml4cokit_routing_envs(
    env_name: str = "tsp",
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    device: str = "cpu",
    generator_params: Optional[Dict[str, Any]] = None,
    rl4co_kwargs: Optional[Dict[str, Any]] = None,
):
    return ML4COKitRoutingEnvs(
        env_name=env_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        device=device,
        generator_params=generator_params,
        rl4co_kwargs=rl4co_kwargs,
    )


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
