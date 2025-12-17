import torch
from torch import Size
from typing import Any, Dict, List, Optional, Tuple

from tensordict.tensordict import TensorDict

from rl4co.envs.routing.tsp.env import TSPEnv
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.op.env import OPEnv


def _to_numpy(x: Any):
    """Utility: convert torch.Tensor to numpy, leave others unchanged."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class RL4CORoutingEnvs:
    """Wrapper for RL4CO routing environments (TSP / CVRP / OP).

    Designed to fit verl-agent's env_package style:
    - Vectorized over env_num * group_n parallel envs.
    - reset/step API returning (TensorDict, rewards, dones, infos).
    """

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

        generator_params = generator_params or {}
        rl4co_kwargs = rl4co_kwargs or {}

        if self.env_name == "tsp":
            self.base_env = TSPEnv(
                generator=None,
                generator_params=generator_params,
                seed=seed,
                device=self.device,
                **rl4co_kwargs,
            )
        elif self.env_name == "cvrp":
            self.base_env = CVRPEnv(
                generator=None,
                generator_params=generator_params,
                seed=seed,
                device=self.device,
                **rl4co_kwargs,
            )
        elif self.env_name == "op":
            self.base_env = OPEnv(
                generator=None,
                generator_params=generator_params,
                seed=seed,
                device=self.device,
                **rl4co_kwargs,
            )
        else:
            raise ValueError(f"Unsupported RL4CO routing env: {env_name}")

        self._td: Optional[TensorDict] = None

    def reset(self) -> Tuple[TensorDict, List[Dict[str, Any]]]:
        """Reset all sub-environments."""
        batch_size = Size([self.num_processes])
        td = self.base_env.reset(batch_size=batch_size)
        if isinstance(td, dict) and "next" in td:
            td = td["next"]
        if not isinstance(td, TensorDict):
            raise TypeError(f"Expected TensorDict from reset, got {type(td)}")
        self._td = td
        infos: List[Dict[str, Any]] = [{} for _ in range(self.num_processes)]
        return td, infos

    def step(
        self, actions: List[int]
    ) -> Tuple[TensorDict, Any, Any, List[Dict[str, Any]]]:
        """Step all environments with integer actions."""
        if self._td is None:
            raise RuntimeError("Environment not reset. Call reset() first.")
        if len(actions) != self.num_processes:
            raise ValueError(
                f"Expected {self.num_processes} actions, got {len(actions)}"
            )
        action_tensor = torch.as_tensor(
            actions, device=self.device, dtype=torch.int64
        )
        self._td.set("action", action_tensor)

        out = self.base_env.step(self._td)
        next_td = out["next"] if isinstance(out, dict) and "next" in out else out
        if not isinstance(next_td, TensorDict):
            raise TypeError(f"Expected TensorDict from step, got {type(next_td)}")
        self._td = next_td

        rewards = _to_numpy(next_td.get("reward", None))
        dones = _to_numpy(next_td.get("done", None))
        infos: List[Dict[str, Any]] = [{} for _ in range(self.num_processes)]
        return next_td, rewards, dones, infos

    def close(self) -> None:
        return


def build_rl4co_routing_envs(
    env_name: str = "tsp",
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    device: str = "cpu",
    generator_params: Optional[Dict[str, Any]] = None,
    rl4co_kwargs: Optional[Dict[str, Any]] = None,
):
    return RL4CORoutingEnvs(
        env_name=env_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        device=device,
        generator_params=generator_params,
        rl4co_kwargs=rl4co_kwargs,
    )

