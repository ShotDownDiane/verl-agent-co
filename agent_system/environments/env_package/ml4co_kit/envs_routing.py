from typing import Any, Dict, List, Optional, Tuple

import torch
import numpy as np
import pathlib
import random
from tensordict.tensordict import TensorDict
from torch.nn.utils.rnn import pad_sequence

# Native ML4CO-Kit routing components
from ml4co_kit import TSPGenerator, TSPWrapper
from ml4co_kit import CVRPGenerator, CVRPWrapper
from ml4co_kit import OPGenerator, OPWrapper
from ml4co_kit import TSPTask, CVRPTask, OPTask



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
        mode: str = "train",
    ):
        self.mode = mode.lower()
        self.env_name = env_name.lower()
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n
        self.device = torch.device(device)

        gen_params = generator_params or {}
        np.random.seed(seed)
        torch.manual_seed(seed)

        if self.env_name == "tsp" and self.mode == "train":
            self.generator = TSPGenerator(
                nodes_num=gen_params.get("num_loc", 20),
                distribution_type=gen_params.get("distribution_type", TSPGenerator.__init__.__defaults__[0]),
            )
        elif self.env_name == "tsp" and self.mode == "test":
            wrapper = TSPWrapper()
            wrapper.from_tsplib_folder(
                tsp_folder_path=pathlib.Path("/root/autodl-tmp/ML4CO-Kit/test_dataset/tsp/tsplib/problem"),
                tour_folder_path=pathlib.Path("/root/autodl-tmp/ML4CO-Kit/test_dataset/tsp/tsplib/solution"),
                ref=True,
                overwrite=True,
                normalize=True
            )
            import pdb; pdb.set_trace
            self.wrapper = wrapper

        elif self.env_name == "cvrp" and self.mode == "train":
            self.generator = CVRPGenerator(
                nodes_num=gen_params.get("num_loc", 20),
                min_demand=gen_params.get("min_demand", 1),
                max_demand=gen_params.get("max_demand", 9),
                min_capacity=gen_params.get("min_capacity", 40),
                max_capacity=gen_params.get("max_capacity", 40),
                distribution_type=gen_params.get("distribution_type", CVRPGenerator.__init__.__defaults__[0]),
            )
        elif self.env_name == "cvrp" and self.mode == "test":
            wrapper = CVRPWrapper()
            wrapper.from_cvrp_folder(
                cvrp_folder_path=pathlib.Path("/root/autodl-tmp/ML4CO-Kit/test_dataset/cvrp/cvrp/problem"),
                solution_folder_path=pathlib.Path("/root/autodl-tmp/ML4CO-Kit/test_dataset/cvrp/cvrp/solution"),
                ref=True,
                overwrite=True,
                normalize=True
            )
        elif self.env_name == "op" and self.mode == "train":
            self.generator = OPGenerator(
                nodes_num=gen_params.get("num_loc", 20),
                max_length=gen_params.get("max_length", 3.0),
                distribution_type=gen_params.get("distribution_type", OPGenerator.__init__.__defaults__[0]),
            )
        elif self.env_name == "op" and self.mode == "test":
            wrapper = OPWrapper()
            wrapper.from_op_folder(
                op_folder_path=pathlib.Path("/root/autodl-tmp/ML4CO-Kit/test_dataset/op/op/problem"),
                solution_folder_path=pathlib.Path("/root/autodl-tmp/ML4CO-Kit/test_dataset/op/op/solution"),
                ref=True,
                overwrite=True,
                normalize=True
            )
            self.wrapper = wrapper
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
        if self.mode == "train":
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
        elif self.mode == "test":
            task_list = self.wrapper.task_list[:self.env_num]
            for task in task_list:
                tasks.append(task)
                if isinstance(task, TSPTask):
                    locs_list.append(torch.tensor(task.points, dtype=torch.float32))
                    print("node size: ",task.points.shape)
                elif isinstance(task, CVRPTask):
                    coords = np.concatenate([task.depots[None, :], task.points], axis=0)
                    locs_list.append(torch.tensor(coords, dtype=torch.float32))
                    demands_list.append(torch.tensor(task.demands, dtype=torch.float32))
                    capacities.append(float(task.capacity))
                elif isinstance(task, OPTask):
                    coords = np.concatenate([task.depots[None, :], task.points], axis=0)
                    locs_list.append(torch.tensor(coords, dtype=torch.float32))
                    prizes_list.append(torch.tensor(task.prizes, dtype=torch.float32))

        # Convert variable-length locs_list (list of tensors (n_i, 2)) into a
        # padded tensor and mask so TensorDict can accept a batched tensor field.
        if locs_list:
            lengths = [t.shape[0] for t in locs_list]
            locs_padded = pad_sequence(locs_list, batch_first=True)  # (batch, max_n, 2)
            locs_mask = torch.zeros((self.env_num, locs_padded.shape[1]), dtype=torch.bool)
            for i, l in enumerate(lengths):
                locs_mask[i, : l] = True
            data: Dict[str, torch.Tensor] = {"locs": locs_padded, "locs_mask": locs_mask}
        else:
            data: Dict[str, torch.Tensor] = {"locs": torch.zeros((self.env_num, 0, 2), dtype=torch.float32), "locs_mask": torch.zeros((self.env_num, 0), dtype=torch.bool)}
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
            if self.env_name == "tsp" and act[0] != act[-1]:
                act = act + [act[0]]
            try:
                if isinstance(task, TSPTask):
                    route = np.array(act, dtype=int)
                    if route.size == 0:
                        raise ValueError("Empty route")
                    cost = float(task.evaluate(route))
                    if self.mode == "test":
                        task.sol = route
                        sol_cost, ref_cost, gap = task.evaluate_w_gap()
                        print("sol cost: ", sol_cost)
                        print("ref_cost: ", ref_cost)
                        cost = gap
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


def build_ml4cokit_routing_envs(
    env_name: str = "tsp",
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    device: str = "cpu",
    mode: str = "train",
    generator_params: Optional[Dict[str, Any]] = None,
    rl4co_kwargs: Optional[Dict[str, Any]] = None,
):
    return ML4COKitRoutingEnvs(
        env_name=env_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        device=device,
        mode=mode,
        generator_params=generator_params,
        rl4co_kwargs=rl4co_kwargs,
    )

if __name__ == "__main__":
    # Test the build_ml4cokit_routing_envs function
    env = build_ml4cokit_routing_envs(env_name="tsp", mode="test")
    env.reset()