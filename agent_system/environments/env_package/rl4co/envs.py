import ray
import gymnasium as gym

import random
import os
from sympy.simplify.fu import greedy
import torch
from torch import Size
import logging
import numpy as np

from rl4co.envs.routing.tsp.env import TSPEnv
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.op.env import OPEnv

import matplotlib.pyplot as plt
from matplotlib.style import available
from tensordict.tensordict import TensorDict
from typing import Any, Dict, List, Optional, Tuple
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas



def _to_numpy(x: Any):
    """Utility: convert torch.Tensor to numpy, leave others unchanged."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x

def greedy_policy(td: TensorDict, env_idx: int, env_name: str = "tsp") -> list:
    """
    根据当前的 TensorDict 状态，计算贪婪动作（Greedy Action）。
    策略：选择距离当前节点最近的有效节点。
    """
    batch_size = td.batch_size[0] if td.batch_size else 1
    locs = td["locs"][env_idx]               # (batch, num_loc, 2)
    mask = td["action_mask"][env_idx]        # (batch, num_loc)
    
    # 获取当前节点索引 (current_node)
    # 在 RL4CO 中，初始状态对于 TSP 可能是 None/Empty，对于 CVRP 是 0 (Depot)
    curr_node = td.get("current_node", None)[env_idx]

    actions = []

    for i in range(batch_size):
        env_locs = locs[i]      # 当前环境的所有节点坐标
        env_mask = mask[i]      # 当前环境的动作掩码
        
        # --- 1. 确定当前位置 ---
        current_pos = None
        
        # 检查是否是第一步（或者 current_node 不可用）
        # 注意：在 TSP 中，第一步还没有 current_node，我们需要选一个起点
        first_step_tsp = (env_name == 'tsp' and (curr_node is None))
        
        if first_step_tsp:
            # TSP 第一步：贪婪策略通常固定选 Node 0 作为起点，或者选第一个可行的节点
            # 这样对于轮循是对称的，且保证确定性
            pass 
        else:
            # 获取当前节点索引
            if curr_node is not None:
                idx = curr_node
                # 处理 Tensor 类型的索引
                if isinstance(idx, torch.Tensor):
                    idx = idx.item()
                # 只有当索引有效时才获取坐标（RL4CO 中 depot 也是有效索引）
                if idx is not None:
                    current_pos = env_locs[idx]

        # --- 2. 计算距离并选择动作 ---
        
        valid_indices = torch.nonzero(env_mask).squeeze(-1)
        if valid_indices.numel() == 0:
            # 如果没有有效动作（理论上 step 循环会由 done 控制，不应进这里），返回 0 防止报错
            actions.append(0)
            continue

        if current_pos is None:
            # 如果没有当前位置（TSP 第一步），直接选择第一个有效的节点（通常是 0）
            # 或者选择索引最小的有效节点
            best_action = valid_indices[0].item() if valid_indices.numel() > 1 else valid_indices.item()
        else:
            # 计算当前点到所有点的欧几里得距离
            # dists: (num_loc,)
            dists = torch.norm(env_locs - current_pos, p=2, dim=-1)
            
            # 应用掩码：将无效节点的距离设为无穷大
            # env_mask 为 True 表示有效，False 表示无效
            # 我们需要找最小距离，所以无效动作设为 inf
            inf_tensor = torch.tensor(float('inf'), device=dists.device)
            dists_masked = torch.where(env_mask.bool(), dists, inf_tensor)
            
            # 选择距离最近的节点索引
            best_action = torch.argmin(dists_masked).item()
            
        actions.append(best_action)

    return actions

class RouteWorker:
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
        device: str = "cpu",
        num_loc: int = 10,
        loc_distribution: str = "uniform"
    ):
        self.env_name = env_name.lower()
        self.env_num = env_num
        self.device = torch.device(device)
        self.actions: List[int] = []

        generator_params={
            "num_loc": num_loc,
            "loc_distribution": loc_distribution,
        }

        if self.env_name == "tsp":
            self.base_env = TSPEnv(
                generator=None,
                generator_params=generator_params,
                seed=seed,
                device=self.device
            )
        elif self.env_name == "cvrp":
            self.base_env = CVRPEnv(
                generator=None,
                generator_params=generator_params,
                seed=seed,
                device=self.device
            )
        elif self.env_name == "op":
            self.base_env = OPEnv(
                generator=None,
                generator_params=generator_params,
                seed=seed,
                device=self.device
            )
        else:
            raise ValueError(f"Unsupported RL4CO routing env: {env_name}")

        self._td: Optional[TensorDict] = None

    def reset(self) -> Tuple[TensorDict, List[Dict[str, Any]]]:
        """Reset all sub-environments."""
        self.done = False
        batch_size = Size([self.env_num])
        td = self.base_env.reset(batch_size=batch_size)
        ################# 强制同步
        for i in range(1,self.env_num):
            td['locs'][i]=td['locs'][0] 
        #################
        self._td = td
        infos = [{}]*self.env_num
        self.actions=[]
        obs = self.build_obs(self._td)
        return obs, infos
    
    def get_greedy_actions(self):
        actions = []
        for i in range(self.env_num):
            action = greedy_policy(self._td, i)
            print(action)
            actions.append(action)
        return actions

    def build_obs(self, td):
        """Convert TensorDict numeric state into textual observations.

        Returns a list of string observations, one per batch element.
        """
        batch_size = td.batch_size[0] if td.batch_size else 1
        obs_list: List[str] = []

        for i in range(batch_size):
            # TSP / routing common: extract locs and respect optional mask
            locs = td["locs"][i]
            # extract first_node and current_node if present
            first_node = None
            current_node = None
            if self.actions!=[]:
                fn = _to_numpy(td["first_node"][i])
                first_node = int(fn) if hasattr(fn, "__int__") else int(fn[0])
                cn = _to_numpy(td["current_node"][i])
                current_node = int(cn) if hasattr(cn, "__int__") else int(cn[0])
            
            if "locs_mask" in td.keys():
                mask = td["locs_mask"][i]
                if mask.numel() > 0:
                    valid_n = int(mask.sum().item())
                    locs = locs[:valid_n]

            # Scale coordinates to integers for more compact textual representation
            locs_np = _to_numpy(locs)
            try:
                locs_scaled = (locs_np * 1000).astype(int)
            except Exception:
                # Fallback: if already integer-like
                locs_scaled = np.array(locs_np, dtype=int)

            # Build per-environment string depending on env type
            # prepend metadata: start, current, trajectory
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

            if self.env_name == "tsp":
                lines = []
                for node_idx, (x, y) in enumerate(locs_scaled.tolist()):
                    lines.append(f"Node {node_idx}, coordinates: [{x}, {y}];")
                obs_str = " ".join(lines) + "\n" +meta_prefix
            elif self.env_name == "cvrp":
                demands = td.get("demand", None)
                if demands is not None:
                    d_np = _to_numpy(demands[i])
                else:
                    d_np = None
                capacity_tensor = td.get("capacity", td.get("vehicle_capacity", None))
                capacity = None
                if capacity_tensor is not None:
                    try:
                        capacity = float(_to_numpy(capacity_tensor)[0])
                    except Exception:
                        capacity = None

                lines = []
                for node_idx, (x, y) in enumerate(locs_scaled.tolist()):
                    demand_val = int(d_np[node_idx]) if (d_np is not None and node_idx < len(d_np)) else 0
                    lines.append(f"Node {node_idx}, coordinates: [{x}, {y}], demand: {demand_val};")
                cap_str = f" Vehicle capacity: {int(capacity)}." if capacity is not None else ""
                obs_str = " ".join(lines) + cap_str
            elif self.env_name == "op":
                prize = td.get("prize", None)
                if prize is not None:
                    p_np = _to_numpy(prize[i])
                else:
                    p_np = None
                # max length fields
                if "max_length" in td.keys():
                    max_len_tensor = td["max_length"][i]
                elif "max_route_length" in td.keys():
                    max_len_tensor = td["max_route_length"][i]
                else:
                    max_len_tensor = None
                max_route_length = None
                if max_len_tensor is not None:
                    try:
                        max_route_length = float(_to_numpy(max_len_tensor).item())
                    except Exception:
                        try:
                            max_route_length = float(_to_numpy(max_len_tensor)[0])
                        except Exception:
                            max_route_length = None

                lines = []
                for node_idx, (x, y) in enumerate(locs_scaled.tolist()):
                    prize_val = int(p_np[node_idx]) if (p_np is not None and node_idx < len(p_np)) else 0
                    lines.append(f"Node {node_idx}, coordinates: [{x}, {y}], prize: {prize_val};")
                max_len_str = f" Max route length: {max_route_length}." if max_route_length is not None else ""
                obs_str = " ".join(lines) + max_len_str
            else:
                obs_str = ""

            obs_list.append(obs_str)

        return obs_list

    def action_projection(self,env_idx, action):
        """Project an (possibly invalid) action into a valid one according to the current `action_mask` in `self._td`.

        If the provided action is valid (mask==True) it is returned unchanged.
        If it is invalid, a random valid action is sampled uniformly from available actions.
        If no mask is present or no valid actions exist, the original action is returned.
        """
        try:
            if not isinstance(self._td, TensorDict):
                return action
            if "action_mask" not in self._td.keys():
                return action

            mask = self._td.get("action_mask")[env_idx]
            # mask may be batched: prefer first batch element
            if isinstance(mask, torch.Tensor):
                if mask.dim() == 2 and mask.shape[0] >= 1:
                    mask0 = mask[0]
                else:
                    mask0 = mask
                mask_bool = mask0.bool()
                # if no available action, return original
                if mask_bool.sum().item() == 0:
                    return action

                # convert action to int
                try:
                    act_int = int(action) if not isinstance(action, torch.Tensor) else int(action.item())
                except Exception:
                    act_int = None

                if act_int is not None and 0 <= act_int < mask_bool.shape[0] and mask_bool[act_int]:
                    return action

                # sample a random available action (uniform over available)
                idx = torch.multinomial(mask_bool.float(), 1).item()
                print("Invalid action! Using random policy replace.")
                return int(idx)
        except Exception:
            # on any error, fall back to original action
            return action
        return action
    
    def actions_projection(self, actions):
        projected_actions = []
        for i,action in enumerate(actions):
            action = self.action_projection(i,action)
            projected_actions.append(action)
        return projected_actions

    def step(self, action) -> Tuple[TensorDict, Any, Any, List[Dict[str, Any]]]:
        """Step all environments with integer actions."""
        if self.done == True:
            obs = ["The enviroment has closed."]*self.env_num
            rewards = [0]*self.env_num
            dones = [True]*self.env_num
            info = [{}]*self.env_num
            return obs, rewards, dones, info
        # project action into valid space before applying
        projected_action = self.actions_projection(action)
        self.actions.append(projected_action)
        action_tensor = torch.as_tensor(
            projected_action, device=self.device, dtype=torch.int64
        )
        # self._td中包括action mask，需要进行action projection
        self._td.set("action", action_tensor)
        out = self.base_env.step(self._td)
        next_td = out["next"] if isinstance(out, dict) and "next" in out else out
        if not isinstance(next_td, TensorDict):
            raise TypeError(f"Expected TensorDict from step, got {type(next_td)}")
        self._td = next_td

        rewards = [0]*self.env_num
        dones = next_td["done"]
        if dones.all() == True:
            actions = torch.as_tensor(self.actions, device=self.device, dtype=torch.int64) # step*batch_size
            actions = actions.transpose(1,0) # batch_size*step
            rewards = self.base_env.get_reward(self._td, actions)
            self.done=True
        infos = [{}]*self.env_num # placeholder
        obs = self.build_obs(self._td)   
        return obs, rewards, dones, infos
    

class RouteEnvs(gym.Env):
    def __init__(self, env_name, seed, env_num, group_n, device, resources_per_worker, is_train=True, env_kwargs={}):
        super().__init__()

        self.env_name = env_name
        self.num_processes = env_num*group_n
        self.env_num=env_num
        self.group_n = group_n

        # Ensure Ray is initialized
        if not ray.is_initialized():
            ray.init()

        # Extract generator params if provided
        generator_params = env_kwargs.get("generator_params", {}) if isinstance(env_kwargs, dict) else {}
        num_loc = generator_params.get("num_loc", 10)
        loc_distribution = generator_params.get("loc_distribution", "uniform")
        if not isinstance(num_loc,list):
            num_loc = [num_loc]*self.env_num
        if not isinstance(loc_distribution,list):
            loc_distribution=[loc_distribution]*self.env_num

        # Create Ray remote actors. If resources_per_worker is empty, call ray.remote(RouteWorker)
        if resources_per_worker:
            env_worker = ray.remote(**resources_per_worker)(RouteWorker)
        else:
            env_worker = ray.remote(RouteWorker)
        self.workers = []
        # Create one worker per group and replicate it `group_n` times so that
        # instances within the same group are identical (share the same actor).
        for g in range(env_num):
            worker = env_worker.remote(env_name, seed + g, self.group_n, device, num_loc[g], loc_distribution[g])
            self.workers.append(worker)

    def step(self, actions):
        assert len(actions) == self.num_processes, "The num of actions must be equal to the num of processes"

        # action首先需要按照group划分成env_num*group_num
        actions = np.array(actions)
        actions = actions.reshape((self.env_num,self.group_n))
        # Send step commands to all workers
        futures = [worker.step.remote(actions[i]) for i, worker in enumerate(self.workers)]
        results = ray.get(futures) # env_num,4,group_n

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
        results = ray.get(futures) # env_num*2*group_num
        text_obs_list=[]
        info_list = []
        self.has_done=[False]*self.num_processes
        for i in range(self.env_num):
            for j in range(self.group_n):
                text_obs_list.append(results[i][0][j])
                info_list.append(results[i][1][j])

        return text_obs_list, info_list

    def close(self):
        for worker in self.workers:
            try:
                ray.kill(worker)
            except Exception:
                pass


def build_route_envs(
    env_name: str = "tsp",
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    device: str = "cpu",
    generator_params: Optional[Dict[str, Any]] = None,
    rl4co_kwargs: Optional[Dict[str, Any]] = None,
):
    # Package generator / rl4co kwargs into env_kwargs for RouteEnvs.
    env_kwargs: Dict[str, Any] = {}
    if generator_params is not None:
        env_kwargs["generator_params"] = generator_params
    if rl4co_kwargs is not None:
        env_kwargs["rl4co_kwargs"] = rl4co_kwargs

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
        env_kwargs=env_kwargs,
    )

if __name__ == "__main__":
    # Simple smoke test for RouteEnvs
    print("Running simple smoke test for RouteEnvs...")
    try:
        env = build_route_envs(
            env_name="tsp",
            seed=0,
            env_num=1,
            group_n=1,
            device="cpu",
            generator_params={"num_loc": 5},
            rl4co_kwargs={},
        )
        obs, infos = env.reset()
        print("Reset text obs (first):", obs)

        actions = [0]  # choose node 0 as a simple action for single-process env
        obs2, rewards, dones, infos2 = env.step(actions)
        for i in range(1):
            action = env.workers[i].get_greedy_actions.remote()
            print(action)
    except Exception as e:
        print("Smoke test failed with exception:", e)
    finally:
        try:
            env.close()
        except Exception:
            pass