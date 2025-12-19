# agent_system/environments/poi_placement/env.py
import ray
import gymnasium as gym
from gymnasium import Env
import numpy as np
from .online_env_llm import LLMWrapperEnv  # 你之前定义的 LLMWrapperEnv
from .. import utils

@ray.remote(num_cpus=0.25)
class PoiWorker:
    def __init__(self, seed, env_kwargs):
        np.random.seed(seed)
        self.env = LLMWrapperEnv(**env_kwargs)

    def reset(self, idx=None):
        obs, info = self.env.reset(seed=idx)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        try:
            self.env.close()
        except:
            pass

class PoiMultiProcEnv(Env):
    def __init__(self, seed=0, env_num=1, group_n=1, env_kwargs=None):
        super().__init__()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.env_kwargs = env_kwargs or {}
        self.env_num = env_num
        self.group_n = group_n
        self.num_workers = env_num * group_n
        self.workers = [
            PoiWorker.remote(seed + i//group_n, self.env_kwargs)
            for i in range(self.num_workers)
        ]
        self._rng = np.random.RandomState(seed)

    def reset(self):
        idxs = self._rng.randint(0, 10000, size=self.num_workers)
        futures = [w.reset.remote(i) for w, i in zip(self.workers, idxs)]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def step(self, actions):
        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]
        results = ray.get(futures)
        obs, rews, terms, truns, infos = zip(*results)
        return list(obs), list(rews), list(terms), list(truns), list(infos)

    def render(self, mode="human", env_idx=None):
        if env_idx is not None:
            return ray.get(self.workers[env_idx].render.remote(mode))
        return ray.get([w.render.remote(mode) for w in self.workers])

    def close(self):
        for w in self.workers:
            w.close.remote()
            ray.kill(w)


def build_poi_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""

    location = env_kwargs.get("location", "default_location")
    root_dir = utils.get_root_dir()
    graph_file = f"{root_dir}/agent_system/environments/env_package/poi_placement/Data/Graph/{location}/{location}.graphml"
    node_file = f"{root_dir}/agent_system/environments/env_package/poi_placement/Data/Graph/{location}/nodes_extended_{location}.txt"
    plan_file = f"{root_dir}/agent_system/environments/env_package/poi_placement/Data/Graph/{location}/existingplan_{location}.pkl"
    env_kwargs = {
        "my_graph_file": graph_file,
        "my_node_file": node_file,
        "my_plan_file": plan_file
    }
    return PoiMultiProcEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        env_kwargs=env_kwargs,
    )