# tests/test_poi_env.py

import pytest
import numpy as np
import gymnasium as gym
import ray

from .online_env_tensor import StationPlacement  # Adjust import based on your project structure
from .envs import PoiMultiProcEnv
from .online_env_llm import LLMWrapperEnv
from typing import List

# Mock base_env for testing
"""Test LLMWrapperEnv single-step behavior."""
location = "Qiaonan"
graph_file = f"Data/Graph/{location}/{location}.graphml"
node_file = f"Data/Graph/{location}/nodes_extended_{location}.txt"
plan_file = f"Data/Graph/{location}/existingplan_{location}.pkl"
# -------------------------
# Fixtures
# -------------------------
@pytest.fixture(scope="module", autouse=True)
def ray_init():
    ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()

# -------------------------
# Tests
# -------------------------
def test_llm_wrapper_env_basic():


    base_env = StationPlacement(graph_file, node_file, plan_file)
    env = LLMWrapperEnv(base_env=base_env)
    
    obs, info = env.reset(seed=42)
    assert isinstance(obs, str) and "[Observation]" in obs
    
    step_action = {"answer": "A", "summary": "test"}
    next_obs, reward, terminated, truncated, info = env.step(step_action)
    
    assert isinstance(next_obs, str) and "Observation" in next_obs
    assert isinstance(reward, (int, float))
    assert terminated is False and truncated is False
    assert "available_actions" not in info  # LLMWrapperEnv doesn't inject it

def test_poi_multi_proc_env():
    """Test multi-process behavior with dummy base env."""
    env = PoiMultiProcEnv(
        seed=1, env_num=2, group_n=1, env_kwargs={"base_env": StationPlacement(graph_file, node_file, plan_file)}
    )
    obs_list, info_list = env.reset()
    assert len(obs_list) == 2
    assert isinstance(obs_list[0], str)
    assert isinstance(info_list, list)
    
    actions = ["A", "B"]
    outs = env.step(actions)
    assert len(outs) == 5
    obs2, rews, terms, truns, infos = outs
    assert all(isinstance(o, str) for o in obs2)
    assert rews == ["A", "B"] or isinstance(rews, list)
    assert all(isinstance(d, bool) for d in terms)
    assert isinstance(infos, list)

def test_integration_poi_roundtrip():
    """Full round-trip with projection and manager."""
    from sys import path
    path.append("../..")  # Adjust path to import from parent directory
    from env_manager import PoiEnvironmentManager
    from .envs import PoiMultiProcEnv
    
    def dummy_proj(text_actions: List[str]):
        return ["A" for _ in text_actions], [1 for _ in text_actions]
    
    env = PoiMultiProcEnv(seed=0, env_num=2, group_n=1, env_kwargs={"base_env": StationPlacement(graph_file, node_file, plan_file)})
    mgr = PoiEnvironmentManager(env, dummy_proj, env_name="poi_placement")
    
    obs_struct, infos = mgr.reset()
    assert "text" in obs_struct and isinstance(obs_struct["text"], list)
    
    text_in = obs_struct["text"]
    next_obs, rews, dones, infos2 = mgr.step(text_in)
    assert isinstance(next_obs["text"], list)
    assert isinstance(rews, np.ndarray)
    assert rews.shape[0] == 2
    assert isinstance(dones, np.ndarray)
    assert infos2 and isinstance(infos2, list)
