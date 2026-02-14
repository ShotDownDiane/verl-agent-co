# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf, DictConfig, ListConfig
from types import SimpleNamespace
from PIL import Image


def _to_container(obj, resolve=True):
    """Safely convert SimpleNamespace or OmegaConf object to dict/list.
    
    This helper function allows make_envs to work with both OmegaConf configs
    (used in production) and SimpleNamespace objects (used in manual testing).
    """
    if isinstance(obj, SimpleNamespace):
        # Convert SimpleNamespace to dict recursively
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, SimpleNamespace):
                    result[key] = _to_container(value, resolve=resolve)
                elif isinstance(value, (list, tuple)):
                    result[key] = [
                        _to_container(item, resolve=resolve) if isinstance(item, SimpleNamespace) else item
                        for item in value
                    ]
                else:
                    result[key] = value
            return result
        else:
            return {}
    elif isinstance(obj, (DictConfig, ListConfig)):
        # Use OmegaConf.to_container for OmegaConf objects
        return OmegaConf.to_container(obj, resolve=resolve)
    else:
        return obj


class RL4COEnvironmentManager(EnvironmentManagerBase):
    """EnvironmentManager for RL4CO routing envs (starting with TSP).
    """
    SYSTEM_TEMPLATEs ={
        "tsp": RL4CO_TSP_SYSTEM_TEMPLATE,
        "cvrp": RL4CO_CVRP_SYSTEM_TEMPLATE,
        "flp": RL4CO_FLP_SYSTEM_TEMPLATE,
        "mclp": RL4CO_MCLP_SYSTEM_TEMPLATE,
        "stp": RL4CO_STP_SYSTEM_TEMPLATE,
        "tdtsp": RL4CO_TDTSP_SYSTEM_TEMPLATE,
        "tdtsp_tw": RL4CO_TDTSP_TW_SYSTEM_TEMPLATE,
        "tdvrp": RL4CO_TDVRP_SYSTEM_TEMPLATE,
        "lrp": RL4CO_LRP_SYSTEM_TEMPLATE,
    } 
    USER_TEMPLATEs ={
        "tsp": RL4CO_TSP_USER_TEMPLATE,
        "cvrp": RL4CO_CVRP_USER_TEMPLATE,
        "flp": RL4CO_FLP_USER_TEMPLATE,
        "mclp": RL4CO_MCLP_USER_TEMPLATE,
        "stp": RL4CO_STP_USER_TEMPLATE,
        "tdtsp": RL4CO_TDTSP_USER_TEMPLATE,
        "tdtsp_tw": RL4CO_TDTSP_TW_USER_TEMPLATE,
        "tdvrp": RL4CO_TDVRP_USER_TEMPLATE,
        "lrp": RL4CO_LRP_USER_TEMPLATE,
    }
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        
        # [优化] 更健壮的环境名解析
        rl4co_cfg_raw = getattr(config.env, "rl4co", None)
        self.rl4co_env_name = getattr(config.env, "env_name", "tsp").lower()
        # 处理 "rl4co/tsp" 或 "tsp" 两种情况
        self.sub_env_name = self.rl4co_env_name.split("/")[-1] 
        
        self.return_topk_options = config.data.return_topk_options
        self.trajectorys = []
        
        # Reward Configuration
        rl4co_cfg = rl4co_cfg_raw if rl4co_cfg_raw else {}
        self.use_format_reward = getattr(rl4co_cfg, "use_format_reward", True)
        
        # [关键修改] 惩罚力度要大
        self.format_reward_weight = getattr(rl4co_cfg, "format_reward_weight", 0.01) # 如果格式对，给一点点甜头
        self.format_penalty = getattr(rl4co_cfg, "format_penalty", -0.5) # [新增] 如果格式错，重罚
        
        # [新增] 环境奖励缩放，用于对齐量级 (例如 TSP 距离通常很大，乘 0.1 或 0.01)
        self.env_reward_scale = getattr(rl4co_cfg, "env_reward_scale", 1) 

        self.SYSTEM_TEMPLATE = self.SYSTEM_TEMPLATEs.get(self.sub_env_name, RL4CO_TSP_SYSTEM_TEMPLATE)
        self.USER_TEMPLATE = self.USER_TEMPLATEs.get(self.sub_env_name, RL4CO_TSP_USER_TEMPLATE)
        self._env_rewards = [[] for _ in range(envs.env_num * envs.group_n * envs.batch_size)]
        self._format_rewards = [[] for _ in range(envs.env_num * envs.group_n * envs.batch_size)]
        self._final_rewards = [[] for _ in range(envs.env_num * envs.group_n * envs.batch_size)]

        # [新增] 图片尺寸配置
        # self.image_max_size = getattr(config.data, "image_max_pixels", 448) # 默认对齐 448

        super().__init__(envs, projection_f, config)

    def path_to_numpy(self, img: Any) -> np.ndarray:
        if isinstance(img, np.ndarray):
            return img
        assert isinstance(img, str), "img must be a path string or a numpy array"
        
        image = Image.open(img)
        if image.mode != 'RGB':
            image = image.convert('RGB')        
        return np.array(image)

    def reset(self, kwargs=None) -> Dict[str, Any]:
        # Some env implementations accept kwargs, others do not.
        try:
            res = self.envs.reset(kwargs=kwargs) if kwargs is not None else self.envs.reset()
        except TypeError:
            res = self.envs.reset()

        # Accept different return signatures:
        # (text_obs_list, image_obs_list, infos) or (text_obs_list, infos)
        obs_list, info_list = res

        batch_size = len(obs_list) if hasattr(obs_list, "__len__") else 1
        self.memory.reset(batch_size=batch_size)
        self.pre_text_obs = [""] * batch_size
        self.trajectorys = []

        image_obs_list = [self.path_to_numpy(obs['image']) for obs in obs_list] if 'image' in obs_list[0] else None
        has_image = image_obs_list is not None
        text_obs_list = self.build_text_obs(obs_list, has_image=has_image)
        observations = {"text": text_obs_list, "image": image_obs_list, "anchor": obs_list, "system_template": self.SYSTEM_TEMPLATE}    
        return observations, info_list

    def step(self, text_actions: List[str]):
        # 1. 投影动作
        actions, valids = self.projection_f(text_actions)

        # 2. 执行环境步骤
        res = self.envs.step(actions)
        # 解包 (兼容性处理)
        if len(res) == 5:
             next_obs, image_obs_list_raw, env_rewards, dones, infos = res
        else:
             next_obs, env_rewards, dones, infos = res

        # 3. 处理 Observation
        # [修复] 必须提取 text，否则 return 时会报错
        next_raw_text_obs = [obs['obs'] for obs in next_obs] 
        next_image_obs = [self.path_to_numpy(obs['image']) for obs in next_obs] if 'image' in next_obs[0] else None
        
        # 4. 存储历史
        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        env_rewards = to_numpy(env_rewards)
        valids = to_numpy(valids)

        # 5. [核心修改] 奖励计算逻辑
        final_rewards = np.zeros_like(env_rewards, dtype=np.float32)

        for i in range(len(env_rewards)):
            if valids[i] == 1:
                # Case A: 格式正确
                # Reward = (缩放后的环境奖励) + (格式奖励，通常给0或者很小)
                # 只有格式正确时，我们才关心它走得怎么样
                r_env = env_rewards[i] * self.env_reward_scale
                r_fmt = self.format_reward_weight * (1-dones[i]) # 格式奖励只在未完成时生效
                final_rewards[i] = r_env + r_fmt
                self._env_rewards[i].append(r_env)
                self._format_rewards[i].append(r_fmt)
                self._final_rewards[i].append(final_rewards[i])
            else:
                # Case B: 格式错误
                # Reward = 固定惩罚值 (例如 -1.0)
                # [关键] 直接忽略 env_rewards[i]，因为那是基于默认动作产生的噪音
                final_rewards[i] = self.format_penalty
        
        rewards = final_rewards

        # 6. 更新 Info (用于监控)
        for i, info in enumerate(infos):
            if info is None: infos[i] = {}
            infos[i]["is_action_valid"] = int(valids[i])
            infos[i]["raw_env_reward"] = float(env_rewards[i])
            infos[i]["final_reward"] = float(final_rewards[i])
            infos[i]["sum_env_reward"] = float(np.sum(self._env_rewards[i]))
            infos[i]["sum_format_reward"] = float(np.sum(self._format_rewards[i]))
            infos[i]["sum_final_reward"] = float(np.sum(self._final_rewards[i]))
            infos[i]["actions"] = actions[i]


        dones = to_numpy(dones) if dones is not None else None
        has_image = next_image_obs is not None
        
        text_obs = self.build_text_obs(next_obs, has_image)
        self.pre_text_obs = text_obs
        
        next_observations = {
            "text": text_obs, 
            "image": next_image_obs, 
            "anchor": text_obs, # [修复] 这里现在引用的是存在的变量
            "system_template": self.SYSTEM_TEMPLATE
        }
        
        return next_observations, rewards, dones, infos
    
    def build_text_obs(self, next_obs, has_image: bool = False) -> List[str]:
        postprocess_text_obs = []
        obs_str = [obs['obs'] for obs in next_obs]
        candidates_str = [obs['candidates'] for obs in next_obs]
        for i in range(len(next_obs)):
            obs = self.USER_TEMPLATE.format(obs_text=obs_str[i], candidates=candidates_str[i])
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


def make_envs(config):
    """
    Create enviroments 
    """ 
    group_n = config.env.rollout.n
    env_name = config.env.env_name

    
    from agent_system.environments.env_package.rl4co import (
        build_route_envs,
        build_graph_env,
        co_projection,
        co_projection_selected,
    )

    # Resolve rl4co-specific config (with sensible defaults)
    rl4co_env_name = env_name.split("/")[1]
    rl4co_device = "cpu" # default to cpu
    env_nums = config.data.train_batch_size
    return_topk_options = config.data.return_topk_options
        
    if rl4co_env_name in ["flp", "mclp", "stp"]:
        _envs = build_graph_env(
            env_name=rl4co_env_name,
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            device=rl4co_device,
            return_topk_options=return_topk_options,
        )
        _val_envs = build_graph_env(
            env_name=rl4co_env_name,
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            device=rl4co_device,
            return_topk_options=return_topk_options,
        )
        if return_topk_options > 0:
            projection_f = partial(
                co_projection_selected,
                env_name=rl4co_env_name,
            )
        else:
            projection_f = partial(
                co_projection,
                env_name=rl4co_env_name,
            )
        envs = RL4COEnvironmentManager(_envs, projection_f, config)
        val_envs = RL4COEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif rl4co_env_name == "mixed":
        from agent_system.environments.env_package.rl4co.mixed_env import build_mixed_envs

        sampler_config = _to_container(getattr(config.env, "sampler_config", None))
        if not sampler_config:
            # use default config
            worker_schema = [
                {"env_name": "tsp"}, {"env_name": "tsp"},
                {"env_name": "cvrp"}, {"env_name": "cvrp"},
                {"env_name": "flp"}, {"env_name": "flp"},
                {"env_name": "mclp"}, {"env_name": "mclp"},
                {"env_name": "stp"}, {"env_name": "stp"},
            ]
            difficulty_levels = [30, 40, 50, 60]


        # We pass the full list of mixed_configs as a pool.
        # The MixedCOEnvs will handle sampling one config per batch for synchronization.
        
        _envs = build_mixed_envs(
            env_configs=worker_schema,
            difficulty_levels=difficulty_levels,
            seed=config.env.seed,
            group_n=group_n,
            device=rl4co_device,
            return_topk_options=return_topk_options,
        )
        _val_envs = build_mixed_envs(
            env_configs=worker_schema,
            difficulty_levels=[50],
            seed=config.env.seed + 1000,
            group_n=1,
            device=rl4co_device,
            return_topk_options=return_topk_options,
        )

        if return_topk_options > 0:
            projection_f = partial(co_projection_selected, env_name="mixed")
        else:
            projection_f = partial(co_projection, env_name="mixed")
        
        envs = RL4COEnvironmentManager(_envs, projection_f, config)
        val_envs = RL4COEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        _envs = build_route_envs(
            env_name=rl4co_env_name,
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            device=rl4co_device,
            return_topk_options=return_topk_options,
            batch_size=config.env.batch_size,
        )
        _val_envs = build_route_envs(
            env_name=rl4co_env_name,
            seed=config.env.seed + 1000,
            env_num=1, # GPU的数量
            group_n=config.data.val_batch_size,
            device=rl4co_device,
            return_topk_options=return_topk_options,
            mode="test",
        )
        if return_topk_options > 0:
            projection_f = partial(
                co_projection_selected,
                env_name=rl4co_env_name,
            )
        else:
            projection_f = partial(
                co_projection,
                env_name=rl4co_env_name,
            )
        envs = RL4COEnvironmentManager(_envs, projection_f, config)
        val_envs = RL4COEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    

if __name__ == "__main__":
    """
    Manual quick-play supporting current backbones:
    - rl4co/*     : step-by-step routing (tsp/cvrp/op)
    - ml4co-kit/* : one-shot routing (tsp/cvrp/op)
    - rl4co/mixed : mixed environment wrapper
    """
    import time

    env_name = "rl4co/tdvrp"  # Testing mixed environment
    cfg = {
        "data":{
            "train_batch_size": 3, # Use batch size > 1 to verify sync
            "val_batch_size": 1,
            "return_topk_options": 26,
        },
        "env":{
            "env_name": env_name,
            "batch_size": 3,
            "rollout": {
                "n": 3,
            },
            "seed": 1234, # Fixed seed
        },
        "device":"cpu",
    }
    cfg = OmegaConf.create(cfg)
    print(f"[Manual test] env_name = {env_name}")
    
    t0 = time.time()
    envs, val_envs = make_envs(cfg)
    print("Init time:", time.time() - t0)

    # Reset multiple times to see different environments being sampled
    for reset_idx in range(1):
        print(f"\n[Reset {reset_idx+1}]")
        obs, infos = envs.reset()
        
        # Check if all envs in the batch are the same type (sync check)
        # We can infer type from the text observation or internal state if accessible
        # For now, let's print the first observation
        print("\n" + "=" * 80)
        print("Initial text observation (first env):")
        print("=" * 80)
        print(obs["text"][0])
        print("=" * 80)
        
        # Simple step test
        print("Stepping with dummy actions...")
        batch_size = len(obs["text"])
        
        # Create dummy actions (just "0" for simplicity, though invalid for some states it handles string input)
        while True:
            acts = ["<Observation> Option A [Node 30] anchors the geometric median of a dense, orphaned cluster in the South-West quadrant. </Observation>\n<Thought> Implementing Greedy Capture, this site secures a distance reduction of 15.2, maximizing immediate coverage and minimizing total distance for the unserved region. </Thought>\n<Decision> \\boxed{A} </Decision>"] * batch_size

            next_obs, rewards, dones, infos = envs.step(acts)
            
            if dones.all():
                print("\nTest completed.")
                for i in range(batch_size):
                    env_reward = infos[i].get("env_reward", 0)
                    final_reward = infos[i].get("final_reward", 0)
                    print(f"  Env {i}: Env={env_reward:.4f}, Final={final_reward:.4f}")

            done_flag = dones.all()
            if done_flag:
                print("\n" + "=" * 80)
                print("Episode finished!")
                print("=" * 80)
                break
