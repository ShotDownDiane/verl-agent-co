import ray
import torch
import random
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from tensordict import TensorDict
from torch import Size

from .base_env import BaseCOWorker, BaseCOEnvs
from .config_sampler import ConfigSampler

# --- Import Specific Envs ---
from rl4co.envs.routing.tsp.env import TSPEnv
from rl4co.envs.routing.cvrp.env import CVRPEnv
from rl4co.envs.routing.op.env import OPEnv

try:
    from rl4co.envs.graph import FLPEnv, MCPEnv
    try:
        from rl4co.envs.graph import MCLPEnv
    except ImportError:
        MCLPEnv = None
    try:
        from rl4co.envs.graph import STPEnv
    except ImportError:
        STPEnv = None
except ImportError:
    FLPEnv, MCLPEnv, MCPEnv, STPEnv = None, None, None, None

# --- Import Observation Builders ---
from .route_obs import build_obs_tsp, build_obs_cvrp, build_obs_op
from .graph_obs import build_obs_flp, build_obs_mclp, build_obs_mcp, build_obs_stp

class MixedWorker(BaseCOWorker):
    """
    通用 Worker，支持动态参数重置。
    (这部分代码与上一版保持一致，为了完整性再次列出)
    """
    
    ENV_CONFIG = {
        'tsp': {'cls': TSPEnv, 'builder': build_obs_tsp},
        'cvrp': {'cls': CVRPEnv, 'builder': build_obs_cvrp},
        'op': {'cls': OPEnv, 'builder': build_obs_op},
        'flp': {'cls': FLPEnv, 'builder': build_obs_flp},
        'mclp': {'cls': MCLPEnv, 'builder': build_obs_mclp},
        'mcp': {'cls': MCPEnv, 'builder': build_obs_mcp},
        'stp': {'cls': STPEnv, 'builder': build_obs_stp},
    }

    def __init__(self, env_name, seed, env_num, device, return_topk_options=0, **kwargs):
        self.image_obs = kwargs.get('image_obs', 'rgb')
        self.env_kwargs = kwargs.get('env_kwargs', {})
        self.seed = seed
        super().__init__(env_name, seed, env_num, device, return_topk_options, **kwargs)

    def _init_env(self, seed, **kwargs):
        env_key = self.env_name.lower()
        if env_key not in self.ENV_CONFIG:
             raise ValueError(f"Unsupported Mixed env: {self.env_name}")
        config = self.ENV_CONFIG[env_key]
        env_cls = config['cls']
        if env_cls is None: raise ImportError(f"Environment {self.env_name} unavailable.")

        generator_params = kwargs.copy()
        print(f"env_name: {self.env_name}, generator_params:", generator_params)
        generator = generator_params.pop("generator", None)
        
        return env_cls(
            generator=generator,
            generator_params=generator_params,
            seed=seed,
            device=self.device
        )
    
    def reset_with_config(self, new_config: Dict[str, Any]) -> Tuple[List[str], List[Dict[str, Any]]]:
        """核心：接收新配置 -> 重建环境 -> Reset"""

        if 'env_name' in new_config:
            self.env_name = new_config['env_name']
        
        # 更新参数，排除 env_name
        self.env_kwargs = {k: v for k, v in new_config.items() if k != 'env_name'}
        
        # 改变种子以保证随机性
        self.seed = (self.seed + 1234567) % (2**32)

        # 重新实例化 (Hard Reset)
        self.base_env = self._init_env(self.seed, **self.env_kwargs)
        return self.reset()

    # ... (reset, build_obs 等标准方法省略，保持原样即可) ...
    def reset(self):
        self.done = False
        batch_size = Size([self.env_num])
        td = self.base_env.reset(batch_size=batch_size)
        if self.synchronous: td = self._sync_instances(td)
        self._td = td
        self.actions = []
        infos = [{}] * self.env_num
        self._td = self._post_reset_hook(self._td)
        return self.build_obs(self._td), infos

    def build_obs(self, td):
        env_key = self.env_name.lower()
        builder = self.ENV_CONFIG[env_key]['builder']
        if env_key in ['tsp', 'cvrp', 'op']:
             return builder(
                td=td, 
                env_num=self.env_num, 
                trajectory=self.actions,
                top_k=self.topk_k,
                image_obs=self.image_obs,
            )
        else:
            try: return builder(td, self.env_num, top_k=self.topk_k, image_obs=self.image_obs)
            except: return builder(td, self.env_num, top_k=self.topk_k)
    
    def _sync_instances(self, td):
        for i in range(1, self.env_num):
            for key in td.keys(): td[key][i] = td[key][0]
        return td
    
    def _post_reset_hook(self, td):
        if self.return_topk_options:
            self._td["action"] = torch.tensor([0]*self.env_num, device=self.device)
            self._td = self.base_env.step(self._td)['next']
            self.actions.append([0]*self.env_num)
        return self._td

class MixedCOEnvs(BaseCOEnvs):
    """
    支持 Config Sampler 的环境管理器。
    """
    def __init__(
        self, 
        env_configs: List[Dict[str, Any]], # 初始配置，决定 Worker 数量和类型
        config_sampler: Callable[[], List[Dict[str, Any]]] = None, # 核心：采样函数
        seed: int = 0,
        group_n: int = 1,
        device: str = "cpu",
        resources_per_worker: Dict[str, Any] = {},
        return_topk_options: int = 0,
        env_kwargs: Dict[str, Any] = {}
    ):
        self.env_configs = env_configs
        self.config_sampler = config_sampler
        self.env_num = len(env_configs)
        self.group_n = group_n
        
        super().__init__(
            worker_cls=MixedWorker,
            env_name="mixed",
            seed=seed,
            env_num=self.env_num,
            group_n=group_n,
            device=device,
            resources_per_worker=resources_per_worker,
            return_topk_options=return_topk_options,
            env_kwargs=env_kwargs 
        )

    def _get_worker_args(self, worker_idx, env_name, seed, group_n, device, return_topk_options, env_kwargs):
        # 初始启动参数
        config = self.env_configs[worker_idx]
        current_env_name = config.get('env_name', 'tsp')
        worker_kwargs = config.copy()
        if 'env_name' in worker_kwargs: del worker_kwargs['env_name']
        worker_kwargs.update(env_kwargs)
        return (current_env_name, seed + worker_idx, group_n, device, return_topk_options), worker_kwargs
    
    def reset(self):
        """
        每次 Reset 时：
        1. 调用 sampler 生成一组新的配置 (N个)。
        2. 将第 i 个配置发送给第 i 个 Worker。
        """
        # 1. 获取新配置列表
        if self.config_sampler is not None:
            new_configs_list = self.config_sampler()
            print(new_configs_list)
            # 完整性检查：确保生成的配置数量与 Worker 数量一致
            assert len(new_configs_list) == self.env_num, \
                f"Sampler generated {len(new_configs_list)} configs, but there are {self.env_num} workers."
        else:
            # 如果没有 sampler，就复用初始配置（相当于不做随机化）
            new_configs_list = self.env_configs

        # 2. 分发配置
        futures = []
        for g in range(self.env_num):
            # 将对应的 config 发送给对应的 worker
            futures.append(self.workers[g].reset_with_config.remote(new_configs_list[g]))

        results = ray.get(futures)
        
        # 3. 收集结果
        obs_list, info_list = [], []
        for i in range(self.env_num):
            w_obs, w_info = results[i]
            if isinstance(w_obs, list):
                obs_list.extend(w_obs)
                info_list.extend(w_info)
            else:
                obs_list.append(w_obs)
                info_list.append(w_info)
        return obs_list, info_list

def build_mixed_envs(
    env_configs: List[Dict[str, Any]], # 必填，定义初始结构
    difficulty_levels: Optional[List[int]] = None, # 选填，定义随机策略
    seed: int = 0,
    group_n: int = 1,
    device: str = "cpu",
    return_topk_options: int = 0,
    resources_per_worker: Optional[Dict[str, Any]] = None
):
    if resources_per_worker is None: resources_per_worker = {}
    config_sampler = ConfigSampler(
        worker_schema=env_configs,
        step_levels=difficulty_levels,
    )
    return MixedCOEnvs(
        env_configs=env_configs,
        config_sampler=config_sampler,
        seed=seed,
        group_n=group_n,
        device=device,
        resources_per_worker=resources_per_worker,
        return_topk_options=return_topk_options
    )

    