from typing import List, Dict, Any, Optional
import torch
from tensordict.tensordict import TensorDict

# Import RL4CO graph environments
# Note: Ensure these are installed/accessible in the environment
try:
    from rl4co.envs.graph import FLPEnv, MCLPEnv, MCPEnv, STPEnv
except ImportError:
    # Fallback or placeholder if direct import fails, though user environment suggests they exist
    FLPEnv, MCLPEnv, MCPEnv, STPEnv = None, None, None, None

from .base_env import BaseCOWorker, BaseCOEnvs
from .graph_obs import build_obs_flp, build_obs_mclp, build_obs_mcp, build_obs_stp

class GraphWorker(BaseCOWorker):
    """
    Universal Worker for Graph environments (FLP, MCLP, MCP, STP) in RL4CO.
    """
    
    # Map environment names to their corresponding classes and observation builders
    ENV_CONFIG = {
        'flp': {'cls': FLPEnv, 'builder': build_obs_flp},
        'mclp': {'cls': MCLPEnv, 'builder': build_obs_mclp},
        'mcp': {'cls': MCPEnv, 'builder': build_obs_mcp},
        'stp': {'cls': STPEnv, 'builder': build_obs_stp},
    } 

    def __init__(
        self,
        env_name: str = "mclp",
        seed: int = 0,
        env_num: int = 1,
        device: str = "cpu",
        num_loc: int = 20, # Default for graph envs if needed
        return_topk_options: int = 0,
        env_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.num_loc = num_loc
        self.env_kwargs = env_kwargs

        super().__init__(
            env_name=env_name,
            seed=seed,
            env_num=env_num,
            device=device,
            return_topk_options=return_topk_options
        )

    def _init_env(self, seed: int, **kwargs):
        """Initialize the specific RL4CO graph environment."""
        env_key = self.env_name.lower()
        
        if env_key not in self.ENV_CONFIG:
            available = list(self.ENV_CONFIG.keys())
            raise ValueError(f"Unsupported graph environment: {self.env_name}. Available: {available}")
            
        config = self.ENV_CONFIG[env_key]
        env_cls = config['cls']
        
        if env_cls is None:
             raise ImportError(f"Could not import environment class for {self.env_name}. Please check rl4co installation.")

        # Prepare generator params
        # Some graph envs (like FLP, MCLP) use num_loc/num_nodes
        # We try to pass them if available in self.num_loc or env_kwargs
        generator_params = {}
        if self.num_loc is not None:
             # Common param name. Note: Specific envs might use 'num_cust', 'num_nodes' etc.
             # RL4CO usually standardizes on generator_params
             # For now we put it in generic 'num_loc' and hope the env generator handles it or we map it.
             # Actually, let's look at RouteWorker. It sets 'num_loc'.
             generator_params["num_loc"] = self.num_loc

        # Merge with any extra generator params from env_kwargs
        if self.env_kwargs and "generator_params" in self.env_kwargs:
            generator_params.update(self.env_kwargs["generator_params"])

        # Initialize environment
        return env_cls(
            generator_params=generator_params,
            seed=seed,
            device=self.device,
            **kwargs
        )
    
    def _sync_instances(self, td: TensorDict) -> TensorDict:
        """Force synchronization of locations across the batch."""
        for i in range(1, self.env_num):
            for key in td.keys():
                td[key][i] = td[key][0]
        return td


    def build_obs(self, td: TensorDict) -> List[str]:
        """Delegate observation building to specific functions defined in graph_obs.py"""
        env_key = self.env_name.lower()
        builder = self.ENV_CONFIG[env_key]['builder']
        
        if builder:
            # Graph builders typically take (td, env_num, top_k)
            # They don't typically use trajectory like Route envs do, 
            # or if they do, it's embedded in td (e.g. 'chosen').
            return builder(td, self.env_num, top_k=self.topk_k)
        else:
            return [f"No observation builder defined for {self.env_name}"] * self.env_num


class GraphEnvs(BaseCOEnvs):
    """Ray-based Wrapper for Graph Environments."""
    
    def __init__(self, env_name, seed, env_num, group_n, device, resources_per_worker, is_train=True, return_topk_options=True, env_kwargs=None):
        
        # Prepare params to be stored for _get_worker_args
        self.num_loc_list = None
        
        generator_params = env_kwargs.get("generator_params", {}) if isinstance(env_kwargs, dict) else {}
        # Try to find num_loc or equivalent
        num_loc = generator_params.get("num_loc", 20) 
        
        # Handle list expansion for per-worker configuration
        if not isinstance(num_loc, list):
            self.num_loc_list = [num_loc] * env_num
        else:
            self.num_loc_list = num_loc
            
        # Call Base init
        super().__init__(
            worker_cls=GraphWorker,
            env_name=env_name,
            seed=seed,
            env_num=env_num,
            group_n=group_n,
            device=device,
            resources_per_worker=resources_per_worker,
            return_topk_options=return_topk_options,
            env_kwargs=env_kwargs
        )

    def _get_worker_args(self, worker_idx, env_name, seed, group_n, device, return_topk_options, env_kwargs):
        """Prepare specific arguments for GraphWorker."""
        
        current_num_loc = self.num_loc_list[worker_idx]
        
        # Create a worker-specific env_kwargs to ensure scalars are passed
        worker_env_kwargs = env_kwargs.copy()
        if "generator_params" in worker_env_kwargs:
            worker_env_kwargs["generator_params"] = worker_env_kwargs["generator_params"].copy()
            worker_env_kwargs["generator_params"]["num_loc"] = current_num_loc

        # args matching GraphWorker.__init__ signature
        args = (
            env_name, 
            seed + worker_idx, 
            env_num := group_n, # Note: BaseCOEnvs passes group_n as env_num to worker
            device, 
            current_num_loc, 
            return_topk_options, 
            worker_env_kwargs
        )
        return args, {}

def build_graph_env(
    env_name: str = "mclp",
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1, # Default to 1 env per worker if not specified
    device: str = "cpu",
    generator_params: Optional[Dict[str, Any]] = None,
    rl4co_kwargs: Optional[Dict[str, Any]] = None,
    return_topk_options: int = 0
):
    # Package generator / rl4co kwargs into env_kwargs
    env_kwargs: Dict[str, Any] = {}
    if generator_params is not None:
        env_kwargs["generator_params"] = generator_params
    if rl4co_kwargs is not None:
        env_kwargs["rl4co_kwargs"] = rl4co_kwargs

    resources_per_worker: Dict[str, Any] = {}

    return GraphEnvs(
        env_name=env_name,
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        device=device,
        resources_per_worker=resources_per_worker,
        return_topk_options=return_topk_options,
        env_kwargs=env_kwargs
    )


