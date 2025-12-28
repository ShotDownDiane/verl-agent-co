import os
import json
import random
import argparse
import logging
from omegaconf import OmegaConf
from datetime import datetime
from functools import partial
from types import SimpleNamespace
from agent_system.environments.env_manager import RL4CORoutingEnvironmentManager, RL4COSchedulingEnvironmentManager
from agent_system.environments.env_package.rl4co import (
    build_rl4co_routing_envs,
    rl4co_projection,
)

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
    else:
        # Use OmegaConf.to_container for OmegaConf objects
        return OmegaConf.to_container(obj, resolve=resolve)

# Simple Random Agent for testing

class RandomAgent:
    def __init__(self, seed: int | None = None):
        if seed is not None:
            random.seed(seed)

    def sample_route(self, num_nodes: int):
        route = list(range(num_nodes))
        random.shuffle(route)
        return route

    def act_for_env(self, env_manager, idx: int):
        # determine node count from current_td if available
        num_nodes = None
        if hasattr(env_manager, "current_td") and env_manager.current_td is not None and "locs" in env_manager.current_td.keys():
            locs = env_manager.current_td["locs"][idx]
            # respect mask if present
            if "locs_mask" in env_manager.current_td.keys():
                mask = env_manager.current_td["locs_mask"][idx]
                if mask.numel() > 0:
                    num_nodes = int(mask.sum().item())
            if num_nodes is None:
                # first dim is nodes
                num_nodes = int(locs.shape[0])
        if num_nodes is None:
            num_nodes = 10

        # decide mode (one-step full route vs step-by-step single-index)
        try:
            one_step_mode = bool(args.one_step)
        except Exception:
            one_step_mode = False

        env_name_local = env_name.split("/")[-1].lower()
        route = self.sample_route(num_nodes)

        if one_step_mode:
            if env_name_local == "cvrp":
                payload = {"routes": [[0] + route[1:] + [0]]} if len(route) > 0 else {"routes": []}
            else:
                payload = {"route": route, "objective": 0}
            return json.dumps(payload)
        else:
            # step-by-step: propose a single node index (as string)
            next_idx = route[0] if route else 0
            return str(int(next_idx))

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", default="rl4co")
    parser.add_argument("--sub_env", default="tsp")
    parser.add_argument("--env_num", default="1")
    parser.add_argument("--group_size", default="1")
    parser.add_argument("--one_step", action="store_true", help="Run in one-step (full route) mode")
    parser.add_argument("--num_loc", default="20")

    parser.add_argument("--seed", default="0")
    args = parser.parse_args()

    env_name = args.env_name
    sub_env = args.sub_env if args.sub_env is not None else (env_name.split("/", 1)[1] if "/" in env_name else "tsp")
    env_num = int(args.env_num)
    group_n = int(args.group_size)
    # Populate minimal config-like structures expected by downstream code
    # Many utilities expect args.env and args.data to exist (OmegaConf-like).
    if not hasattr(args, "env"):
        args.env = SimpleNamespace()
    # rl4co config container
    if not hasattr(args.env, "rl4co"):
        args.env.rl4co = SimpleNamespace(env_name=sub_env, device="cpu", generator_params=SimpleNamespace(num_loc=int(args.num_loc)), rl4co_kwargs=SimpleNamespace())
    # ensure seed is available on args.env
    try:
        args.env.seed = int(args.seed)
    except Exception:
        args.env.seed = 0
    # data config container for batch sizes
    if not hasattr(args, "data"):
        args.data = SimpleNamespace(train_batch_size=env_num, val_batch_size=1)
    else:
        if not hasattr(args.data, "train_batch_size"):
            args.data.train_batch_size = env_num
        if not hasattr(args.data, "val_batch_size"):
            args.data.val_batch_size = 1

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"logs/vlm_agent/{env_name}_{sub_env}"
    trajectory_dir = os.path.join(log_dir, f"trajectories_{timestamp}")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(trajectory_dir, exist_ok=True)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info("Greedy Agent Pipeline")

    # building envs
    rl4co_cfg = getattr(args.env, "rl4co", {})
    rl4co_env_name = getattr(rl4co_cfg, "env_name", "tsp")
    rl4co_device = getattr(rl4co_cfg, "device", "cpu")
    generator_params = getattr(rl4co_cfg, "generator_params", {
        "num_loc": 20
    })
    rl4co_kwargs = getattr(rl4co_cfg, "rl4co_kwargs", {})

    generator_params = _to_container(generator_params, resolve=True)
    rl4co_kwargs = _to_container(rl4co_kwargs, resolve=True)
    _envs = build_rl4co_routing_envs(
        env_name=rl4co_env_name,
        seed=args.env.seed,
        env_num=args.data.train_batch_size,
        group_n=group_n,
        device=rl4co_device,
        generator_params=generator_params,
        rl4co_kwargs=rl4co_kwargs,
    )
    _val_envs = build_rl4co_routing_envs(
        env_name=rl4co_env_name,
        seed=args.env.seed + 1000,
        env_num=args.data.val_batch_size,
        group_n=1,
        device=rl4co_device,
        generator_params=generator_params,
        rl4co_kwargs=rl4co_kwargs,
    )

    projection_f = partial(
        rl4co_projection,
        env_name=rl4co_env_name,
        one_step=bool(getattr(args, "one_step", False)),
    )
    envs = RL4CORoutingEnvironmentManager(_envs, projection_f, args)
    val_envs = RL4CORoutingEnvironmentManager(_val_envs, projection_f, args)

    agent = RandomAgent(seed=int(args.seed))
    import pdb; pdb.set_trace()
    # Run a single reset + one-step evaluation for quick testing
    observations, infos = envs.reset(kwargs={})
    print("Observations (text) sample:")
    print(observations["text"][0][:400] + "..." if len(observations["text"][0]) > 400 else observations["text"][0])

    actions = []
    batch = observations.get("text", [])
    batch_size = len(batch)
    for i in range(batch_size):
        actions.append(agent.act_for_env(envs, i))

    print("Actions (sample):", actions[:2])
    next_obs, rewards, dones, infos = envs.step(actions)
    print("Rewards:", rewards)
    print("Infos (sample):", infos[:2])

    
    

