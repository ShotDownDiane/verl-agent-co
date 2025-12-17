import argparse
import json
import random

from omegaconf import OmegaConf

from agent_system.environments.env_manager import make_envs


def build_min_config(
    env_name: str,
    train_batch_size: int = 2,
    val_batch_size: int = 2,
):
    """Build a minimal OmegaConf config compatible with make_envs for ML4CO-Kit."""
    sub_env = env_name.split("/", 1)[1] if "/" in env_name else env_name
    
    # Default generator params for routing
    if sub_env == "tsp":
        routing_gen = {"num_loc": 20, "min_loc": 0.0, "max_loc": 1.0}
    elif sub_env == "cvrp":
        routing_gen = {"num_loc": 20, "min_loc": 0.0, "max_loc": 1.0}
    elif sub_env == "op":
        routing_gen = {
            "num_loc": 20,
            "min_loc": 0.0,
            "max_loc": 1.0,
            "min_prize": 1.0,
            "max_prize": 2.0,
        }
    else:
        routing_gen = {"num_loc": 20, "min_loc": 0.0, "max_loc": 1.0}

    cfg_dict = {
        "env": {
            "env_name": env_name,
            "seed": 0,
            "max_steps": 50,
            "history_length": 2,
            "rollout": {"n": 1},
            "resources_per_worker": {"num_cpus": 0.1, "num_gpus": 0},
            # sub-configs for other env branches (unused here but required)
            "alfworld": {"eval_dataset": "eval_in_distribution"},
            "search": {
                "log_requests": False,
                "search_url": "",
                "topk": 3,
                "timeout": 60,
            },
            "sokoban": {
                "dim_room": [6, 6],
                "num_boxes": 1,
                "search_depth": 30,
                "mode": "tiny_rgb_array",
            },
            "webshop": {"use_small": True, "human_goals": False},
            "appworld": {},
            "rl4co": {
                "env_name": "tsp",
                "device": "cpu",
                "generator_params": routing_gen,
                "rl4co_kwargs": {},
                "one_step": False,
            },
            "rl4co_scheduling": {
                "env_name": "jssp",
                "device": "cpu",
                "generator_params": {},
                "rl4co_kwargs": {},
                "one_step": False,
            },
            "ml4co_kit": {
                "env_name": sub_env,
                "device": "cpu",
                "generator_params": routing_gen if sub_env in ("tsp", "cvrp", "op") else {},
                "rl4co_kwargs": {},
                "k_nn": 2,
                "k": 2,
                "use_format_reward": True,
                "format_reward_weight": 0.05,
                "feasibility_reward_weight": 0.15,
                "use_conditional_reward": True,
                "feasibility_threshold": 0.9,
                "normalize_env_reward": True,
                "env_reward_range": [-20.0, 0.0],
                "fixed_scale_reference": 8.0,
            },
        },
        "data": {
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
        },
    }
    return OmegaConf.create(cfg_dict)


def run_quick_rollout(env_name: str):
    """Create ML4CO-Kit envs and test one-shot mode with a dummy solution."""
    if not env_name.startswith("ml4co-kit/"):
        raise ValueError(f"Expected env_name starting with 'ml4co-kit/', got: {env_name}")
    
    cfg = build_min_config(env_name=env_name)
    envs, _ = make_envs(cfg)

    print(f"[quick-test one-shot] env_name={env_name}")
    obs, infos = envs.reset(kwargs={})
    print("Initial text obs[0] (truncated):")
    print(obs["text"][0][:800], "...\n")

    batch_size = len(obs["text"])
    sub_env = env_name.split("/", 1)[1]

    # Construct dummy full solutions
    if sub_env in ("tsp", "op"):
        # Simple permutation route
        base_route = list(range(20))
        solutions = []
        for _ in range(batch_size):
            r = base_route[:]
            random.shuffle(r)
            solutions.append({"route": r})
        text_actions = [json.dumps(sol) for sol in solutions]
    elif sub_env == "cvrp":
        # Simple route with depot markers
        base_route = list(range(1, 20))  # customers only
        solutions = []
        for _ in range(batch_size):
            r = base_route[:]
            random.shuffle(r)
            # Split into 2 routes
            mid = len(r) // 2
            routes = [[0] + r[:mid] + [0], [0] + r[mid:] + [0]]
            solutions.append({"routes": routes})
        text_actions = [json.dumps(sol) for sol in solutions]
    elif sub_env in ("jssp", "pfsp"):
        # Random job sequence schedule
        num_jobs = 6
        solutions = []
        for _ in range(batch_size):
            seq = list(range(num_jobs))
            random.shuffle(seq)
            solutions.append({"schedule": seq})
        text_actions = [json.dumps(sol) for sol in solutions]
    else:
        raise ValueError(f"Unsupported ml4co-kit env: {sub_env}")

    print("=" * 80)
    print("[One-shot mode] Providing complete solution.")
    print(f"Solution format: {text_actions[0][:100]}...")
    print("=" * 80)

    obs, rewards, dones, infos = envs.step(text_actions)
    
    print("\nResults:")
    print("=" * 80)
    print(f"Final Rewards: {rewards}")
    print(f"Dones: {dones}")
    print("-" * 80)
    
    if infos and "format_bonus" in infos[0]:
        print("Reward Breakdown:")
        print("-" * 80)
        for i, info in enumerate(infos):
            print(f"Environment {i}:")
            print(f"  Format Bonus: {info.get('format_bonus', 0):.4f}")
            print(f"  Feasibility Bonus: {info.get('feasibility_bonus', 0):.4f}")
            print(f"  Environment Reward (original): {info.get('env_reward', 0):.4f}")
            if "format_reward" in info:
                print(f"  Format Reward: {info.get('format_reward', 0):.4f}")
                print(f"  Feasibility Reward: {info.get('feasibility_reward', 0):.4f}")
                print(f"  Scaled Env Reward: {info.get('scaled_env_reward', 0):.4f}")
                print(f"  Env Reward Weight: {info.get('env_reward_weight', 0):.4f}")
                print(f"  Meets Feasibility Threshold: {info.get('meets_feasibility_threshold', False)}")
            print(f"  └─ Final Reward: {rewards[i]:.4f}")
            print()
    
    print("Solution:")
    print("-" * 80)
    if sub_env in ("tsp", "op"):
        print(f"Route: {infos[0].get('route', 'N/A')}")
    elif sub_env == "cvrp":
        print(f"Routes: {infos[0].get('routes', 'N/A')}")
    elif sub_env in ("jssp", "pfsp"):
        print(f"Schedule: {infos[0].get('schedule', 'N/A')}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick one-shot smoke test for ml4co-kit routing & scheduling envs."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="ml4co-kit/tsp",
        help="Env name: ml4co-kit/tsp, ml4co-kit/cvrp, ml4co-kit/op, ml4co-kit/jssp, ml4co-kit/pfsp",
    )
    args = parser.parse_args()

    run_quick_rollout(args.env)


if __name__ == "__main__":
    main()

