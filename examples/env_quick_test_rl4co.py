import argparse
import random

from omegaconf import OmegaConf
from tensordict.tensordict import TensorDict

from agent_system.environments.env_manager import make_envs


def build_min_config(
    env_name: str,
    rl4co_env: str | None = None,
    rl4co_sched_env: str | None = None,
    train_batch_size: int = 2,
    val_batch_size: int = 2,
):
    """Build a minimal OmegaConf config compatible with make_envs."""
    # default generator params for routing
    if rl4co_env == "tsp":
        routing_gen = {"num_loc": 20, "min_loc": 0.0, "max_loc": 1.0}
    elif rl4co_env == "cvrp":
        routing_gen = {"num_loc": 20, "min_loc": 0.0, "max_loc": 1.0}
    elif rl4co_env == "op":
        # ensure prize sampler has low < high
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
                "env_name": rl4co_env or "tsp",
                "device": "cpu",
                "generator_params": routing_gen,
                "rl4co_kwargs": {},
                "one_step": False,
            },
            "rl4co_scheduling": {
                "env_name": rl4co_sched_env or "jssp",
                "device": "cpu",
                "generator_params": {},
                "rl4co_kwargs": {},
                "one_step": False,
            },
        },
        "data": {
            "train_batch_size": train_batch_size,
            "val_batch_size": val_batch_size,
        },
    }
    return OmegaConf.create(cfg_dict)


def run_quick_rollout(env_name: str, steps: int = 5):
    """Create envs via make_envs and run a short rollout with random integer actions."""
    if env_name.startswith("rl4co/"):
        rl4co_env = env_name.split("/", 1)[1]
        cfg = build_min_config(env_name=f"rl4co/{rl4co_env}", rl4co_env=rl4co_env)
    elif env_name.startswith("rl4co_scheduling/"):
        sched_env = env_name.split("/", 1)[1]
        cfg = build_min_config(
            env_name=f"rl4co_scheduling/{sched_env}", rl4co_sched_env=sched_env
        )
    else:
        raise ValueError(f"Unsupported env_name for quick test: {env_name}")

    envs, _ = make_envs(cfg)

    print(f"[quick-test] env_name={env_name}")
    obs, infos = envs.reset(kwargs={})
    print("Initial text obs[0] (truncated):")
    print(obs["text"][0][:800], "...\n")

    for t in range(steps):
        print(f"--- step {t} ---")
        batch_size = len(obs["text"])
        # Random integer actions; keep them within valid range when possible
        if env_name.startswith("rl4co_scheduling/") and isinstance(obs["anchor"], TensorDict):
            td = obs["anchor"]
            # choose a valid action index from action_mask for each env
            mask = td["action_mask"]  # shape [bs, n_actions]
            mask_np = mask.cpu().numpy()
            text_actions = []
            for i in range(batch_size):
                valid_idx = [j for j, v in enumerate(mask_np[i]) if v]
                if not valid_idx:
                    # fallback if no valid action (should be rare)
                    text_actions.append("0")
                else:
                    text_actions.append(str(random.choice(valid_idx)))
        else:
            text_actions = [str(random.randint(0, 10)) for _ in range(batch_size)]
        obs, rewards, dones, infos = envs.step(text_actions)
        print("text[0] (truncated):")
        print(obs["text"][0][:400], "...")
        print("rewards:", rewards)
        print("dones:", dones)
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Quick text-wrapper smoke test for rl4co routing & scheduling envs."
    )
    parser.add_argument(
        "--env",
        type=str,
        default="rl4co/tsp",
        help="Env name: rl4co/tsp, rl4co/cvrp, rl4co/op, rl4co_scheduling/jssp, rl4co_scheduling/ffsp",
    )
    parser.add_argument("--steps", type=int, default=5, help="Number of steps to simulate")
    args = parser.parse_args()

    run_quick_rollout(args.env, steps=args.steps)

if __name__ == "__main__":
    main()


