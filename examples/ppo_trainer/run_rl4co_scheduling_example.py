import os

from omegaconf import OmegaConf

from verl.trainer.single_controller import run_trainer


def main(env_name: str = "jssp"):
    from verl.trainer.config import ppo_trainer

    cfg = OmegaConf.structured(ppo_trainer)

    # Switch to rl4co scheduling env
    cfg.env.env_name = f"rl4co_scheduling/{env_name}"
    cfg.env.rl4co_scheduling.env_name = env_name
    cfg.env.rl4co_scheduling.device = "cpu"

    # Example generator params (tune as needed)
    if env_name == "jssp":
        cfg.env.rl4co_scheduling.generator_params.num_jobs = 6
        cfg.env.rl4co_scheduling.generator_params.num_machines = 6
    elif env_name == "ffsp":
        cfg.env.rl4co_scheduling.generator_params.num_jobs = 6
        cfg.env.rl4co_scheduling.generator_params.num_stages = 3
        cfg.env.rl4co_scheduling.generator_params.num_machines = [2, 2, 2]

    run_trainer(cfg)


if __name__ == "__main__":
    env = os.environ.get("RL4CO_SCHED_ENV", "jssp")
    main(env)



