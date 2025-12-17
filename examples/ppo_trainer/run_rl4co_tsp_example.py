import os

from omegaconf import OmegaConf

from verl.trainer.single_controller import run_trainer


def main():
    # Load base PPO trainer config
    from verl.trainer.config import ppo_trainer

    cfg = OmegaConf.structured(ppo_trainer)

    # Minimal overrides to demonstrate RL4CO TSP env usage
    cfg.env.env_name = "rl4co/tsp"
    cfg.env.rl4co.env_name = "tsp"
    cfg.env.rl4co.device = "cpu"

    # Optionally tweak TSP size here:
    # cfg.env.rl4co.generator_params.num_loc = 20

    run_trainer(cfg)


if __name__ == "__main__":
    main()



