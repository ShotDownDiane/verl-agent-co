import sys
import os
import torch

# Add the project root to path so we can import the new modules
sys.path.append("/root/autodl-tmp/verl-agent-co")

from agent_system.environments.env_package.rl4co.route_envs import RouteWorker

def test_route_env(env_name, num_loc=20, generator_params=None):
    print(f"\n{'='*20} Testing {env_name.upper()} {'='*20}")
    try:
        env_kwargs = {}
        if generator_params:
            env_kwargs["generator_params"] = generator_params

        worker = RouteWorker(
            env_name=env_name,
            seed=1234,
            env_num=1,
            device="cpu",
            num_loc=num_loc,
            return_topk_options=5,  # Test with Top-K enabled
            env_kwargs=env_kwargs
        )
        
        print("Resetting environment...")
        obs, infos = worker.reset()
        print(f"Observation (First 500 chars):\n{obs[0][:500]}...")
        
        # Verify action_candidates in td
        # For routing problems, actions are usually node indices.
        # We'll just pick node 0 (depot/start) or node 1.
        # RouteWorker expects a list of actions (one per env)
        
        actions = [0] 
        print(f"Taking step with action: {actions}")
        obs, rewards, dones, infos = worker.step(actions)
        
        while not dones[0]:
            actions = [0] 
            obs, rewards, dones, infos = worker.step(actions)

        print(f"Observation after step (First 500 chars):\n{obs[0][:500]}...")
        print(f"Reward: {rewards[0]}")
        print(f"Done: {dones[0]}")
        # print(f"Info: {infos}") # Info might be large

    except Exception as e:
        print(f"Error testing {env_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test TSP
    test_route_env("tsp")
    
    # Test CVRP
    # test_route_env("cvrp")
    
    # Test OP - fix for Uniform(1.0, 1.0) error
    # test_route_env("op", generator_params={"min_prize": 0.0, "max_prize": 1.0})
