
import torch
import sys
import os

# Add the project root to path so we can import the new modules
sys.path.append("/root/autodl-tmp/verl-agent-co")

from agent_system.environments.env_package.rl4co.graph_env import GraphWorker

def test_graph_env(env_name, **kwargs):
    print(f"\n{'='*20} Testing {env_name.upper()} {'='*20}")
    try:
        worker = GraphWorker(
            env_name=env_name,
            seed=1234,
            env_num=1,  # Reduced to 1 to keep output clean
            device="cpu",
            return_topk_options=10
        )
        
        print("Resetting environment...")
        obs, infos = worker.reset()
        print(f"Observation (First 500 chars):\n{obs[0]}")
        
        # Verify action_candidates in td
        actions=[0]
        obs, rewards, dones, infos = worker.step(actions)
        print(f"Observation (First 500 chars):\n{obs[0]}")
        print(f"Reward: {rewards[0]}")
        print(f"Dones: {dones[0]}")
        print(f"Info: {infos}")
        

    except Exception as e:
        print(f"Error testing {env_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test FLP
    test_graph_env("flp")
    
    # Test MCLP
    test_graph_env("mclp")
    
    # Test STP
    test_graph_env("stp")

    # Test build_graph_env wrapper
    print(f"\n{'='*20} Testing build_graph_env Wrapper {'='*20}")
    from agent_system.environments.env_package.rl4co.graph_env import build_graph_env
    try:
        envs = build_graph_env("mclp", env_num=2, group_n=2)
        print("Successfully built GraphEnvs with build_graph_env")
    except Exception as e:
        print(f"Error building GraphEnvs: {e}")
        import traceback
        traceback.print_exc()
