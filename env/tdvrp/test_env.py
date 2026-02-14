import torch
from .env import TDVRPEnv
from .generator import TDVRPGenerator

def test_tdvrptw_env():
    print("Testing TDVRPTWEnv...")
    
    # Initialize generator and env
    generator = TDVRPGenerator(
        instance_path="/root/autodl-tmp/vrptdt-benchmark/instances/berlin_10.json",
        matrix_path="/root/autodl-tmp/vrptdt-benchmark/instances/berlin_10_tt.json.bz2",
        num_matrix_steps=37,
        num_nodes=10
    )
    env = TDVRPEnv(generator=generator)
    
    # Reset
    td = env.reset(batch_size=[2])
    print(f"Reset successful. Batch size: {td.batch_size}")
    print(f"Initial action mask: {td['action_mask']}")
    
    # Take some steps
    # Action 1 (Customer 1)
    td.set("action", torch.tensor([1, 1], dtype=torch.long))
    td = env.step(td)["next"]
    print(f"Step 1 (to customer 1) successful.")
    print(f"Current time: {td['current_time'].squeeze().tolist()}")
    print(f"Visited: {td['visited']}")
    print(f"Action mask: {td['action_mask']}")
    
    # Visit all customers
    # The generator has 10 customers (1-10)
    for i in range(12): # Take more steps if needed
        # Pick an available action from mask
        mask = td["action_mask"]
        actions = []
        for b in range(2):
            available = torch.where(mask[b])[0]
            # Prefer visiting customers (non-zero nodes) if available
            customers = available[available != 0]
            if len(customers) > 0:
                # Earliest Deadline First (EDF)
                customer_tws = td["time_windows"][b, customers, 1]
                best_idx = torch.argmin(customer_tws)
                actions.append(customers[best_idx].item())
            else:
                actions.append(0)
        
        td.set("action", torch.tensor(actions, dtype=torch.long))
        td = env.step(td)["next"]
        print(f"Step {i+2} to customers {actions} successful. Time: {td['current_time'].squeeze().tolist()}")
        print(f"Visited sum: {td['visited'].sum(dim=-1).tolist()}")
        
        # Check reward before returning to depot (if we just visited the last customer)
        if (td["visited"][:, 1:]).all() and not td["done"].any():
            print(f"All customers visited. Current Node: {td['current_node'].squeeze().tolist()}")
            reward_before_return = env.get_reward(td, actions=None)
            print(f"Reward before explicit return action (includes predicted return cost): {reward_before_return}")
            
        if td["done"].all():
            print(f"Environment finished at step {i+2}")
            break

    # Finally return to depot (if not already there)
    if not td["done"].all():
        print("Returning to depot...")
        td.set("action", torch.tensor([0, 0], dtype=torch.long))
        td = env.step(td)["next"]
    
    print(f"Final state reached.")
    print(f"Done: {td['done']}")
    print(f"Final Step Reward: {td['reward']}")
    reward_after_return = env.get_reward(td, actions=None)
    print(f"Total Reward (Negative Cost): {reward_after_return}")
    
    # If we captured reward_before_return, compare it
    try:
        diff = torch.abs(reward_before_return - reward_after_return)
        print(f"Reward difference (predicted vs actual return): {diff}")
        assert (diff < 1e-3).all(), f"Reward before and after return action should be consistent. Diff: {diff}"
    except NameError:
        print("Note: reward_before_return was not captured because loop finished differently.")
    
    assert td["done"].all(), "Environment should be done after visiting all customers and returning to depot"
    print("Test passed!")

if __name__ == "__main__":
    test_tdvrptw_env()
