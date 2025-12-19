from stable_baselines3 import DQN, PPO
from stable_baselines3.common.monitor import Monitor
import pickle
import online_env_tensor as ev
import os

"""
Generate a charging plan based on the model.
"""
# Instantiate the env
location = "Qiaonan"  # take a location of your choice
graph_file = "Data/Graph/" + location + "/" + location + ".graphml"
node_file = "Data/Graph/" + location + "/nodes_extended_" + location + ".txt"
plan_file = "Data/Graph/" + location + "/existingplan_" + location + ".pkl"

env = ev.StationPlacement(graph_file, node_file, plan_file)
log_dir = "tmp_Toy_Example/"

"""
Ab hier kommt die Evaluation.
"""
print("Evaluation for best model")
env = Monitor(env, log_dir)  # new environment for evaluation

model = PPO.load(log_dir + "best_model_" + location + "_33500.zip")

obs,_ = env.reset()
done = False
best_plan, best_node_list = None, None
while not done:
    # import pdb
    # pdb.set_trace()  # Debugging line to inspect variables if needed
    action, _states = model.predict(obs)
    print(f"Action taken: {action}")  # Debugging line to see the action taken
    obs, reward, done,_, info = env.step(action)
    env.render()
    if done:
        best_node_list, best_plan = env.render()

os.makedirs("Results/" + location, exist_ok=True)  # Ensure the results directory exists
pickle.dump(best_plan, open("Results/" + location + "/plan_RL.pkl", "wb"))
with open("Results/" + location + "/nodes_RL.txt", 'w') as file:
    file.write(str(best_node_list))