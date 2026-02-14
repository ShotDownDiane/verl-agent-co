import torch
from rl4co.envs.routing.tdtsp.env import TDTSPMatrixEnv
from rl4co.envs.routing.tdtsp.env_tw import TDTSPTWGenerator
from agent_system.environments.prompts import *

from functools import partial
from projection import co_projection_selected as co_projection
from route_envs import RouteWorker
import os
import re
import numpy as np
import swanlab
import time
from PIL import Image
from tqdm import trange
from examples.prompt_agent.llm_agent import LLMAgent
from examples.prompt_agent.vlm_agent import VLMAgent

class MockAgent:
    """A mock agent that always picks Option A for testing purposes."""
    def batch_generate(self, system_prompts, texts, images=None, **kwargs):
        # Always pick Option A
        return ["<OBS>xxxx</OBS><Thought> Picking Option A to test the environment loop. </Thought>\n<Decision> \\boxed{A} </Decision>" for _ in range(len(texts))] 

class WorkerManager:
    SYSTEM_TEMPLATEs ={
        "tsp": RL4CO_TSP_SYSTEM_TEMPLATE,
        "cvrp": RL4CO_CVRP_SYSTEM_TEMPLATE,
        "flp": RL4CO_FLP_SYSTEM_TEMPLATE,
        "mclp": RL4CO_MCLP_SYSTEM_TEMPLATE,
        "stp": RL4CO_STP_SYSTEM_TEMPLATE,
        "tdtsp": RL4CO_TDTSP_SYSTEM_TEMPLATE,
        "tdtsp_tw": RL4CO_TDTSP_TW_SYSTEM_TEMPLATE,
        "tdvrp": RL4CO_TDVRP_SYSTEM_TEMPLATE,
    } 
    USER_TEMPLATEs ={
        "tsp": RL4CO_TSP_USER_TEMPLATE,
        "cvrp": RL4CO_CVRP_USER_TEMPLATE,
        "flp": RL4CO_FLP_USER_TEMPLATE,
        "mclp": RL4CO_MCLP_USER_TEMPLATE,
        "stp": RL4CO_STP_USER_TEMPLATE,
        "tdtsp": RL4CO_TDTSP_USER_TEMPLATE,
        "tdtsp_tw": RL4CO_TDTSP_TW_USER_TEMPLATE,
        "tdvrp": RL4CO_TDVRP_USER_TEMPLATE,
    }
    def __init__(self, env_name):
        self.env_name = env_name
        self.system_template = self.SYSTEM_TEMPLATEs[env_name]
        self.user_template = self.USER_TEMPLATEs[env_name]
        self.action_projection = partial(co_projection, env_name=env_name)

    def process_text_obs(self, next_obs):
        postprocess_text_obs = []
        obs_str = [obs['obs'] for obs in next_obs]
        candidates_str = [obs['candidates'] for obs in next_obs]
        for i in range(len(next_obs)):
            obs = self.user_template.format(obs_text=obs_str[i], candidates=candidates_str[i])
            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def process_action(self, action_text):
        parsed_actions, valids = self.action_projection(action_text)
        return parsed_actions, valids
    
    def process_image(self, img) -> np.ndarray:
        # if isinstance(img, np.ndarray):
        #     return img
        # assert isinstance(img, str), "img must be a path string or a numpy array"
        
        # image = Image.open(img)
        # if image.mode != 'RGB':
        #     image = image.convert('RGB')        
        # return np.array(image)
        return img
    
    def process_step(self, next_obs):
        text_obs = self.process_text_obs(next_obs)
        img_obs = [self.process_image(obs['image']) for obs in next_obs]
        next_observations = {
            "texts": text_obs, 
            "images": img_obs,
            "system_prompts": [self.system_template for _ in range(len(text_obs))]
        }
        return next_observations


def test_tdvrp_greedy_full():
    print("\n" + "#"*60)
    print("Starting Full Greedy (Option A) Test for TDVRP (Cost & Hard TW)...")
    print("#"*60)
    
    # 1. Setup paths
    cities = ["berlin"]
    node_configs = [20, 50]
    city = cities[0]
    num_nodes = node_configs[1]
    test_data_path = f"/root/autodl-tmp/tdtsp_dataset_random/{city}_{num_nodes}_random_test.npz"
    base_data_path = "/root/autodl-tmp/vrptdt-benchmark/instances"
    matrix_path = "/root/autodl-tmp/vrptdt-benchmark/instances" 

    swanlab.init(
        project="TDVRP-Agent-Test",
        experiment_name="vlm-agent-test",
        config={
            "cities": cities,
            "node_configs": node_configs,
        }
    )

    # 2. Setup Worker for tdvrp
    all_rewards = []
    all_lates = []
    start_time = time.time()
    num_sample = 500
    batch_size = 25

    worker = RouteWorker(
        env_name="tdvrp",
        env_num=batch_size,
        device="cpu",
        env_kwargs={
            "data_path": test_data_path,
            "base_data_path": base_data_path,
            "matrix_path": matrix_path,
            "random_sample": False,
            "synchronous": False
        },
        return_topk_options=26,
        image_obs="path"
    )
    
    # agent = MockAgent() # debug
    # real agent
    api_base_url = "http://localhost:8000/v1"
    api_key = "token-abc123456"
    agent = VLMAgent(
        api_key=api_key,
        api_base_url=api_base_url,
        model_name="vlm"
    )

    manager = WorkerManager(env_name="tdvrp")

    for current_idx in trange(0, num_sample-1, batch_size, desc="Full Trajectory Loop"):
        
        # 3. Full Trajectory Loop
        obs_list, infos = worker.reset()
        
        done = False
        step_count = 0

        while not done:
            step_count += 1
            next_observations = manager.process_step(obs_list)

            raw_actions = agent.batch_generate(**next_observations)
            actions, valids = manager.process_action(raw_actions)
            
            obs_list, rewards, dones, infos = worker.step(actions)
            done = all(dones)
        
        for info in infos:
            final_cum_reward = info.get('cumulative_reward', None)
            all_rewards.append(final_cum_reward)
            violations = info.get('total_penalty', 0)
            all_lates.append(1 if violations > 0 else 0)
            
            # Log individual instance results
            swanlab.log({
                f"{num_nodes}nodes/{city}/individual_reward": final_cum_reward,
                f"{num_nodes}nodes/{city}/individual_violations": violations,
            })

    avg_cost = -np.mean(all_rewards)
    elapsed_time = time.time() - start_time
    swanlab.log({
        f"{num_nodes}nodes/{city}/avg_cost": avg_cost,
        f"{num_nodes}nodes/{city}/execution_time": elapsed_time,
    })

if __name__ == "__main__":
    test_tdvrp_greedy_full()    
