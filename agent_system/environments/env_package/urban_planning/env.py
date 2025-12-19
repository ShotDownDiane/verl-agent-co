import argparse
import os
import logging
import copy


import setproctitle
import warnings
import networkx as nx
import numpy as np
import pandas as pd
import torch
import swanlab as wandb

from .utils.config import Config
from .envs import CityEnv
from .urban import decision_making

warnings.simplefilter(action='ignore', category=FutureWarning)

def tensorfy(np_list, device=torch.device('cpu')):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x).to(device) for x in y] for y in np_list]
    else:
        return [torch.tensor(y).to(device) for y in np_list]

def create_logger(filename, file_handle=True):
    # create logger
    logger = logging.getLogger(filename)
    logger.propagate = False
    logger.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    stream_formatter = logging.Formatter('[%(asctime)s] %(message)s')
    ch.setFormatter(stream_formatter)
    logger.addHandler(ch)

    if file_handle:
        # create file handler which logs even debug messages
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fh = logging.FileHandler(filename, mode='a')
        fh.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter('[%(asctime)s] %(message)s')
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger

class UrbanEnvironment:
    def __init__(self, cfg, root_dir, tmp, agent_type, global_seed, iteration):
        self.root_dir = root_dir
        self.cfg_name = cfg
        self.tmp = tmp
        self.agent_type = agent_type
        self.global_seed = global_seed
        self.iteration = iteration
        setproctitle.setproctitle('urban_planning')
        # Load configuration
        self.cfg = Config(self.cfg_name, self.global_seed, self.tmp, self.root_dir, self.agent_type)
        self.setup_env()
        self.training = True
        self.setup_logger(num_threads=1)
        self.fault_times = 0
        self.service_types = ['outside', 'feasible', 'road', 'boundary', 'residential', 'business', 'office', 'green', 'green', 'school', 'hospital', 'hospital', 'recreation']
        self.service_types_with_size = ['outside', 'feasible', 'road', 'boundary', 'residential', 'business', 'office', 'green_l', 'green_s', 'school', 'hospital_l', 'hospital_s', 'recreation']

    def setup_env(self):
        self.env = CityEnv(self.cfg)
        self.numerical_feature_size = self.env.get_numerical_feature_size()
        self.node_dim = self.env.get_node_dim()
    
    def setup_logger(self, num_threads):
        cfg = self.cfg
        self.logger = create_logger(os.path.join(cfg.log_dir, f'log_{"train"}.txt'),
                                    file_handle=True) 
        self.reward_offset = 0.0
        self.best_rewards = -1000.0
        self.best_plans = []
        self.current_rewards = -1000.0
        self.current_plans = []
        self.save_best_flag = False
        cfg.log(self.logger, None)

        self.thread_loggers = []
        for i in range(num_threads):
            self.thread_loggers.append(
                create_logger(os.path.join(cfg.log_dir, f'log_{"train" if self.training else "eval"}_{i}.txt'),
                              file_handle=True))
    
    def reset(self,seed=None):
        """Reset the environment."""
        self.env.reset(seed=seed)

        # get_observation
        graph, residential_regions, service_regions, feasible_regions = self.get_env_features(self.env._plc)
        _, all_regions = self.get_region_data_text(graph, residential_regions, service_regions, feasible_regions)
        data_text, answer_option_form, land_use_type, available_edges, available_options, available_option_ids = self.get_observation(self.env, self.logger, graph, residential_regions, service_regions, feasible_regions)

        self._current_state = {
            "land_use_type": land_use_type,
            "available_edges": available_edges,
            "available_options": available_options,
            "available_option_ids": available_option_ids,
            "last_residential_region": residential_regions,
            "last_all_regions": all_regions,
            "last_data_text": data_text
        }
        # decision making
        obs = data_text
        info = {"data_text": data_text,"available_actions": available_options}
        return obs, info
    
    def step(self, action=None):
        """Take a step in the environment."""
        # env step
        # parse action
        if action is None:
            action = action.replace("_", " ")
            m = re.search(r'feasible region \d+', action)
            if m != -1:
                action = m.group(0) 

        if action in self._current_state["available_options"]:
            action = self._current_state["available_edges"][self._current_state["available_options"].index(action)]
        else:
            if self.fault_times < 3:
                self.fault_times += 1
                print(f"Invalid action: {action}. Using the first available edge instead.")
                env_feedbacks=f"Because your last action {action} is invalid, the environment will not be updated. Please try again.\n"
                data_text = env_feedbacks+self._current_state['last_data_text']
                info = {
                    "data_text": data_text,
                    "available_actions": self._current_state["available_options"],
                }
                return data_text, -0.1, False, False, info
            else:            
                action = self._current_state["available_edges"][0]
            # 这里需要加一个惩罚，保证答错之后不会一直答错
        self.fault_times = 0
        next_state, reward, done, truncated, info = self.env.step(action, self._current_state['land_use_type'])# land_use_type=land_use_type['type'])
        # get_new_observation
        graph, residential_regions, service_regions, feasible_regions = self.get_env_features(self.env._plc)
        _, all_regions = self.get_region_data_text(graph, residential_regions, service_regions, feasible_regions)
        env_feedbacks = self.get_env_feedback(residential_regions, self._current_state["last_all_regions"], self._current_state["last_residential_region"])

        feedback_text = ""
        for service in self.service_types[5:]:
            service_distance_decreases = []
            for region in env_feedbacks:
                service_distance_decreases.append(env_feedbacks[region][service])
            feedback_text += f"Distance decrease to {service}: {round(np.mean(service_distance_decreases) * 100, 2)}%\n"
    
        # get_observation'
        if self.env._stage == "done":
            print("Urban planning task is completed.")
            data_text= "The urban planning task is completed.\n"
            answer_option_form = "test"
            land_use_type = self._current_state['land_use_type']
            available_edges = []
            available_options = []
            available_option_ids = []

        else:
            data_text, answer_option_form, land_use_type, available_edges, available_options, available_option_ids = self.get_observation(self.env, self.logger, graph, residential_regions, service_regions, feasible_regions)

            self._current_state = {
                "land_use_type": land_use_type,
                "available_edges": available_edges,
                "available_options": available_options,
                "available_option_ids": available_option_ids,
                "last_residential_region": residential_regions,
                "last_all_regions": all_regions,
                "last_data_text": data_text,
                "last_reward": reward
            }

        info = {
            "data_text": data_text,
            "available_actions": available_options,
            "won":0,
            "service": 0,
            "greenness": 0
        }

        done = self.env._stage == "done"
        if done:
            episode_success = (reward != self.env.FAILURE_REWARD) and \
                                (reward != self.env.INTERMEDIATE_REWARD)
            final_rewards, rewards_info = self.env.get_reward_info()
            info["won"] = final_rewards
            info["service"] = rewards_info["life_circle"]
            info["greenness"] = rewards_info["greenness"]

        state = next_state

        final_obs = feedback_text + data_text
        
        return final_obs, reward, done, done, info
    
    def get_env_feedback(self, new_residential_regions, all_regions, residential_regions):
        feedbacks = {}
        for j, residential_region in new_residential_regions.iterrows():
            for service in self.service_types[5:]:
                origin_service_distance = residential_regions[f'distance_to_{service}'][j]
                new_service_distance = new_residential_regions[f'distance_to_{service}'][j]
                if all_regions['service_idx'][j] not in feedbacks:
                    feedbacks[all_regions['service_idx'][j]] = {
                        service: (origin_service_distance - new_service_distance) / origin_service_distance
                    }
                else:
                    feedbacks[all_regions['service_idx'][j]][service] = (
                            (origin_service_distance - new_service_distance) / origin_service_distance)
        return feedbacks
        
    def get_observation(self, env, logger, ir_sub_graph, residential_regions, service_regions, feasible_regions):
        region_text, all_regions = self.get_region_data_text(ir_sub_graph, residential_regions, service_regions, feasible_regions)
        target_edges = []
        land_use_type, _ = env._get_land_use_and_mask()
        land_use = self.service_types[land_use_type['type']]

        # Options
        distance_from_service_to_residential_regions = {}
        for j, res_region in residential_regions.iterrows():
            # data analysis
            analysis_texts = []
            for service_type in self.service_types[5:]:
                try:
                    analysis_texts.append(f"- {service_type}: {round(res_region[f'distance_to_{service_type}'], 2)}m")
                except Exception:
                    import pdb
                    pdb.set_trace()

                if service_type not in distance_from_service_to_residential_regions:
                    distance_from_service_to_residential_regions[service_type] = {
                        j: res_region[f'distance_to_{service_type}']}
                else:
                    distance_from_service_to_residential_regions[service_type][j] = res_region[
                        f'distance_to_{service_type}']
        distance_from_target_service_to_residential_regions = sorted(distance_from_service_to_residential_regions[land_use].items(), key=lambda x: -x[1])
        target_residential_region = distance_from_target_service_to_residential_regions[0][0]

        # calculate distance to feasible regions
        gdf, whole_graph = env._plc._get_current_gdf_and_graph()
        # Get node dict
        node_graph2gdf_dict = {node: gdf.index[i] for i, node in enumerate(whole_graph.nodes)}
        node_gdf2node_dict = {gdf_index: node for node, gdf_index in node_graph2gdf_dict.items()}

        feasible_regions = gdf[gdf['type'] == 1]
        distance_to_feasible_regions = []
        for j, fea_region in feasible_regions.iterrows():
            try:
                distance_to_feasible_regions.append([j, nx.shortest_path_length(ir_sub_graph, target_residential_region, j, weight='weight')])
            except nx.NetworkXNoPath:
                distance_to_feasible_regions.append([j, np.inf])

        samples = sorted(distance_to_feasible_regions, key=lambda x: x[1])

        options = [all_regions.service_idx[sample[0]] for sample in samples]
        option_ids = [sample[0] for sample in samples]

        # Environment feedback
        _target_edges = []

        for region in option_ids:
            region_neighbors = whole_graph.neighbors(node_gdf2node_dict[region])
            for neighbor in region_neighbors:
                if gdf['type'][node_graph2gdf_dict[neighbor]] == 13:
                    # graph_region = node_gdf2graph_dict[region]
                    edge = np.array((region, node_graph2gdf_dict[neighbor]))
                    edge_reverse = np.array((node_graph2gdf_dict[neighbor], region))
                    all_edges = np.array(whole_graph.edges)
                    current_graph_nodes_id = gdf.index.to_numpy()
                    all_edges = current_graph_nodes_id[all_edges]
                    edge_idx = np.where((all_edges == edge).all(axis=1))[0]
                    if len(edge_idx) == 0:
                        edge_idx = np.where((all_edges == edge_reverse).all(axis=1))[0]
                    _target_edges.append(edge_idx)
                    break
        
        available_edges = []
        available_options = []
        available_option_ids = []
        for k, edge_idx in enumerate(_target_edges):
            simulation = copy.deepcopy(env)
            try:
                _ = simulation.step(torch.tensor(edge_idx).long(), land_use_type=land_use_type['type'])
            except Exception as e:
                continue
            target_edges.append(edge_idx)
            available_edges.append(edge_idx)
            available_options.append(options[k])
            available_option_ids.append(option_ids[k])

        # get the options
        option_text = "[" + ", ".join([f"{all_regions['service_idx'][available_option_ids[i]]}" for i in range(len(available_option_ids))]) + "]"
        data_text = (f"{region_text}\n\n"
                     f"Feasible regions to plan:\n"
                     f"{option_text}\n\n"
                     f"The next service to be built:\n"
                     f"{land_use} (with {round(land_use_type['area'] / 1000000, 4)}km²)")
        answer_option_form = "\"" + "/".join([all_regions['service_idx'][available_option_ids[i]] for i in range(len(available_option_ids))]) + "\""

        return data_text, answer_option_form, land_use_type['type'], available_edges, available_options, available_option_ids
    
    def get_region_data_text(self, graph, residential_regions, service_regions, feasible_regions):
        # index regions
        region_idx_dict = {}
        all_regions = pd.concat([residential_regions, service_regions, feasible_regions])
        for i, region in all_regions.iterrows():
            service_type = region['type']
            mapped_type = self.service_types[service_type]  # 获取映射后的类型

            if mapped_type not in region_idx_dict:
                region_idx_dict[mapped_type] = 0
            else:
                region_idx_dict[mapped_type] += 1

            # 生成索引字符串
            idx = region_idx_dict[mapped_type]
            service_idx_str = f"{mapped_type} region {idx}"
            if i in residential_regions.index:
                residential_regions.at[i, "service_idx"] = service_idx_str
            elif i in service_regions.index:
                service_regions.at[i, "service_idx"] = service_idx_str
            else:
                feasible_regions.at[i, "service_idx"] = service_idx_str
            all_regions.at[i, "service_idx"] = service_idx_str

        # observation preparation
        residential_region_texts = []
        for i, res_region in residential_regions.iterrows():
            area = residential_regions.area[i]
            residential_region_texts.append(f"{residential_regions.service_idx[i]}:\n"
                                            f"- area: {round(area/1000000, 4)}km²")

        feasible_region_texts = []
        for i, feasible_region in feasible_regions.iterrows():
            area = feasible_regions.area[i]
            feasible_region_texts.append(f"{feasible_regions.service_idx[i]}:\n"
                                         f"- area: {round(area/1000000, 4)}km²")

        service_region_texts = []
        for i, service_region in service_regions.iterrows():
            area = service_regions.area[i]
            service_region_texts.append(f"{service_regions.service_idx[i]}:\n"
                                        f"- area: {round(area/1000000, 4)}km²")

        edges_str = [f"({all_regions.service_idx[u]}, {all_regions.service_idx[v]}, {round(d['weight'], 2)}m)" for u, v, d in graph.edges(data=True)]
        adj_info_text = f"Connectivity of regions:\n[{', '.join(edges_str)}]"
        region_text = "\n\n".join(feasible_region_texts + residential_region_texts + service_region_texts + [adj_info_text])

        return region_text, all_regions
    
    def get_env_features(self,placemen_client):
        graph, entity_dict = placemen_client.get_region_graph()
        centroids = entity_dict.geometry.centroid
        residential_regions = entity_dict[entity_dict['type'] == 4].copy()
        service_regions = entity_dict[entity_dict['type'].isin(list(range(5, 13)))].copy()
        feasible_regions = entity_dict[entity_dict['type'] == 1].copy()

        # QA: Topology
        # calculate distance to service

        for i, res_region in residential_regions.iterrows():
            service_distance_dict = {}
            service_distance_dict_center = {}
            for j, service in service_regions.iterrows():
                if service['type'] not in service_distance_dict:
                    service_distance_dict[service['type']] = [nx.shortest_path_length(graph, i, j, weight='weight')]
                    service_distance_dict_center[service['type']] = [centroids[i].distance(centroids[j])]
                else:
                    service_distance_dict[service['type']].append(nx.shortest_path_length(graph, i, j, weight='weight'))
                    service_distance_dict_center[service['type']].append(centroids[i].distance(centroids[j]))

            # find min distance
            for service_id in range(5, 13):
                if service_id not in service_distance_dict:
                    residential_regions.loc[i, f'distance_to_{self.service_types[service_id]}'] = float('inf')
                    residential_regions.loc[i, f'distance_to_{self.service_types[service_id]}_center'] = float('inf')
                else:
                    min_distance = min(service_distance_dict[service_id])
                    residential_regions.loc[i, f'distance_to_{self.service_types[service_id]}'] = min_distance
                    min_distance_center = min(service_distance_dict_center[service_id])
                    residential_regions.loc[i, f'distance_to_{self.service_types[service_id]}_center'] = min_distance_center

        # find the service farther than 1000m
        for i, res_region in residential_regions.iterrows():
            distant_services = []
            for s in range(5, 13):
                if f'distance_to_{self.service_types[s]}' in res_region and res_region[f'distance_to_{self.service_types[s]}'] > 1000:
                    distant_services.append(self.service_types[s])
            residential_regions.loc[i, 'distant_services'] = json.dumps(distant_services)
        

        # QA: distance/duration
        for i, res_region in residential_regions.iterrows():
            # find 1-hop neighbors
            neighbors = list(graph.neighbors(i))

            # calculate neighbors' distance
            neighbors_distance = {}
            for neighbor in neighbors:
                neighbors_distance[str(neighbor)] = centroids[i].distance(centroids[neighbor])

            residential_regions.loc[i, 'neighbors_distance'] = json.dumps(neighbors_distance)

        return graph, residential_regions, service_regions, feasible_regions
    
    def render(self, mode='human'):
        pass
    
    def close(self):
        """Close the environment."""
        pass




import ray
@ray.remote(num_cpus=0.25)
class UrbanWorker:
    def __init__(self, seed, env_kwargs):
        np.random.seed(seed)
        self.env = UrbanEnvironment(**env_kwargs)

    def reset(self, idx=None):
        obs, info = self.env.reset(seed=idx)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        return self.env.render()

    def close(self):
        try:
            self.env.close()
        except:
            pass


from gymnasium import Env
import numpy as np
class UrbanMultiProcEnv(Env):
    def __init__(self, seed=0, env_num=1, group_n=1, env_kwargs=None):
        super().__init__()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.env_kwargs = env_kwargs or {}
        self.env_num = env_num
        self.group_n = group_n
        self.num_workers = env_num * group_n
        self.workers = [
            UrbanWorker.remote(seed + i//group_n, self.env_kwargs)
            for i in range(self.num_workers)
        ]
        self._rng = np.random.RandomState(seed)

    def reset(self):
        idxs = self._rng.randint(0, 10000, size=self.num_workers)
        futures = [w.reset.remote(i) for w, i in zip(self.workers, idxs)]
        results = ray.get(futures)
        obs_list, info_list = zip(*results)
        return list(obs_list), list(info_list)

    def step(self, actions):
        futures = [w.step.remote(a) for w, a in zip(self.workers, actions)]
        results = ray.get(futures)
        obs, rews, terms, truns, infos = zip(*results)
        return list(obs), list(rews), list(terms), list(truns), list(infos)

    def render(self, mode="human", env_idx=None):
        if env_idx is not None:
            return ray.get(self.workers[env_idx].render.remote(mode))
        return ray.get([w.render.remote(mode) for w in self.workers])

    def close(self):
        for w in self.workers:
            w.close.remote()
            ray.kill(w)

def build_urban_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""

    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(root_dir, "records")

    cfg = "hlg"
    agent_type = "rule-centralized"
    global_seed = seed
    iteration = "0"

    env_kwargs = {
        "cfg": cfg,
        "root_dir": root_dir,
        "tmp": False,
        "agent_type": agent_type,
        "global_seed": global_seed,
        "iteration": iteration,
    }
    
    return UrbanMultiProcEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        env_kwargs=env_kwargs,
    )

import re
import json
from typing import List, Tuple

def urban_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

    return actions, valids