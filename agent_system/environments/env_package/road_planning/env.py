from .env_utils import RoadEnv
from .env_utils.utils import Config
import copy
from .urban import decision_making
from .env_utils.utils import *

class RoadPlanningEnvironment:
    """
    Simulation environment for planning road connections.
    """

    def __init__(self, cfg: Config):
        """
        Initialize the simulation environment.

        :param cfg: Configuration dictionary for the environment.
        """
        self.cfg = cfg
        self.env = RoadEnv(cfg)
        self.connectivity = None
        self.NR = -1
        self.game_over = False
        self.step_count = 0
        self.stage = 1
        self.available_roads = []
        self.fault_times = 0


    def _create_answer_option_forms(self) -> list:
        """
        Create answer option forms based on available roads.

        :return: List of formatted answer options.
        """
        answer_option_form = ["\"" + '/'.join([f"road {road_id}" for road_id in self.available_roads]) + "\""]
        return answer_option_form

    def reset(self,seed=None):
        """
        Reset the environment to its initial state.
        """
        self.env.reset(seed=seed)
        self.game_over = False
        self.step_count = 0
        obs = self.get_observation()
        text_obs = self.get_data_texts()
        self.last_observations = obs
        self.available_roads = [str(road_data['road_id']) for road_data in obs['available_roads']]
        return text_obs, {"available_actions": self.available_roads, "text_obs": text_obs}

    def step(self, action) -> dict:
        """
        Take an environment step based on the decision and provide feedback.

        :param last_observations: Observations before the step.
        :param decision: Decision string provided by the agent.
        :return: Environment feedback as a dictionary.
        """
        last_available_roads = [str(road_data['road_id']) for road_data in self.last_observations['available_roads']]
        last_unconnected_regions = len(self.last_observations.get('unconnected_regions', []))
        last_avgdist = self.last_observations['avgdist']
        last_stage = self.stage

        # Parse decision to get the selected road id; default to first available if parsing fails
        selected_road_id = self._parse_decision(action)
        if selected_road_id not in last_available_roads:
            if self.fault_times < 1:
                self.fault_times += 1
                data_obs = self.get_data_texts()
                env_feedback = f"Because your decision {action} is invalid, The environment does not change. Please try again.\n{data_obs}"
                last_available_roads = self.last_observations['available_roads']
                last_available_roads = [road_data['road_id'] for road_data in last_available_roads]
                NR, AD, SC = self.get_metrics()
                info = {
                    "last_observations": self.last_observations,
                    "available_actions": [str(road_data) for road_data in last_available_roads],
                    "num_roads": NR,
                    "avg_dist": AD,
                    "cost": SC
                } 
                return env_feedback, -0.1, False, False, info  # -0.1 penalty for invalid action, False for done and truncated
            else:
                selected_road_id = self.available_roads[0]
        
        self.fault_times = 0

        next_state, reward, done, info = self.env.step(selected_road_id)
        self.check_stage()
        next_observations = self.get_observation(done)
        if done:
            self.game_over = True

        next_available_roads = [road_data['road_id'] for road_data in next_observations['available_roads']]
        new_available_roads = list(set(next_available_roads) - set(last_available_roads))
        new_roads_feedback = self._get_new_roads_feedback(new_available_roads, last_stage)
        next_unconnected_regions = len(next_observations.get('unconnected_regions', []))
        next_avgdist = next_observations['avgdist']
        unconnected_regions_decrease = last_unconnected_regions - next_unconnected_regions
        avgdist_decrease = last_avgdist - next_avgdist

        if last_stage == 1:
            env_feedback = {
                f"road_{selected_road_id}": {
                    "newly_connected_regions": unconnected_regions_decrease,
                    "potential_highest_connected_regions_via_new_roads": new_roads_feedback
                }
            }
        else:
            env_feedback = {
                f"road_{selected_road_id}": {
                    "distance_reduction_among_regions": avgdist_decrease,
                    "potential_highest_distance_reduction_via_new_roads": new_roads_feedback
                }
            }

        self.last_observations = next_observations
        data_obs = self.get_data_texts()
        self.available_roads = next_available_roads
        answer_option_forms = self._create_answer_option_forms()
        text_obs = self.get_env_feedback_text(env_feedback) + data_obs 
        NR, AD, SC = self.get_metrics()
        info = {
            "last_observations": self.last_observations,
            "next_observations": next_observations,
            "decision": action,
            "reward": reward,
            "done": done,
            "feadback": env_feedback,  
            "text_obs": text_obs,
            "avg_dist": AD,
            "num_roads": NR,
            "cost": SC,
            "available_actions": [str(road_data) for road_data in next_available_roads],
        }

        return text_obs, reward, done, False, info # False for truncated as not used in this context

    def _parse_decision(self, decision: str) -> int:
        """
        Extract the road id from the decision string.

        :param decision: Decision string (e.g., "road_3").
        :return: Parsed road id as integer.
        """
        decision = decision.lower()
        selected_road_id = decision
        if '_' in decision:
            try:
                selected_road_id = int(decision.split('_')[-1])
            except ValueError:
                pass
        elif " " in decision:
            try:
                selected_road_id = int(decision.split(' ')[-1])
            except ValueError:
                pass

        return selected_road_id

    def get_env_feedback_text(self, env_feedback: dict) -> str:
        """
        Generate a human-readable feedback text from environment feedback.

        :param env_feedback: Feedback dictionary from the environment step.
        :return: Formatted feedback string.
        """
        build_road = list(env_feedback.keys())[0]
        unit_text = 'km' if "distance_reduction_among_regions" in env_feedback[build_road] else ""
        road_id = int(build_road.split('_')[-1])
        keys = list(env_feedback[build_road].keys())
        results_type_text1 = keys[0]
        results_value1 = env_feedback[build_road][results_type_text1]
        results_type_text2 = keys[1]
        results_value2 = env_feedback[build_road][results_type_text2]
        text = (
            f"After your action building road {road_id}, the environment has been changed:\n"
            f"- {results_type_text1}: {results_value1}{unit_text}\n"
            f"- {results_type_text2}: {results_value2}{unit_text}\n"
        )
        return text

    def check_stage(self):
        """
        Update the simulation stage based on environment status.
        """
        if self.stage == 1 and self.env._stage == 'full_connected':
            self.NR = len(self.env._mg.road_edges)
        self.stage = 2 if self.env._stage == 'full_connected' else 1

    def get_observation(self, done=False) -> dict:
        """
        Retrieve and construct the current state observation.

        :return: Observation dictionary with regions, connectivity, and available roads.
        """
        observation = {}
        observation['stage'] = 2 if self.env._stage == 'full_connected' else 1

        mg = self.env._mg
        region_list = mg.inner_facelist
        road_list = mg.road_edges
        edge_list = mg.edge_list
        node_list = mg.node_list
        edge_length_list = mg.edge_length

        # Determine connected regions
        connected_regions = []
        for road in road_list:
            edge_id = edge_list.index(road)
            for region in mg.edge_face_index[edge_id]:
                if region not in connected_regions:
                    connected_regions.append(region)
        observation['connected_regions'] = []
        for region in connected_regions:
            region_id = region_list.index(region)
            region_nodes = [node_list.index(node) for node in region.nodes]
            observation['connected_regions'].append({
                'region_id': region_id,
                'region_nodes': region_nodes
            })

        # Determine unconnected regions (only in stage 1)
        if observation['stage'] == 1:
            unconnected_regions = [region for region in region_list if region not in connected_regions]
            observation['unconnected_regions'] = []
            for region in unconnected_regions:
                region_id = region_list.index(region)
                region_nodes = [node_list.index(node) for node in region.nodes]
                observation['unconnected_regions'].append({
                    'region_id': region_id,
                    'region_nodes': region_nodes
                })

        # Determine potential roads
        potential_roads = []
        edge_mask = mg._get_edge_mask()
        for edge_id, edge in enumerate(edge_list):
            if edge_mask[edge_id] != 0:
                potential_roads.append(edge)
        observation['available_roads'] = []
        observation['connectivity'] = []

        # Build connectivity information
        for road in road_list:
            road_length = round(edge_length_list[edge_list.index(road)], 2)
            road_nodes = (node_list.index(road.nodes[0]), node_list.index(road.nodes[1]))
            observation['connectivity'].append((road_nodes[0], road_nodes[1], road_length))

        # Build available roads data
        for road in potential_roads:
            road_id = edge_list.index(road)
            road_length = round(edge_length_list[edge_list.index(road)], 2)
            road_nodes = (node_list.index(road.nodes[0]), node_list.index(road.nodes[1]))
            if done:
                new_roads_data = []
            else:
                new_roads_data = self.get_new_road_data(road_id, potential_roads)
            road_data = {
                'road_id': road_id,
                'edge': road_nodes,
                'length': road_length,
                'new_roads': new_roads_data
            }
            observation['available_roads'].append(road_data)
        observation['avgdist'] = mg.get_f2f_avgdist()
        return observation
    

    def get_new_road_data(self, road_id: int, potential_roads: list) -> list:
        """
        Retrieve data for new roads that become available after building a given road.

        :param road_id: The id of the road being built.
        :param potential_roads: List of currently potential roads.
        :return: List of dictionaries containing new road data.
        """
        new_env = copy.deepcopy(self.env)
        mg = new_env._mg
        node_list = mg.node_list
        edge_list = mg.edge_list
        edge_length_list = mg.edge_length
        new_env.step(road_id)
        new_roads_data = []
        edge_mask = mg._get_edge_mask()
        for edge_id, edge in enumerate(edge_list):
            if edge_mask[edge_id] != 0 and edge not in potential_roads:
                road_data = {
                    'road_id': edge_id,
                    'edge': (node_list.index(edge.nodes[0]), node_list.index(edge.nodes[1])),
                    'length': round(edge_length_list[edge_list.index(edge)], 2)
                }
                new_roads_data.append(road_data)
        return new_roads_data

    def get_data_texts(self) -> str:
        """
        Construct a text summary of the current state for decision making.

        :return: Data text string.
        """
        mg = self.env._mg
        stage = 2 if self.env._stage == 'full_connected' else 1
        region_list = mg.inner_facelist
        road_list = mg.road_edges
        edge_list = mg.edge_list
        node_list = mg.node_list
        edge_length_list = mg.edge_length

        # Determine connected regions
        connected_regions = []
        for road in road_list:
            edge_id = edge_list.index(road)
            for region in mg.edge_face_index[edge_id]:
                if region not in connected_regions:
                    connected_regions.append(region)
        # Determine unconnected regions
        unconnected_regions = [region for region in region_list if region not in connected_regions]

        # Determine potential roads
        potential_roads = []
        edge_mask = mg._get_edge_mask()
        for edge_id, edge in enumerate(edge_list):
            if edge_mask[edge_id] != 0:
                potential_roads.append(edge)

        # Construct data text
        
        data_text = f"We are at stage {stage}.\n\n"
        if stage == 2:
            data_text += "Connected regions:\n\n"
            for region in connected_regions:
                region_id = region_list.index(region)
                region_nodes = [node_list.index(node) for node in region.nodes]
                data_text += f"region {region_id}:\n- region_nodes: {region_nodes}\n\n"
            data_text += "Connectivity:\n["
            for i, road in enumerate(road_list):
                if i != 0:
                    data_text += ", "
                road_length = round(edge_length_list[edge_list.index(road)], 2)
                road_nodes = (node_list.index(road.nodes[0]), node_list.index(road.nodes[1]))
                data_text += f"(node {road_nodes[0]}, node {road_nodes[1]}, {road_length}km)"
            data_text += "]\n\n"
        if stage == 1:
            data_text += "Unconnected regions:\n\n"
            for region in unconnected_regions:
                region_id = region_list.index(region)
                region_nodes = [node_list.index(node) for node in region.nodes]
                data_text += f"region {region_id}:\n- region_nodes: {region_nodes}\n"
        data_text += "Available roads:\n\n"
        potential_roads_dict = {}
        for road in potential_roads:
            road_id = edge_list.index(road)
            road_length = round(edge_length_list[edge_list.index(road)], 2)
            road_text = f"(node {node_list.index(road.nodes[0])}, node {node_list.index(road.nodes[1])}, {road_length}km)"
            new_roads_id_list, new_roads_text = self.get_new_roads(road_id, potential_roads)
            potential_roads_dict[road_id] = {
                'distance': edge_length_list[road_id],
                'duration': edge_length_list[road_id]
            }
            road_connected_regions = mg.edge_face_index[road_id]
            # if stage == 1:
            #     new_regions=[]
            #     potential_roads_dict[road_id]['topology1'] = 0
            #     potential_roads_dict[road_id]['topology2'] = 0
            #     new_connected_regions = connected_regions.copy()
            #     for region in road_connected_regions:
            #         if region in unconnected_regions:
            #             new_connected_regions.append(region)
            #             new_regions.append(region)
            #             potential_roads_dict[road_id]['topology1'] += 1
            #     topology2_list = []
            #     for new_r_id in new_roads_id_list:
            #         topology2 = 0
            #         road_connected_regions = mg.edge_face_index[new_r_id]
            #         for region in road_connected_regions:
            #             if region not in new_connected_regions:
            #                 topology2 += 1
            #         topology2_list.append(topology2)
            #     potential_roads_dict[road_id]['topology2'] = max(topology2_list) if topology2_list else 0
            # else:
            #     potential_roads_dict[road_id]['topology1'] = self._get_decreased_distance(road_id)
            #     decreased_distance_list = []
            #     for new_r_id in new_roads_id_list:
            #         decreased_distance = self._get_new_road_decreased_distance_after_road(road_id, new_r_id)
            #         decreased_distance_list.append(decreased_distance)
            #     potential_roads_dict[road_id]['topology2'] = max(decreased_distance_list) if decreased_distance_list else 0
            data_text += f"road {road_id}:\n"
            data_text += f"- road_edge: {road_text}\n"
            if len(road_connected_regions)>0:
                new_connected_regions = [f"region {region_list.index(_region)}" for _region in road_connected_regions]
                road_region_text = ", ".join(new_connected_regions)
                data_text += f"- connect regions: {road_region_text}\n"
            data_text += f"- new_roads: {new_roads_text}\n"
        data_text += "\n"
        return data_text

    def _calc_road_unconnected_region(self, road_id: int) -> int:
        """
        Calculate how many unconnected regions would be linked by constructing the given road.

        :param road_id: ID of the road to test.
        :return: Number of newly connected regions.
        """
        mg = self.env._mg
        region_list = mg.inner_facelist
        road_list = mg.road_edges
        edge_list = mg.edge_list
        node_list = mg.node_list

        connected_regions = []
        for road in road_list:
            edge_id = edge_list.index(road)
            for region in mg.edge_face_index[edge_id]:
                if region not in connected_regions:
                    connected_regions.append(region)
        unconnected_regions = [region for region in region_list if region not in connected_regions]

        road = edge_list[road_id]
        road_nodes = (node_list.index(road.nodes[0]), node_list.index(road.nodes[1]))
        new_connected_regions = []
        for region in unconnected_regions:
            region_nodes = [node_list.index(node) for node in region.nodes]
            if road_nodes[0] in region_nodes or road_nodes[1] in region_nodes:
                new_connected_regions.append(region)
        return len(new_connected_regions)

    def _get_new_roads_feedback(self, new_available_roads: list, last_stage: int, done: bool = False) -> int:
        """
        Evaluate the maximum potential feedback from newly available roads.

        :param new_available_roads: List of new road ids.
        :param last_stage: The previous simulation stage.
        :param done: Whether the simulation has ended.
        :return: Maximum feedback value.
        """
        feedback_values = []
        for road_id in new_available_roads:
            if last_stage == 1:
                feedback_value = self._calc_road_unconnected_region(road_id)
            else:
                feedback_value = 0 if done else self._get_new_road_decreased_distance(road_id)
            feedback_values.append(feedback_value)
        return max(feedback_values) if feedback_values else 0

    def _get_decreased_distance(self, road_id: int) -> float:
        """
        Calculate the reduction in average distance after constructing a road.

        :param road_id: Road id to test.
        :return: Decreased distance value.
        """
        new_env = copy.deepcopy(self.env)
        mg = new_env._mg
        origin_avgdist = mg.get_f2f_avgdist()
        new_env.step(road_id)
        after_avgdist = mg.get_f2f_avgdist()
        return origin_avgdist - after_avgdist

    def _get_new_road_decreased_distance(self, road_id: list) -> float:
        """
        Calculate the decreased average distance for a new road option.

        :param road_id: The ids of the road being built.
        :return: Decreased distance value.
        """
        origin_avgdist = self.env._mg.get_f2f_avgdist()
        new_env = copy.deepcopy(self.env)
        new_env.step(road_id)
        after_avgdist = new_env._mg.get_f2f_avgdist()
        return origin_avgdist - after_avgdist
    
    def _get_new_road_decreased_distance_after_road(self, road_id, new_road_id):
        """
        Calculate the decreased average distance after building a road.
        :param road_id: The id of the road being built.
        :param new_road_id: The id of the new road being built.
        :return: Decreased distance value.
        """
        origin_avgdist = self.env._mg.get_f2f_avgdist()
        new_env = copy.deepcopy(self.env)
        next_state, reward, done, info = new_env.step(road_id)
        if done:
            return 0
        else:
            orginal_avgdist = new_env._mg.get_f2f_avgdist()
            next_state, reward, done, info = new_env.step(new_road_id)
            after_avgdist = new_env._mg.get_f2f_avgdist()
            return orginal_avgdist - after_avgdist

    def get_decision_making_texts(self, data_text: str) -> str:
        """
        Generate the prompt text for the decision-making process.

        :param data_text: Data summary text.
        :return: Full decision-making prompt text.
        """
        return data_text

    def get_analysis_text(self, potential_roads_dict: dict) -> str:
        """
        Generate an analysis text ranking the potential roads based on various metrics.

        :param potential_roads_dict: Dictionary containing potential road metrics.
        :return: Analysis text string.
        """
        stage = 2 if self.env._stage == 'full_connected' else 1
        analysis_text = "## Analysis\n\n"
        if stage == 1:
            analysis_text += "Rank of the available roads by the number of regions that can be connected by them:\n"
            topology2_text = (
                "Rank of the available roads by the average number of unconnected regions that can be linked to the already connected regions "
                "through newly buildable roads after constructing the initial available road:\n"
            )
        else:
            analysis_text += "Rank of the available roads by the effect of reducing travel distance among regions:\n"
            topology2_text = (
                "Rank of the available roads by the average deduction on travel distance of regions through newly buildable roads after constructing the initial available road:\n"
            )
        topology1_rank = self.get_road_id_sorted_by_values(potential_roads_dict, 'topology1')
        analysis_text += self.get_rank_text(topology1_rank, potential_roads_dict, 'topology1')
        analysis_text += topology2_text
        topology2_rank = self.get_road_id_sorted_by_values(potential_roads_dict, 'topology2')
        analysis_text += self.get_rank_text(topology2_rank, potential_roads_dict, 'topology2')
        analysis_text += "Rank of the road lengths:\n"
        distance_rank = self.get_road_id_sorted_by_values(potential_roads_dict, 'distance')
        analysis_text += self.get_rank_text(distance_rank, potential_roads_dict, 'distance')
        return analysis_text

    def get_road_id_sorted_by_values(self, potential_roads_dict: dict, key_name: str) -> list:
        """
        Sort road IDs based on a given metric.

        :param potential_roads_dict: Dictionary with road metrics.
        :param key_name: The key in the dictionary to sort by.
        :return: List of road ids sorted accordingly.
        """
        scores = [(road_id, info[key_name]) for road_id, info in potential_roads_dict.items()]
        reverse_order = key_name in ['topology1', 'topology2']
        sorted_ids = sorted(scores, key=lambda x: x[1], reverse=reverse_order)
        return [road_id for road_id, _ in sorted_ids]

    def get_rank_text(self, sorted_ids: list, potential_roads_dict: dict, key_name: str) -> str:
        """
        Generate a ranking string based on sorted road ids and their corresponding metric.

        :param sorted_ids: List of sorted road ids.
        :param potential_roads_dict: Dictionary with road metrics.
        :param key_name: The key corresponding to the metric.
        :return: Formatted ranking string.
        """
        separator = " < " if key_name in ['distance', 'duration'] else " > "
        rank_text = ""
        for i, road_id in enumerate(sorted_ids):
            if i != len(sorted_ids) - 1:
                current_val = round(potential_roads_dict[road_id][key_name], 2)
                next_val = round(potential_roads_dict[sorted_ids[i + 1]][key_name], 2)
                sep = " = " if current_val == next_val else separator
                rank_text += f"road {road_id}{sep}"
            else:
                rank_text += f"road {road_id}\n\n"
        return rank_text

    def get_new_roads(self, road_id: int, potential_roads: list) -> tuple:
        """
        Get new roads that become available after constructing a given road.

        :param road_id: The id of the road being built.
        :param potential_roads: List of currently potential roads.
        :return: Tuple with list of new road ids and a formatted text description.
        """
        new_env = copy.deepcopy(self.env)
        mg = new_env._mg
        edge_list = mg.edge_list
        new_env.step(road_id)
        new_roads = []
        new_roads_id_list = []
        edge_mask = mg._get_edge_mask()
        for edge_id, edge in enumerate(edge_list):
            if edge_mask[edge_id] != 0 and edge not in potential_roads:
                new_roads.append(edge)
                new_roads_id_list.append(edge_id)
        node_list = mg.node_list
        new_roads_text = []
        for r_id, r in zip(new_roads_id_list, new_roads):
            node0 = node_list.index(r.nodes[0])
            node1 = node_list.index(r.nodes[1])
            length = round(mg.edge_length[edge_list.index(r)], 2)
            new_roads_text.append(f"(node {node0}, node {node1}, {length}km)")
        return new_roads_id_list, f"[{', '.join(new_roads_text)}]"
    
    def _get_sum_of_costs(self):
        road_list = self.env._mg.road_edges
        edge_list = self.env._mg.edge_list
        edge_length_list = self.env._mg.edge_length
        road_length_list = [edge_length_list[edge_list.index(road)] for road in road_list]
        return sum(road_length_list)

    def get_metrics(self):
        """
        Calculate and return the number of road segments, average distance, and sum of costs.
        :return: Tuple containing NR, AD, and SC."
        """
        NR = self.NR
        AD = self.env._mg.get_f2f_avgdist()
        SC = self._get_sum_of_costs()
        return NR, AD, SC
    
    def render(self, mode="human"):
        """
        Render the environment state.

        :param mode: Rendering mode (default is "human").
        :return: Rendered output.
        """
        pass
    
    def close(self):
        """
        Close the environment and release resources.
        """
        try:
            self.env.close()
        except:
            pass

import ray
@ray.remote(num_cpus=0.25)
class RoadWorker:
    def __init__(self, seed, env_kwargs):
        np.random.seed(seed)
        self.env = RoadPlanningEnvironment(**env_kwargs)

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
class RoadMultiProcEnv(Env):
    def __init__(self, seed=0, env_num=1, group_n=1, env_kwargs=None):
        super().__init__()
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.env_kwargs = env_kwargs or {}
        self.env_num = env_num
        self.group_n = group_n
        self.num_workers = env_num * group_n
        self.workers = [
            RoadWorker.remote(seed + i//group_n, self.env_kwargs)
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

from .. import utils
def build_road_envs(
    seed: int = 0,
    env_num: int = 1,
    group_n: int = 1,
    is_train: bool = True,
    env_kwargs: dict = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    slum_name = 'CapeTown1'
    root_dir = utils.get_root_dir()
    root_dir = f'{root_dir}/agent_system/environments/env_package/road_planning/logs'
    cfg = Config('demo', slum_name, seed, False, root_dir)
    env_kwargs = {
        "cfg": cfg
    }
    return RoadMultiProcEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        env_kwargs=env_kwargs,
    )

import re
import json
from typing import List, Tuple

def road_projection(actions: List[str]):
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

