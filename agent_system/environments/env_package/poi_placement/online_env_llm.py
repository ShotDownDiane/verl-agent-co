from gymnasium import Env, spaces
import numpy as np
from . import evaluation_framework_plus as ef
from .online_env_tensor import StationPlacement
from jinja2 import Template


class LLMWrapperEnv(Env):
    """
    Wraps the StationPlacement env to produce string/text-based observations and accept discrete LLM-style actions.
    """

    def __init__(self, **env_kwargs):
        super(LLMWrapperEnv, self).__init__()
        self.base_env = StationPlacement(**env_kwargs)
        self.action_space = spaces.Discrete(5)  # A-E mapped to 0-4
        self.observation_space = spaces.Text(5120)  # Dummy placeholder; actual output is string
        self.node_list = self.base_env.node_list
        self.previous_action = None
        self.old_node_benefit_info = None
        self.fault_times = 0
        self.new_node_benefit_info = None

    def reset(self, seed=None, options=None):
        obs, _ = self.base_env.reset(seed=int(seed))
        my_benefit, my_cost, charg_time, wait_time, travel_time = ef.existing_score(self.base_env.plan_instance.plan,self.base_env.node_list)
        self.old_node_benefit_info = {
            "cov": my_benefit,
            "cos": my_cost,
            "chg": charg_time,
            "wat": wait_time,
            "tra": travel_time
        }
        text_obs = self._generate_llm_observation()
        return text_obs, {}

    def convert_llm_action_to_env_action(self, llm_action):
        """
        Converts LLM action to environment action.
        Example llm_action: {"summary": "Install 3 fast chargers at node 1", "answer": "A"}
        Maps "A" to 0, "B" to 1, etc.
        """
        action_map = {
            "A": 0,
            "B": 1,
            "C": 2,
            "D": 3,
            "E": 4
        }
        return action_map.get(llm_action, -1)

    def step(self, llm_action):
        """
        Example action:
        {
            "summary": "Install 3 fast chargers at node 1",
            "answer": "A"
        }
        """
        self.previous_action = llm_action
        action = self.convert_llm_action_to_env_action(llm_action)
        # 连续错误3次后，默认创建一个充电站
        if action == -1:
            self.fault_times += 1
            if self.fault_times > 2:
                action = 0  # Default to creating by coverage if too many invalid actions
                self.fault_times = 0
            else:
                text_obs = self._generate_llm_observation()
                return text_obs, 0, False, False, {"benefit_info": self.old_node_benefit_info}
        # Reset fault times on valid action
        self.fault_times = 0
        print(f"LLM action: {llm_action}, converted to env action: {action}")
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        if self.new_node_benefit_info is not None:
            self.old_node_benefit_info = self.new_node_benefit_info
        self.new_node_benefit_info = info["benefit_info"]
        text_obs = self._generate_llm_observation()
        info["available_actions"] = ["A", "B", "C", "D", "E"]
        return text_obs, reward, terminated, truncated, info

    def render(self):
        return self.base_env.render()

    def get_location_data_text(self, nodes):       
        for station in nodes:
            
            if "covered" not in station[1]:
                cover = ef.node_coverage(self.base_env.plan_instance.plan, station[0])
                station[0][1]["covered"]=cover
                station = (station[0][0],station[0][1],station[2])

            if station:
                wait_times = station[2]["D_s"] * station[2]["W_s"] if len(station) > 2 else 0
                charging_times = station[2]["D_s"] / station[2]["service rate"] if len(station) > 2 else 0
                text = ""
                text += f"Station {station[0]}:\n"
                text += f"  Location: {station[1]['x']:.3f},{station[1]['y']:.3f}\n"
                text += f"  Demand: {station[1]['demand']:.2f}\n"
                text += f"  Coverage: {station[1]['covered']:.2f}\n"
                text += f"  Wait Time: {wait_times*1000:.2f}\n" if len(station) > 2 else "  Wait Time: NaN\n"
                text += f"  Charging Time: {charging_times*1000:.2f}\n" if len(station) > 2 else "  Charging Time: NaN\n"

        return text

#     def _get_task_info(self, data_text):
#         task_info = {
#             "task_description": "We are tasked with determining the optimal location for a new electric vehicle (EV) charging station to maximize benefits by effectively meeting demand.",
#             "data_schema": "- Coverage: The expected coverage of the charging station after placement.\n- Demand: The number of nearby vehicles requiring charging.\n- Distance: The average distance to nearby existing charging stations.\n- Waiting Time: The average waiting time at nearby existing charging stations. NaN indicates no station nearby and the highest waiting time.\n- Charging Time: The average charging time at nearby existing charging stations. NaN indicates no station nearby and with the highest charging time.",
#             "domain_knowledge": "- Higher expected coverage indicates a more sufficient supply after placement.\n- Higher demand suggests greater potential benefits.\n- Longer average distance to existing charging stations indicates a greater need for new stations nearby.\n- You MUST first prioritize maximizing coverage, while reducing overall travel, waiting, and charging time is of secondary importance.",
#             "task_target": "identify which location would deliver the most substantial benefits for an EV charging station, focusing on enhancing overall station coverage, minimizing overall travel time, reducing overall waiting time, and reducing overall charging time. You can ONLY select one location to build the new station.",
#             "task_output_type": "decision",
#             "action_space": "- A: Create by coverage: Select the node with the lowest coverage and create a new charging station.\n - B: Create by demand: Select the node with the highest weakened demand and create a new charging station.\n- C: Add by coverage: Add one charger to an existing station selected by lowest coverage.\n- D: Add by demand: Add one charger to an existing station selected by highest demand.\n- E: Relocate charger: Move one charger from the lowest-benefit station to the station with the highest waiting and charging time."
#         }
#         template = """
#         You are an expert in spatiotemporal analysis with a focus on urban management tasks. 
# Apply your domain knowledge and reasoning abilities to answer the following question.

# ## Task
# Determine the optimal action to maximize benefits of EV charging station placement, prioritizing coverage first, then minimizing travel, waiting, and charging time.

# ## Data Schema
# - Coverage: Expected coverage after placement.
# - Demand: Number of nearby vehicles needing charging.
# - Distance: Avg distance to existing stations.
# - Waiting Time: Avg waiting time at existing stations (NaN = no station nearby, highest wait).
# - Charging Time: Avg charging time at existing stations (NaN = no station nearby, highest charge time).

# ## Data
# <data_text>

# ## environment change 
# <environment>

# ## Domain Knowledge
# - Prioritize maximizing **coverage**.
# - Secondary: minimize travel, waiting, charging time.
# - High demand = higher potential benefit.
# - Long distance to stations = greater need.

# ## Actions
# A: Create by coverage — build at node with lowest coverage  
# B: Create by demand — build at node with highest demand  
# C: Add by coverage — add charger at station with lowest coverage  
# D: Add by demand — add charger at station with highest demand  
# E: Relocate charger — move charger from lowest-benefit station to highest-need station  

# ## Instruction 
# ou must first provide your detailed reasoning inside <think> tags. \n And then provide ONLY a valid JSON inside <answer> tags, like:\
# {
# "summary": "YOUR_SUMMARY"
# "answer": "<A|B|C|D|E>",  
# }
# IMPORTANT: The <answer> tag must contain only the JSON object, no other text.
# Limit response to 500 words.
#         """

#         obs = template.replace("<data_text>", data_text)\
#                     .replace("<environment>", self.get_env_change_text())
#         obs = obs.strip()
#         return obs
    def _get_task_info(self, data_text):
        obs = self.get_env_change_text()
        obs += data_text
        return obs

    def get_env_change_text(self):
        old_node_benefit_info = self.old_node_benefit_info
        new_node_benefit_info = self.new_node_benefit_info
        new_node = self.previous_action
        if new_node not in ["A", "B", "C", "D", "E"]:
            return f"Because your last action {self.previous_action} is not valid, the environment will not change. Please generate a valid response."
        
        if self.new_node_benefit_info is None:
            text  = (f"Environment initialized with existing stations, initial state:\n"
                    f"Benefit: {round(old_node_benefit_info['cov'], 2)}, Cost: {round(old_node_benefit_info['cos'], 2)}, "
                    f"Waiting Time: {round(old_node_benefit_info['wat'], 2)}, Charging Time: {round(old_node_benefit_info['chg'], 2)}, "
                    f"Travel Time: {round(old_node_benefit_info['tra'], 2)}.\n")
            self.new_node_benefit_info = old_node_benefit_info

            return text

        text = (f"Note that a negative value indicates adverse effects, while a positive value signifies benefits.\n"
                f"After you last action {new_node}, the enviromant has been changed:\n"
                f"Coverage: {round(old_node_benefit_info['cov'], 2)}, Coverage Increase: {round((new_node_benefit_info['cov'] - old_node_benefit_info['cov']) * 100 / old_node_benefit_info['cov'], 2)}%, "
                f"Cost: {round(old_node_benefit_info['cos'], 2)}, Cost Decrease: {round((old_node_benefit_info['cos'] - new_node_benefit_info['cos']) * 100 / old_node_benefit_info['cos'], 2)}%, "
                f"Waiting Time: {round(old_node_benefit_info['wat'], 2)}, Waiting Time Decrease: {round((old_node_benefit_info['wat'] - new_node_benefit_info['wat']) * 100 / old_node_benefit_info['wat'], 2)}%, "
                f"Charging Time: {round(old_node_benefit_info['chg'], 2)}, Charging Time Decrease: {round((old_node_benefit_info['chg'] - new_node_benefit_info['chg']) * 100 / old_node_benefit_info['chg'], 2)}%,"
                f"Travel Time: {round(old_node_benefit_info['tra'], 2)}, Travel Time Decrease: {round((old_node_benefit_info['tra'] - new_node_benefit_info['tra']) * 100 / old_node_benefit_info['tra'], 2)}%.\n") 

        return text

    def _generate_llm_observation(self):
        cov_min_nodes = self._get_top_nodes(metric="coverage", with_station=False, top_k=1)
        dem_max_nodes = self._get_top_nodes(metric="demand", with_station=False, top_k=1)
        cov_min_stations = self._get_top_nodes(metric="coverage", with_station=True, top_k=1)
        dem_max_stations = self._get_top_nodes(metric="demand", with_station=True, top_k=1)
        low_benefit_stations = self._get_bottom_stations_by_benefit(top_k=1)
        high_need_station = self._get_highest_neediness_station(top_k=1)

        obs_text = "The current environment state:\n"
        obs_text += "The free node with the lowest coverage:\n" + self.get_location_data_text(cov_min_nodes) + "\n\n" 
        obs_text += "The free node with the highest weakened demand:\n" + self.get_location_data_text(dem_max_nodes) + "\n\n"
        obs_text += "The existing station with lowest coverage:\n" + self.get_location_data_text(cov_min_stations) + "\n\n"
        obs_text += "The existing station selected by highest demand\n" + self.get_location_data_text(dem_max_stations) + "\n\n"
        obs_text += "The lowest-benefit station:\n" + self.get_location_data_text(low_benefit_stations) + "\n\n" if low_benefit_stations else "NaN\n"
        obs_text += "The station with the highest waiting and charging time.:\n" +self.get_location_data_text(high_need_station) + "\n\n"
        obs_text = self._get_task_info(obs_text)
        return obs_text

    def _get_top_nodes(self, metric="coverage", with_station=False, top_k=5):
        node_list = self.base_env.node_list
        station_ids = set(s[0][0] for s in self.base_env.plan_instance.plan)

        def has_station(node): return node[0] in station_ids

        filtered = [
            node for node in node_list if has_station(node) == with_station
        ]

        if metric == "coverage":
            # lowest coverage first
            ranked = sorted(
                filtered, key=lambda n: n[1].get("covered", 0), reverse=False
            )
        elif metric == "demand":
            # highest demand first
            ranked = sorted(
                filtered, key=lambda n: n[1].get("demand", 0), reverse=True
            )
        else:
            raise ValueError("Unsupported metric")

    
        return ranked[:top_k] if len(ranked) >= top_k else ranked

    def _get_bottom_stations_by_benefit(self, top_k=3):
        steal_plan = [s for s in self.base_env.plan_instance.plan if s[0] not in self.base_env.plan_instance.existing_plan]
        plan_list = [station[0][0] for station in steal_plan]
        my_occupied_list = [node for node in self.base_env.node_list if node[0] in plan_list]

        def benefit(station):
            return station[1]["upper bound"]

        ranked = sorted(my_occupied_list, key=benefit)
        return ranked[:top_k] if ranked else None
    
    def _get_highest_neediness_station(self, top_k=3):
        """
        Returns the station with the highest neediness (wait_time+charging_time).
        """
        plans = [s for s in self.base_env.plan_instance.plan]

        def neediness(station):
            charg_time = station[2]["D_s"] / station[2]["service rate"]
            wait_time = station[2]["D_s"] * station[2]["W_s"]
            neediness = (wait_time + charg_time)
            return neediness

        ranked = sorted(plans, key=neediness, reverse=True)
        return ranked[:top_k] if ranked else None
