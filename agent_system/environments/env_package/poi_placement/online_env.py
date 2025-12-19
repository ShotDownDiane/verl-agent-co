import gym
from gym import spaces
from stable_baselines3.common.env_checker import check_env
import pickle
from math import ceil
import itertools
# from .evaluation_framework import *
from evaluation_framework import *
import copy

"""
Custom environment
"""

def prepare_config():
    """
    we prepare the power capacities of the different charging configuration and to find the cheapest ones
    """
    N = len(CHARGING_POWER)
    urn = list(range(0, K + 1)) * N
    config_list = []
    for combination in itertools.combinations(urn, N):
        config_list.append(list(combination))

    my_config_dict = {}
    for config in config_list:
        if np.sum(config) > K:
            continue
        else:
            capacity = np.sum(CHARGING_POWER * config)
            if capacity in my_config_dict.keys():
                # check if we have found a better configuration for the same capacity
                if np.sum(INSTALL_FEE * config) < np.sum(INSTALL_FEE * my_config_dict[capacity]):
                    my_config_dict[capacity] = config
            else:
                my_config_dict[capacity] = config
    # if we have a cheaper price at more capacity, we will use that configuration even if less capacity is required
    key_list = sorted(list(my_config_dict.keys()))
    for index, key in enumerate(key_list):
        cost_list = [np.sum(INSTALL_FEE * my_config_dict[my_key]) for my_key in key_list[index:]]
        best_cost_index = cost_list.index(min(cost_list)) + index
        best_config = my_config_dict[key_list[best_cost_index]]
        my_config_dict[key] = best_config

    my_config_dict.pop(0)
    return my_config_dict


def initial_solution(my_config_dict, my_node_list, s_pos):
    """
    get the initial solution for the charging configuration
    """
    W = 0  # minimum capacity constraint
    radius = 50
    for my_node in my_node_list:
        if haversine(s_pos, my_node) <= radius:
            W += weak_demand(my_node)
    W = ceil(W) * BATTERY
    key_list = sorted(list(my_config_dict.keys()))
    for key in key_list:
        if key > W:
            break
    best_config = my_config_dict[key]
    return best_config


def coverage(my_node_list, my_plan):
    """
    see which nodes are covered by the charging plan
    """
    for my_node in my_node_list:
        cover = node_coverage(my_plan, my_node)
        my_node[1]["covered"] = cover


def choose_node_new_benefit(free_list):
    """
    pick location which the smallest coverage
    """
    upbound_list = [my_node[1]["covered"] for my_node in free_list]
    pos_minindex = upbound_list.index(min(upbound_list))
    chosen_node = free_list[pos_minindex]
    return chosen_node


def choose_node_bydemand(free_list):
    """
    pick location with the highest weakened demand
    """
    demand_list = [my_node[1]["demand"] * (1 - 0.1 * my_node[1]["private CS"]) for my_node in free_list]
    chosen_index = demand_list.index(max(demand_list))
    chosen_node = free_list[chosen_index]
    return chosen_node


def anti_choose_node_bybenefit(my_node_list, my_plan):
    """
    choose station with the least coverage
    """
    plan_list = [station[0][0] for station in my_plan]
    my_occupied_list = [node for node in my_node_list if node[0] in plan_list]
    if not my_occupied_list:
        return None
    upbound_list = [node[1]["upper bound"] for node in my_occupied_list]
    pos_minindex = upbound_list.index(min(upbound_list))
    remove_node = my_occupied_list[pos_minindex]
    plan_index = plan_list.index(remove_node[0])
    remove_station = my_plan[plan_index]
    return remove_station


def support_stations_hilfe(station):
    charg_time = station[2]["D_s"] / station[2]["service rate"]
    wait_time = station[2]["D_s"] * station[2]["W_s"]
    neediness = (wait_time + charg_time)
    return neediness


def support_stations(my_plan, free_list):
    """
    choose a station which needs support due to highest waiting + charging time
    """
    cost_list = [support_stations_hilfe(station) for station in my_plan]
    if not cost_list:
        chosen_node = choose_node_bydemand(free_list)
    else:
        index = cost_list.index(max(cost_list))
        station_sos = my_plan[index]
        if sum(station_sos[1]) < K:
            chosen_node = station_sos[0]
        else:
            # look for nearest node that could support the station
            dis_list = [haversine(station_sos[0], my_node) for my_node in free_list]
            min_index = dis_list.index(min(dis_list))
            chosen_node = free_list[min_index]
    return chosen_node

def get_location_data_text(location_ids, loc_cov, loc_dem, loc_dis, loc_wat, loc_chg):
    text = "\n\n".join([f"location {i}:\n"
                        f"- id: {loc}\n"
                        f"- coverage: {round(loc_cov[loc], 2)}\n"
                        f"- demand: {round(loc_dem[loc], 2)}\n"
                        f"- distance: {round(-loc_dis[loc], 2)}\n"
                        f"- waiting_time: {round(loc_wat[loc], 2) if loc_wat[loc] != -1 else 'NaN'}\n"
                        f"- charging_time: {round(loc_chg[loc], 2) if loc_chg[loc] != -1 else 'NaN'}\n"
                        f"- estate cost: {round(INSTALL_FEE[loc], 2) if loc in INSTALL_FEE else 'NaN'}\n"
                        f"- charger_config: {CHARGER_CONFIG[loc] if loc in CHARGER_CONFIG else 'None'}"
                        for i, loc in enumerate(location_ids)])

    return text

def get_env_change_text(old_node_benefit_info, new_node, station_placement_instance):
    new_cov, _, new_demand, new_distance, new_wait, new_charge = node_social_benefit(new_node, station_placement_instance.node_list,
                                                                                     station_placement_instance.plan_instance.plan)

    text = (f"Note that a negative value indicates adverse effects, while a positive value signifies benefits.\n"
            f"After constructing a charging station at location {old_node_benefit_info['id']}:\n"
            f"Coverage: {round(old_node_benefit_info['cov'], 2)}, Coverage Increase: {round((new_cov - old_node_benefit_info['cov']) * 100 / old_node_benefit_info['cov'], 2)}%, "
            f"Distance: {round(old_node_benefit_info['dis'], 2)}, Distance Decrease: {round((old_node_benefit_info['dis'] - new_distance) * 100 / old_node_benefit_info['dis'], 2)}%, "
            f"Waiting Time: {round(old_node_benefit_info['wat'], 2)}, Waiting Time Decrease: {round((old_node_benefit_info['wat'] - new_wait) * 100 / old_node_benefit_info['wat'], 2)}%, "
            f"Charging Time: {round(old_node_benefit_info['chg'], 2)}, Charging Time Decrease: {round((old_node_benefit_info['chg'] - new_charge) * 100 / old_node_benefit_info['chg'], 2)}%")

    return text

class Plan:
    def __init__(self, my_node_list, my_node_dict, my_cost_dict, my_plan_file):
        with (open(my_plan_file, "rb")) as f:
            self.plan = pickle.load(f)
        self.plan = [s_dictionary(my_station, my_node_list) for my_station in self.plan]
        my_node_list, my_node_dict, my_cost_dict = my_node_list, my_node_dict, my_cost_dict
        self.plan, _ = self.update_node_info(self.plan, my_node_list, my_node_dict, my_cost_dict)

        # update norm
        self.norm_benefit, self.norm_cost, self.norm_charg, self.norm_wait, self.norm_travel = existing_score_fixed(self.plan, my_node_list)
        self.existing_plan = self.plan.copy()
        self.existing_plan = [s[0] for s in self.existing_plan]

    def update_node_info(self, plan, my_node_list, my_node_dict, my_cost_dict):
        my_node_list, _, _ = station_seeking(plan, my_node_list, my_node_dict, my_cost_dict)
        plan = [s_dictionary(my_station, my_node_list) for my_station in plan]

        return plan, my_node_list

    def __repr__(self):
        return "The charging plan is {}".format(self.plan)

    def add_plan(self, my_station):
        self.plan.append(my_station)

    def remove_plan(self, my_station):
        self.plan.remove(my_station)

    def steal_column(self, stolen_station, my_budget):
        """
        steal a charger from the station, give budget back and check which charger type has been stolen
        """
        my_budget += stolen_station[2]["fee"]
        station_index = self.plan.index(stolen_station)
        # we choose the most expensive charging column
        if stolen_station[1][2] > 0:
            self.plan[station_index][1][2] -= 1
            config_index = 2
        elif stolen_station[1][1] > 0:
            self.plan[station_index][1][1] -= 1
            config_index = 1
        else:
            self.plan[station_index][1][0] -= 1
            config_index = 0
        if sum(stolen_station[1]) == 0:
            # this means we remove the entire stations as it only has one charger
            self.remove_plan(stolen_station)
        else:
            # the station remains, we only steal one charging column
            installment_fee(stolen_station)
            my_budget -= stolen_station[2]["fee"]
        return my_budget, config_index


class Station:
    def __init__(self):
        self.s_pos = None
        self.s_x = None
        self.s_dict = {}
        self.station = [self.s_pos, self.s_x, self.s_dict]

    def __repr__(self):
        return "This station is {}".format(self.station)

    def add_position(self, my_node):
        self.station[0] = my_node

    def add_chargers(self, my_config):
        self.station[1] = my_config

    def establish_dictionary(self, node_list):
        self.station = s_dictionary(self.station, node_list)


class StationPlacement(gym.Env):
    """Custom Environment that follows gym interface"""
    node_dict = {}
    cost_dict = {}

    def __init__(self, my_graph_file, my_node_file, my_plan_file):
        super(StationPlacement, self).__init__()
        _graph, self.node_list = prepare_graph(my_graph_file, my_node_file)
        self.plan_file = my_plan_file
        self.node_list = [self.init_hilfe(my_node) for my_node in self.node_list]
        self.node_xlist = None
        self.game_over = None
        self.budget = None
        self.plan_instance = None
        self.plan_length = None
        self.row_length = 5
        self.best_score = None
        self.best_plan = None
        self.best_node_list = None
        self.schritt = None
        self.config_dict = None
        # new action space including all charger types
        self.action_space = spaces.Discrete(5)
        shape = (self.row_length + len(CHARGING_POWER)) * len(self.node_list) + 1
        self.observation_space = spaces.Box(low=-1, high=1, shape=(shape,), dtype=np.float16)
        print("online environment initialized")

        self.reset()

    def reset(self):
        """
        Reset the state of the environment to an initial state
        """
        self.budget = BUDGET
        self.game_over = False
        self.plan_instance = Plan(self.node_list, StationPlacement.node_dict, StationPlacement.cost_dict,
                                  self.plan_file)
        self.best_score, _, _, _, _, _ = norm_score(self.plan_instance.plan, self.node_list,
                                                       self.plan_instance.norm_benefit, self.plan_instance.norm_charg,
                                                       self.plan_instance.norm_wait, self.plan_instance.norm_travel)
        self.plan_length = len(self.plan_instance.existing_plan)
        self.schritt = 0
        self.best_plan = []
        self.best_node_list = []
        self.config_dict = prepare_config()
        coverage(self.node_list, self.plan_instance.plan)

        self.plan_instance.plan, self.node_list = self.plan_instance.update_node_info(self.plan_instance.plan,
                                                                                      self.node_list,
                                                                                      StationPlacement.node_dict,
                                                                                      StationPlacement.cost_dict)

    def init_hilfe(self, my_node):
        StationPlacement.node_dict[my_node[0]] = {}  # prepare node_dict
        StationPlacement.cost_dict[my_node[0]] = {}
        my_node[1]["charging station"] = None
        my_node[1]["distance"] = None
        return my_node

    def new_station_benefit(self, chosen_node, charger_config=None, is_max=False):
        node_list_ = copy.deepcopy(self.node_list)
        plan_ = copy.deepcopy(self.plan_instance.plan)
        if is_max:
            charger_config = [0, 0, K]

        # init a station
        new_station = self.init_station(chosen_node, charger_config, node_list_)

        # calculate the potential benefit
        plan_.append(new_station.station)
        plan_, node_list_ = self.plan_instance.update_node_info(plan_, node_list_, StationPlacement.node_dict, StationPlacement.cost_dict)
        nearby_cov, nearby_cov_num, nearby_demand, nearby_distance, nearby_wait, nearby_charge = (
            social_benefit_new_station_with_demand_collection(chosen_node,
                                                              self.node_list,
                                                              self.plan_instance.plan,
                                                              plan_)
        )

        reward, benefit, cost, charge_time, wait_time, travel_time = norm_score(plan_, node_list_, self.plan_instance.norm_benefit,
                                                                                   self.plan_instance.norm_charg, self.plan_instance.norm_wait,
                                                                                   self.plan_instance.norm_travel)
        reward_info = {
            "benefit": benefit,
            "cost": cost,
            "charge_time": charge_time,
            "wait_time": wait_time,
            "travel_time": travel_time,
            "reward": reward
        }

        return nearby_cov, nearby_cov_num, nearby_demand, nearby_distance, nearby_wait, nearby_charge, reward_info


    def plan_cost(self, plan, node_list):
        cost_travel = travel_cost(node_list)
        charg_time = charging_time(plan)
        wait_time = waiting_time(plan)
        cost = cost_travel + charg_time + wait_time

        return cost

    def new_station_reward(self, chosen_node, charger_config=None, is_max=False):
        # ori reward
        ori_benefit = social_benefit_fixed(self.plan_instance.plan, self.node_list)
        ori_normed_benefit = ori_benefit / self.plan_instance.norm_benefit

        ori_cost = self.plan_cost(self.plan_instance.plan, self.node_list)
        ori_normed_cost = ori_cost / self.plan_instance.norm_cost

        ori_reward = ori_normed_benefit - ori_normed_cost

        # new reward
        node_list_ = copy.deepcopy(self.node_list)
        plan_ = copy.deepcopy(self.plan_instance.plan)
        if is_max:
            charger_config = [0, 0, K]

        # init a station
        new_station = self.init_station(chosen_node, charger_config, node_list_)

        # calculate the potential benefit
        plan_.append(new_station.station)
        plan_, node_list_ = self.plan_instance.update_node_info(plan_, node_list_, StationPlacement.node_dict, StationPlacement.cost_dict)

        # new reward
        new_benefit = social_benefit_fixed(plan_, node_list_)
        new_normed_benefit = new_benefit / self.plan_instance.norm_benefit

        new_cost = self.plan_cost(plan_, node_list_)
        new_normed_cost = new_cost / self.plan_instance.norm_cost

        new_reward = new_normed_benefit - new_normed_cost

        # final reward
        reward = new_reward - ori_reward

        return reward

    def init_station(self, chosen_node, charger_config, node_list):
        new_station = Station()
        new_station.add_position(chosen_node)
        new_station.add_chargers(charger_config)
        new_station.establish_dictionary(node_list)

        return new_station

    def step_heuristic(self, data_list, is_negative):
        """
        choose a location
        """
        station_list = [s[0][0] for s in self.plan_instance.plan]  # all charging stations
        free_list = [node for node in self.node_list if node[0] not in station_list]  # nodes without stations

        # calculate max potential benefits
        free_list_cov_benefits = []
        free_list_cov_num_benefits = []
        free_list_demand_benefits = []
        free_list_distance_benefits = []
        free_list_wait_benefits = []
        free_list_charge_benefits = []
        free_list_reward_info = []
        for node in free_list:
            supply_benefit, supply_benefit_num, demand_benefit, distance_benefit, wait_benefit, charge_benefit, reward_info = self.new_station_benefit(node, is_max=True)
            free_list_cov_benefits.append(supply_benefit)
            free_list_cov_num_benefits.append(supply_benefit_num)
            free_list_demand_benefits.append(demand_benefit)
            free_list_wait_benefits.append(wait_benefit)
            free_list_charge_benefits.append(charge_benefit)
            free_list_reward_info.append(reward_info)

        max_cov, min_cov = np.max(free_list_cov_benefits), np.min(free_list_cov_benefits)
        max_demand, min_demand = np.max(free_list_demand_benefits), np.min(free_list_demand_benefits)

        free_list_cov_benefits = np.array(free_list_cov_benefits)
        free_list_demand_benefits = np.array(free_list_demand_benefits)
        free_list_distance_benefits = np.array(free_list_distance_benefits)

        free_list_cov_benefits_norm = (free_list_cov_benefits - min_cov) / (max_cov - min_cov)
        free_list_demand_benefits_norm = (free_list_demand_benefits - min_demand) / (max_demand - min_demand)

        free_list_benefits = free_list_cov_benefits_norm * 0.7 + free_list_demand_benefits_norm * 0.3

        # select the best location
        best_node_idx = np.argmax(free_list_benefits)
        if is_negative:
            selected_idx = best_node_idx
            while selected_idx == best_node_idx:
                selected_idx = int(np.random.randint(0, len(free_list_benefits)))
        else:
            selected_idx = best_node_idx

        best_node = free_list[selected_idx]
        data_list.append({"cov_benefits": list(free_list_cov_benefits), "cov_benefits_num": free_list_cov_num_benefits,
                          "wait_benefit": list(free_list_wait_benefits), "charge_benefits": list(free_list_charge_benefits),
                          "demand_benefits": list(free_list_demand_benefits), "overall_benefits": list(free_list_benefits),
                          "distance_benefits": list(free_list_distance_benefits), "best_idx": int(best_node_idx),
                          "selected_idx": int(selected_idx), "reward_info": free_list_reward_info})

        """
        Add chargers
        """
        # initiate a config
        config_dict_ = self.config_dict.copy()
        init_charger_config = initial_solution(config_dict_, self.node_list, best_node)
        new_station = self.init_station(best_node, init_charger_config, self.node_list)
        install_fee = new_station.s_dict["fee"]
        if self.budget - install_fee < 0:
            self.game_over = True
            return 0.0, self.game_over

        # adjust the config
        charger_config = init_charger_config.copy()
        while np.sum(charger_config) < K:
            charger_add_rewards = []
            for i in range(len(INSTALL_FEE)):
                charger_config_ = charger_config.copy()
                charger_config_[i] += 1

                reward = self.new_station_reward(best_node, charger_config_)
                charger_add_rewards.append(reward)

            charger_add_rewards_idx = np.argsort(charger_add_rewards)[::-1]
            pointer = 0
            charger_add_success_flag = False
            while pointer < len(charger_add_rewards_idx):
                add_charger_type = charger_add_rewards_idx[pointer]
                charger_config_ = charger_config.copy()
                charger_config_[add_charger_type] += 1

                new_station = self.init_station(best_node, charger_config_, self.node_list)
                install_fee = new_station.s_dict["fee"]
                if self.budget - install_fee >= 0:
                    charger_config = charger_config_
                    charger_add_success_flag = True
                    break
                pointer += 1

            if not charger_add_success_flag:
                break

        new_station = self.init_station(best_node, charger_config, self.node_list)
        install_fee = new_station.s_dict["fee"]
        self.budget -= install_fee
        self.plan_instance.add_plan(new_station.station)
        self.plan_instance.plan, self.node_list = self.plan_instance.update_node_info(self.plan_instance.plan,
                                                                                      self.node_list,
                                                                                      StationPlacement.node_dict,
                                                                                      StationPlacement.cost_dict)

        return install_fee, self.game_over


    def step_llm_agent(self, llm_agent, history):
        """
        choose a location
        """
        station_list = [s[0][0] for s in self.plan_instance.plan]  # all charging stations
        free_list = [node for node in self.node_list if node[0] not in station_list]  # nodes without stations

        # calculate max potential benefits
        free_list_cov_benefits = []
        free_list_dem_benefits = []
        free_list_dis_benefits = []
        free_list_wat_benefits = []
        free_list_chg_benefits = []
        for node in free_list:
            (_, _, demand_benefit, distance_benefit,
             wait_benefit, charge_benefit) = node_social_benefit(node, self.node_list, self.plan_instance.plan)
            supply_benefit, _, _, _, _, _, _ = self.new_station_benefit(node, is_max=True)
            free_list_cov_benefits.append(supply_benefit)
            free_list_dem_benefits.append(demand_benefit)
            free_list_dis_benefits.append(-distance_benefit)
            free_list_wat_benefits.append(wait_benefit)
            free_list_chg_benefits.append(charge_benefit)

        # get the best location
        free_list_cov_benefits = np.array(free_list_cov_benefits)
        free_list_dis_benefits = np.array(free_list_dis_benefits)
        free_list_wat_benefits = np.array(free_list_wat_benefits)
        free_list_chg_benefits = np.array(free_list_chg_benefits)

        # fill nan with max
        max_wat = np.max(free_list_wat_benefits)
        max_chg = np.max(free_list_chg_benefits)
        free_list_wat_benefits_nan_idx = free_list_wat_benefits == -1
        free_list_chg_benefits_nan_idx = free_list_chg_benefits == -1
        free_list_wat_benefits[free_list_wat_benefits_nan_idx] = max_wat
        free_list_chg_benefits[free_list_chg_benefits_nan_idx] = max_chg

        max_cov, min_cov = np.max(free_list_cov_benefits), np.min(free_list_cov_benefits)
        max_dis, min_dis = np.max(free_list_dis_benefits), np.min(free_list_dis_benefits)
        max_wat, min_wat = np.max(free_list_wat_benefits), np.min(free_list_wat_benefits)
        max_chg, min_chg = np.max(free_list_chg_benefits), np.min(free_list_chg_benefits)

        free_list_cov_benefits_norm = (free_list_cov_benefits - min_cov) / (max_cov - min_cov)
        free_list_dis_benefits_norm = (free_list_dis_benefits - min_dis) / (max_dis - min_dis)
        free_list_wat_benefits_norm = (free_list_wat_benefits - min_wat) / (max_wat - min_wat)
        free_list_chg_benefits_norm = (free_list_chg_benefits - min_chg) / (max_chg - min_chg)

        free_list_benefits = (0.4 * free_list_cov_benefits_norm + 0.2 * free_list_dis_benefits_norm +
                              0.2 * free_list_wat_benefits_norm + 0.2 * free_list_chg_benefits_norm)
        free_list_benefits_indices = np.argsort(-free_list_benefits)[:20]

        # llm agent pipeline
        data_text = get_location_data_text(free_list_benefits_indices, free_list_cov_benefits,
                                           free_list_dem_benefits, free_list_dis_benefits,
                                           free_list_wat_benefits, free_list_chg_benefits)

        # data analysis and decision-making
        answer_option_form = "\"" + "/".join([f"Location {(i+1)}" for i in range(50)]) + "\""

        decision_info = llm_agent.hybrid_decision_making_pipeline([data_text], [answer_option_form])[0]
        try:
            answer_idx = int(decision_info["answer"].split(" ")[-1])
        except:
            answer_idx = "0"
        data_analysis = decision_info["data_analysis"]
        decision_summary = decision_info["summary"]

        # select the best location
        best_node_idx = free_list_benefits_indices[int(answer_idx)]
        best_node = free_list[best_node_idx]
        node_cov, _, node_dem, node_dis, node_wat, node_chg = node_social_benefit(best_node,
                                                                                  self.node_list,
                                                                                  self.plan_instance.plan)
        best_node_info = {
            "id": int(answer_idx),
            "cov": node_cov,
            "dem": node_dem,
            "dis": node_dis,
            "wat": node_wat,
            "chg": node_chg
        }

        """
        Add chargers
        """
        # initiate a config
        config_dict_ = self.config_dict.copy()
        init_charger_config = initial_solution(config_dict_, self.node_list, best_node)
        new_station = self.init_station(best_node, init_charger_config, self.node_list)
        install_fee = new_station.s_dict["fee"]
        if self.budget - install_fee < 0:
            self.game_over = True
            return 0.0, self.game_over

        # adjust the config
        charger_config = init_charger_config.copy()
        while np.sum(charger_config) < K:
            charger_add_rewards = []
            for i in range(len(INSTALL_FEE)):
                charger_config_ = charger_config.copy()
                charger_config_[i] += 1

                reward = self.new_station_reward(best_node, charger_config_)
                charger_add_rewards.append(reward)

            charger_add_rewards_idx = np.argsort(charger_add_rewards)[::-1]
            pointer = 0
            charger_add_success_flag = False
            while pointer < len(charger_add_rewards_idx):
                add_charger_type = charger_add_rewards_idx[pointer]
                charger_config_ = charger_config.copy()
                charger_config_[add_charger_type] += 1

                new_station = self.init_station(best_node, charger_config_, self.node_list)
                install_fee = new_station.s_dict["fee"]
                if self.budget - install_fee >= 0:
                    charger_config = charger_config_
                    charger_add_success_flag = True
                    break
                pointer += 1

            if not charger_add_success_flag:
                break

        new_station = self.init_station(best_node, charger_config, self.node_list)
        install_fee = new_station.s_dict["fee"]
        self.budget -= install_fee
        self.plan_instance.add_plan(new_station.station)
        self.plan_instance.plan, self.node_list = self.plan_instance.update_node_info(self.plan_instance.plan,
                                                                                      self.node_list,
                                                                                      StationPlacement.node_dict,
                                                                                      StationPlacement.cost_dict)

        # llm agent self-reflection
        env_change_text = get_env_change_text(best_node_info, free_list[best_node_idx], self)
        if decision_info["answer"]:
            self_reflection = llm_agent.hybrid_self_reflection_pipeline(
                [data_text], [answer_idx], [decision_summary], [env_change_text], [answer_option_form])[0]
        else:
            self_reflection = None
            
        history.append({
            "data_text": data_text,
            "data_analysis": data_analysis,
            "answer_idx": answer_idx,
            "decision_summary": decision_summary,
            "env_change_text": env_change_text,
            "memory": llm_agent.memory,
            "self_reflection": self_reflection
        })

        return install_fee, self.game_over
    
    def step_manual_action(self, action):
        data_analysis = action["data_analysis"]
        decision_summary = action["summary"]

        # select the best location
        best_node_idx = self.free_list_benefits_indices[int(answer_idx)] 
        best_node = self.free_list[best_node_idx] 
        node_cov, _, node_dem, node_dis, node_wat, node_chg = self.get_node_info(best_node)

        best_node_info = {
            "id": int(answer_idx),
            "cov": node_cov,
            "dem": node_dem,
            "dis": node_dis,
            "wat": node_wat,
            "chg": node_chg
        }

        """
        Add chargers
        """
        # initiate a config
        config_dict_ = self.config_dict.copy()
        init_charger_config = initial_solution(config_dict_, self.node_list, best_node)
        new_station = self.init_station(best_node, init_charger_config, self.node_list)
        install_fee = new_station.s_dict["fee"]

        if self.budget - install_fee < 0:
            self.game_over = True
            return 0.0, self.game_over

        # adjust the config
        charger_config = init_charger_config.copy()
        while np.sum(charger_config) < K:
            charger_add_rewards = []
            for i in range(len(INSTALL_FEE)):
                charger_config_ = charger_config.copy()
                charger_config_[i] += 1

                reward = self.new_station_reward(best_node, charger_config_)
                charger_add_rewards.append(reward)

            charger_add_rewards_idx = np.argsort(charger_add_rewards)[::-1]
            pointer = 0
            charger_add_success_flag = False
            while pointer < len(charger_add_rewards_idx):
                add_charger_type = charger_add_rewards_idx[pointer]
                charger_config_ = charger_config.copy()
                charger_config_[add_charger_type] += 1

                new_station = self.init_station(best_node, charger_config_, self.node_list)
                install_fee = new_station.s_dict["fee"]
                if self.budget - install_fee >= 0:
                    charger_config = charger_config_
                    charger_add_success_flag = True
                    break
                pointer += 1

            if not charger_add_success_flag:
                break

        new_station = self.init_station(best_node, charger_config, self.node_list)
        install_fee = new_station.s_dict["fee"]
        self.budget -= install_fee
        self.plan_instance.add_plan(new_station.station)
        self.plan_instance.plan, self.node_list = self.plan_instance.update_node_info(self.plan_instance.plan,
                                                                                      self.node_list,
                                                                                      StationPlacement.node_dict,
                                                                                      StationPlacement.cost_dict)

        # llm agent self-reflection
        env_change_text = get_env_change_text(best_node_info, free_list[best_node_idx], self)
        if decision_info["answer"]:
            self_reflection = llm_agent.hybrid_self_reflection_pipeline(
                [data_text], [answer_idx], [decision_summary], [env_change_text], [answer_option_form])[0]
        else:
            self_reflection = None
            
        history.append({
            "data_text": data_text,
            "data_analysis": data_analysis,
            "answer_idx": answer_idx,
            "decision_summary": decision_summary,
            "env_change_text": env_change_text,
            "memory": llm_agent.memory,
            "self_reflection": self_reflection
        })

        return install_fee, self.game_over


    def update_station_state(self):
        # === 获取当前建站与未建站节点 ===
        station_node_ids = {s[0][0] for s in self.env.plan_instance.plan}
        node_id_to_idx = {n[0]: i for i, n in enumerate(self.node_list)}

        self.free_list = [node for node in self.node_list if node[0] not in station_node_ids]
        self.station_list = [node for node in self.node_list if node[0] in station_node_ids]

        # === 初始化 free_list 特征 ===
        self.free_list_benefits = []
        self.free_list_cov_benefits = []
        self.free_list_dem_benefits = []
        for node in self.free_list:
            benefit, cov_num, demand = free_node_benefit(node, self.node_list, self.env.plan_instance.plan)
            self.free_list_benefits.append(benefit)
            self.free_list_cov_benefits.append(cov_num)
            self.free_list_dem_benefits.append(demand)

        # === 转换为数组并处理异常值 ===
        self.free_list_benefits = np.array(self.free_list_benefits)
        self.free_list_cov_benefits = np.array(self.free_list_cov_benefits)
        self.free_list_dem_benefits = np.array(self.free_list_dem_benefits)

        # === station_list 特征（用于 C, D, E） ===
        self.station_benefits = []
        self.station_cov = []
        self.station_dem = []
        self.station_neediness = []

        for station_node in self.station_list:
            idx = node_id_to_idx[station_node[0]]
            self.station_node_indices.append(idx)

            benefit, cov_num, demand, neediness, wait_time, charging_time = station_node_benefit(station_node, self.node_list, self.env.plan_instance.plan)

            self.station_benefits.append(benefit)
            self.station_cov.append(cov_num)
            self.station_dem.append(demand)
            self.station_neediness.append(neediness)

        self.station_benefits = np.array(self.station_benefits)
        self.station_cov = np.array(self.station_cov)
        self.station_dem = np.array(self.station_dem)
        self.station_neediness = np.array(self.station_neediness)

        # === 示例：如果你需要自定义每个 station 的 benefit 排序，用下面这个生成（否则注释掉） ===
        self.station_benefit_scores = self.station_cov_benefits + self.station_dem_benefits  # or your own formula

    
    def get_node_info(self, node):
        """
        Get the social benefits of a node.
        """
        node_cov, _, node_dem, node_dis, node_wat, node_chg = self.free_list_benefits[node[0]], 0,\
            self.free_list_dem_benefits[node[0]], self.free_list_dis_benefits[node[0]], self.free_list_wat_benefits[node[0]], \
                self.free_list_chg_benefits[node[0]]
        return node_cov, node_dem, -node_dis, node_wat, node_chg
 

    def _get_observation(self, top_k=5):
        """
        构建用于支持动作 A–E 的 observation。
        包括：
            A: 覆盖度最低的未建站点
            B: 需求最高的未建站点
            C: 覆盖度最低的已建站点
            D: 需求最高的已建站点
            E: 效益最差的已建站点
        """

        self.update_station_state()

        obs_sections = []

        # A: coverage 最低的未建站节点
        a_indices = np.argsort(self.free_list_cov_benefits)[:top_k]
        a_text = get_location_data_text(
            a_indices,
            self.free_list_cov_benefits,
            self.free_list_dem_benefits,
            self.free_list_dis_benefits,
            self.free_list_wat_benefits,
            self.free_list_chg_benefits,
            prefix="Action A candidates (Create by Coverage):"
        )
        obs_sections.append(a_text)

        # B: demand 最大的未建站节点
        b_indices = np.argsort(-np.array(self.free_list_dem_benefits))[:top_k]
        b_text = get_location_data_text(
            b_indices,
            self.free_list_cov_benefits,
            self.free_list_dem_benefits,
            self.free_list_dis_benefits,
            self.free_list_wat_benefits,
            self.free_list_chg_benefits,
            prefix="Action B candidates (Create by Demand):"
        )
        obs_sections.append(b_text)

        # 已建站的节点集合
        built_node_ids = [s[0][0] for s in self.env.plan_instance.plan]
        node_id_to_idx = {n[0]: i for i, n in enumerate(self.node_list)}
        station_indices = [node_id_to_idx[nid] for nid in built_node_ids if nid in node_id_to_idx]

        # 过滤构造 benefit 向量（用于 Action E）
        station_cov = [self.free_list_cov_benefits[i] for i in station_indices]
        station_dem = [self.free_list_dem_benefits[i] for i in station_indices]
        station_benefit = self.station_benefit_scores if hasattr(self, "station_benefit_scores") else [self.compute_station_benefit(i) for i in station_indices]

        # C: coverage 最低的已建站节点
        c_indices = np.argsort(station_cov)[:top_k]
        c_text = get_location_data_text(
            [station_indices[i] for i in c_indices],
            self.free_list_cov_benefits,
            self.free_list_dem_benefits,
            self.free_list_dis_benefits,
            self.free_list_wat_benefits,
            self.free_list_chg_benefits,
            prefix="Action C candidates (Add by Coverage):"
        )
        obs_sections.append(c_text)

        # D: demand 最高的已建站节点
        d_indices = np.argsort(-np.array(station_dem))[:top_k]
        d_text = get_location_data_text(
            [station_indices[i] for i in d_indices],
            self.free_list_cov_benefits,
            self.free_list_dem_benefits,
            self.free_list_dis_benefits,
            self.free_list_wat_benefits,
            self.free_list_chg_benefits,
            prefix="Action D candidates (Add by Demand):"
        )
        obs_sections.append(d_text)

        # E: benefit 最小的已建站节点
        e_indices = np.argsort(station_benefit)[:top_k]
        e_text = get_location_data_text(
            [station_indices[i] for i in e_indices],
            self.free_list_cov_benefits,
            self.free_list_dem_benefits,
            self.free_list_dis_benefits,
            self.free_list_wat_benefits,
            self.free_list_chg_benefits,
            prefix="Action E candidates (Relocate from low-benefit):"
        )
        obs_sections.append(e_text)

        return "\n\n".join(obs_sections)


    def render(self, mode='human', close=False):
        """
        Render the environment to the screen
        """
        print(f'Score is: {self.best_score}')
        print(f'Number of stations in charging plan: {len(self.plan_instance.plan)}')
        return self.best_node_list, self.best_plan


if __name__ == '__main__':
    location = "Toy_Example"
    graph_file = "Graph/" + location + "/" + location + ".graphml"
    node_file = "Graph/" + location + "/nodes_extended_" + location + ".txt"
    plan_file = "Graph/" + location + "/existingplan_" + location + ".pkl"

    env = StationPlacement(graph_file, node_file, plan_file)

    check_env(env)
