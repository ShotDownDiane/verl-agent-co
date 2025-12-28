# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict, Union, Any
from collections import defaultdict
import torch
import numpy as np
from functools import partial
import os
from agent_system.environments.prompts import *
from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.format_reward import compute_format_reward
from agent_system.memory import SimpleMemory, SearchMemory
from omegaconf import OmegaConf
from types import SimpleNamespace
from scipy.spatial import cKDTree

def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if 'extra.gamefile' in info:
            gamefile.append(info['extra.gamefile'])
        else:
            gamefile.append(None)
    return gamefile

def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if 'extra.gamefile' in infos[i]:
            infos[i]['extra.gamefile'] = gamefile[i]
        else:
            infos[i]['extra.gamefile'] = None
    return infos

def _to_container(obj, resolve=True):
    """Safely convert SimpleNamespace or OmegaConf object to dict/list.
    
    This helper function allows make_envs to work with both OmegaConf configs
    (used in production) and SimpleNamespace objects (used in manual testing).
    """
    if isinstance(obj, SimpleNamespace):
        # Convert SimpleNamespace to dict recursively
        if hasattr(obj, '__dict__'):
            result = {}
            for key, value in obj.__dict__.items():
                if isinstance(value, SimpleNamespace):
                    result[key] = _to_container(value, resolve=resolve)
                elif isinstance(value, (list, tuple)):
                    result[key] = [
                        _to_container(item, resolve=resolve) if isinstance(item, SimpleNamespace) else item
                        for item in value
                    ]
                else:
                    result[key] = value
            return result
        else:
            return {}
    else:
        # Use OmegaConf.to_container for OmegaConf objects
        return OmegaConf.to_container(obj, resolve=resolve)


def calculate_k_operators_with_lowest_processing_time(instance: np.ndarray, k: int) -> List[List[Tuple[int, int, float]]]:
    """
    Calculates the k operators with the lowest processing time for each job.
    This matches the implementation in LLMCoSolver.

    Parameters
    ----------
    instance : np.ndarray
        A JSSP instance with shape (num_jobs, 2*num_ops) where each operation is (machine, processing_time).
    k : int
        Number of operators to be calculated.

    Returns
    -------
    List[List[Tuple[int, int, float]]]
        A list of k operators with the lowest processing time for each job.
        Each tuple contains (operator_index, machine_index, processing_time).
    """
    num_jobs, num_columns = instance.shape
    num_machines = num_columns // 2
    job_operators = {job: [] for job in range(num_jobs)}

    # Iterate through each job and operator
    for job_idx in range(num_jobs):
        for op_idx in range(num_machines):
            machine_idx = int(instance[job_idx, op_idx * 2])
            processing_time = float(instance[job_idx, op_idx * 2 + 1])
            job_operators[job_idx].append((op_idx, machine_idx, processing_time))

    # Sort operators for each job by processing time and select the k operators with the lowest processing time
    result = []
    for job in range(num_jobs):
        sorted_operators = sorted(job_operators[job], key=lambda x: x[2])  # Sort by processing time
        result.append(sorted_operators[:k])  # Take the k operators with the lowest processing time

    return result


def calculate_top_k_nearest_nodes(nodes: np.ndarray, k: int = 2) -> List[List[Tuple[int, float]]]:
    """
    For each node, calculate its top k nearest neighbors using a k-d tree.
    This matches the implementation in LLMCoSolver.

    Parameters
    ----------
    nodes : np.ndarray
        Coordinates of the nodes. Shape: (N, 2).
    k : int, optional
        Number of nearest neighbors to find for each node. Default is 2.

    Returns
    -------
    List[List[Tuple[int, float]]]
        A list of length N, where each element is a list of k tuples (neighbor_index, distance).
    """
    kdtree = cKDTree(nodes)
    top_k_nearest_nodes = []
    for node in nodes:
        distances, indices = kdtree.query(node, k + 1)  # k+1 to include the node itself
        # Handle scalar vs array return from kdtree.query
        if k == 0:
            # Edge case: k=0 means no neighbors
            neighbors = []
        elif k == 1:
            # When k=1, kdtree.query may return scalars
            if np.isscalar(distances):
                # Only one result (the node itself), skip it
                neighbors = []
            else:
                # Exclude the node itself (first index)
                if len(distances) > 1:
                    neighbors = [(int(indices[1]), float(distances[1]))]
                else:
                    neighbors = []
        else:
            # Exclude the node itself (first index)
            distances = distances[1:]
            indices = indices[1:]
            neighbors = [(int(idx), float(dist)) for idx, dist in zip(indices, distances)]
        top_k_nearest_nodes.append(neighbors)
    return top_k_nearest_nodes


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy()
        }
        
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({
            "search": actions,
            "information": next_obs,
        })

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy()
        }
        
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self,
        text_obs: List[str],
        init: bool = False
    ) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search"
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i]
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs


class RL4COSchedulingEnvironmentManager(EnvironmentManagerBase):
    """EnvironmentManager for RL4CO scheduling envs (JSSP / FFSP)."""

    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.sched_env_name = getattr(
            config.env.rl4co_scheduling, "env_name", "jssp"
        ).lower()
        # Format reward configuration
        self.use_format_reward = getattr(config.env.rl4co_scheduling, "use_format_reward", False)
        self.format_reward_weight = getattr(config.env.rl4co_scheduling, "format_reward_weight", 0.05)
        self.feasibility_reward_weight = getattr(config.env.rl4co_scheduling, "feasibility_reward_weight", 0.15)
        # Conditional reward mechanism parameters
        self.use_conditional_reward = getattr(config.env.rl4co_scheduling, "use_conditional_reward", True)
        self.feasibility_threshold = getattr(config.env.rl4co_scheduling, "feasibility_threshold", 0.9)
        self.normalize_env_reward = getattr(config.env.rl4co_scheduling, "normalize_env_reward", True)
        self.env_reward_range = getattr(config.env.rl4co_scheduling, "env_reward_range", None)
        self.fixed_scale_reference = getattr(config.env.rl4co_scheduling, "fixed_scale_reference", None)
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Dict[str, Any]:
        td, infos = self.envs.reset()
        batch_size = td.batch_size[0] if td.batch_size else len(infos)
        self.memory.reset(batch_size=batch_size)
        self.pre_text_obs = [""] * batch_size

        text_obs = self.build_text_obs(td, init=True)
        observations = {"text": text_obs, "image": None, "anchor": text_obs}
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_td, env_rewards, dones, infos = self.envs.step(actions)
        # update stored TensorDict so external agents can inspect latest instance
        self.current_td = next_td

        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        text_obs = self.build_text_obs(next_td, init=False)
        self.pre_text_obs = text_obs

        env_rewards = to_numpy(env_rewards)

        # Compute format reward if enabled (for step-by-step mode)
        format_bonuses = None
        if self.use_format_reward:
            format_bonuses = np.array([1.0 if v == 1 else 0.0 for v in valids], dtype=np.float32)
            final_rewards = env_rewards + self.format_reward_weight * format_bonuses
            rewards = final_rewards
        else:
            rewards = env_rewards

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
            if self.use_format_reward and format_bonuses is not None:
                info["format_bonus"] = float(format_bonuses[i])
                info["env_reward"] = float(env_rewards[i])

        dones = to_numpy(dones) if dones is not None else None

        next_observations = {"text": text_obs, "image": None, "anchor": text_obs}
        return next_observations, rewards, dones, infos

    def _format_jssp_obs_single(self, td, idx: int, init: bool) -> str:
        # Try to extract JSSP instance from TensorDict
        # rl4co JSSP environment may store instance data in different keys
        # We'll try to reconstruct from available keys or use a fallback
        
        num_jobs = 0
        num_machines = 0
        num_ops = 0
        
        # Try to get dimensions from TensorDict
        if "next_op" in td.keys():
            num_jobs = td["next_op"][idx].shape[0]
        if "ma_assignment" in td.keys():
            ma_assign = td["ma_assignment"][idx]
            if ma_assign.numel() > 0:
                num_machines = int(ma_assign.max().item()) + 1
                # Estimate num_ops from ma_assignment shape
                if len(ma_assign.shape) > 1:
                    num_ops = ma_assign.shape[-1]
                else:
                    num_ops = num_machines  # fallback
        
        # Try to get instance data from generator or other sources
        instance_data = None
        base_env = getattr(self.envs, "base_env", None) or getattr(getattr(self.envs, "inner_env", None), "base_env", None)
        if base_env is not None and hasattr(base_env, "generator") and hasattr(base_env.generator, "data"):
            # Try to get instance from generator
            gen_data = base_env.generator.data
            if gen_data is not None and len(gen_data) > idx:
                instance_data = gen_data[idx]
        
        # If we have instance data, format it according to LLMCoSolver
        if instance_data is not None:
            instance_np = instance_data.cpu().numpy() if isinstance(instance_data, torch.Tensor) else instance_data
            k = getattr(self.config.env.rl4co_scheduling, "k", 2)
            k_operations_with_lowest_processing_time = calculate_k_operators_with_lowest_processing_time(instance_np, k)
            
            # Format job descriptions (matching LLMCoSolver format exactly)
            job_descriptions = []
            for job_idx in range(num_jobs):
                job_info = instance_np[job_idx].tolist()
                job_info_reformated = []
                for op_idx in range(num_ops):
                    job_info_reformated.append((int(job_info[2 * op_idx]), int(job_info[2 * op_idx + 1])))
                lowest_str = [f"{op_info[0]}: ({op_info[1]}, {op_info[2]})" for op_info in k_operations_with_lowest_processing_time[job_idx]]
                job_desc = (
                    f"Job {job_idx}, machines and processing times for operations: {job_info_reformated}, "
                    f"operators with lowest processing time: {lowest_str}; "
                ).replace("'", "")
                job_descriptions.append(job_desc)
            
            job_descriptions_str = "".join(job_descriptions)
            # Replace last semicolon with period (matching LLMCoSolver)
            job_descriptions_str = ".".join(job_descriptions_str.rsplit(";", 1))
        else:
            # Fallback: create a simplified description
            job_descriptions_str = f"Job information not available. There are {num_jobs} jobs and {num_machines} machines."

        obs = RL4CO_JSSP_TEMPLATE_NO_HIS.format(
            num_jobs=num_jobs,
            num_machines=num_machines,
            num_ops=num_ops,
            job_descriptions=job_descriptions_str,
        )

        return obs

    def _format_ffsp_obs_single(self, td, idx: int, init: bool) -> str:
        # FFSPEnv exposes high-level schedule state; we focus on counts.
        job_loc = td["job_location"][idx]  # shape [num_jobs+1] (last may be dummy)
        num_jobs = job_loc.shape[-1] - 1

        # We don't have direct per-stage times here; summarize generically.
        num_stages = int(td.get("num_stage", torch.tensor(0))[0].item()) if "num_stage" in td.keys() else "unknown"
        machines_per_stage = []

        job_stage_times = f"There are {num_jobs} jobs and {num_stages} stages. Each job must be processed at each stage in order."

        current_time = float(td.get("time_idx", torch.tensor(0.0))[idx].item()) if "time_idx" in td.keys() else 0.0

        if init or self.config.env.history_length <= 0:
            obs = RL4CO_FFSP_TEMPLATE_NO_HIS.format(
                num_stages=num_stages,
                machines_per_stage=machines_per_stage,
                job_stage_times=job_stage_times,
            )
        else:
            action_mask = td["action_mask"][idx].bool()
            admissible_indices = [i for i, v in enumerate(action_mask.tolist()) if v]
            admissible_str = "\n".join(str(i) for i in admissible_indices)

            memory_ctx, valid_lens = self.memory.fetch(
                self.config.env.history_length,
                obs_key="text_obs",
                action_key="action",
            )
            action_history = memory_ctx[idx]
            history_length = valid_lens[idx]

            obs = RL4CO_FFSP_TEMPLATE.format(
                num_stages=num_stages,
                machines_per_stage=machines_per_stage,
                job_stage_times=job_stage_times,
                current_time=current_time,
                admissible_actions=admissible_str,
                history_length=history_length,
                action_history=action_history,
            )
        return obs

    def build_text_obs(self, td, init: bool = False) -> List[str]:
        postprocess_text_obs: List[str] = []
        batch_size = td.batch_size[0]
        env_name = getattr(self.config.env.rl4co_scheduling, "env_name", "jssp").lower()

        for i in range(batch_size):
            if env_name == "jssp":
                obs = self._format_jssp_obs_single(td, i, init=init)
            elif env_name == "ffsp":
                obs = self._format_ffsp_obs_single(td, i, init=init)
            else:
                obs = self._format_jssp_obs_single(td, i, init=init)
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask


class ML4COKitSchedulingEnvironmentManager(RL4COSchedulingEnvironmentManager):
    """One-shot manager for ml4co-kit scheduling environments (JSSP/PFSP)."""

    def __init__(self, envs, projection_f, config, env_name: str = "jssp"):
        self.memory = SimpleMemory()
        self.ml4co_env_name = env_name.lower()
        super().__init__(envs, projection_f, config)
        cfg = getattr(config.env, "ml4co_kit", getattr(config.env, "rl4co_scheduling", SimpleNamespace()))
        self.sched_env_name = self.ml4co_env_name
        self.use_format_reward = getattr(cfg, "use_format_reward", self.use_format_reward)
        self.format_reward_weight = getattr(cfg, "format_reward_weight", self.format_reward_weight)
        self.feasibility_reward_weight = getattr(cfg, "feasibility_reward_weight", self.feasibility_reward_weight)
        self.use_conditional_reward = getattr(cfg, "use_conditional_reward", self.use_conditional_reward)
        self.feasibility_threshold = getattr(cfg, "feasibility_threshold", self.feasibility_threshold)
        self.normalize_env_reward = getattr(cfg, "normalize_env_reward", self.normalize_env_reward)
        self.env_reward_range = getattr(cfg, "env_reward_range", self.env_reward_range)
        self.fixed_scale_reference = getattr(cfg, "fixed_scale_reference", self.fixed_scale_reference)
        self.pre_text_obs: List[str] = []

    def reset(self, kwargs):
        observations, infos = super().reset(kwargs)
        self.pre_text_obs = observations["text"]
        return observations, infos

    def build_text_obs(self, td, init: bool = False) -> List[str]:
        """Build text observations using the correct env_name for ML4CO-Kit."""
        postprocess_text_obs: List[str] = []
        batch_size = td.batch_size[0]
        env_name = self.sched_env_name

        for i in range(batch_size):
            if env_name == "jssp":
                obs = self._format_jssp_obs_single(td, i, init=init)
            elif env_name == "pfsp" or env_name == "ffsp":
                obs = self._format_ffsp_obs_single(td, i, init=init)
            else:
                obs = self._format_jssp_obs_single(td, i, init=init)
            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_td, env_rewards, dones, infos = self.envs.step(actions)
        env_rewards_np = to_numpy(env_rewards)

        rewards = env_rewards_np
        format_info: Dict[str, Any] = {}
        if self.use_format_reward:
            num_jobs = None
            if "next_op" in next_td.keys():
                num_jobs = next_td["next_op"].shape[0]
            elif "job_location" in next_td.keys():
                num_jobs = next_td["job_location"].shape[-1] - 1
            final_rewards, format_info = compute_format_reward(
                valids=valids,
                actions=actions,
                env_rewards=env_rewards_np,
                env_name=self.sched_env_name,
                env_type="scheduling",
                format_reward_weight=self.format_reward_weight,
                feasibility_reward_weight=self.feasibility_reward_weight,
                num_jobs=num_jobs,
                use_conditional_reward=self.use_conditional_reward,
                feasibility_threshold=self.feasibility_threshold,
                normalize_env_reward=self.normalize_env_reward,
                env_reward_range=self.env_reward_range,
                fixed_scale_reference=self.fixed_scale_reference,
            )
            rewards = final_rewards

        final_infos: List[Dict[str, Any]] = []
        for i, info in enumerate(infos):
            info_dict = dict(info) if info else {}
            info_dict["is_action_valid"] = to_numpy(valids[i])
            if self.use_format_reward:
                info_dict["format_bonus"] = float(format_info.get("format_bonuses", [0])[i])
                info_dict["feasibility_bonus"] = float(format_info.get("feasibility_bonuses", [0])[i])
                info_dict["env_reward"] = float(format_info.get("env_rewards", env_rewards_np)[i])
                if "format_reward" in format_info:
                    info_dict["format_reward"] = float(format_info["format_reward"][i])
                    info_dict["feasibility_reward"] = float(format_info["feasibility_reward"][i])
                    info_dict["scaled_env_reward"] = float(format_info["env_reward"][i])
                    info_dict["env_reward_weight"] = float(format_info["env_reward_weight"][i])
                    info_dict["meets_feasibility_threshold"] = bool(format_info.get("feasibility_mask", [False])[i])
            final_infos.append(info_dict)

        dones_np = to_numpy(dones) if dones is not None else None
        next_observations = {"text": self.pre_text_obs, "image": None, "anchor": self.pre_text_obs}
        return next_observations, rewards, dones_np, final_infos


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        # initialize the history buffer
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands, init=True)
        return {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions, self.envs.get_admissible_commands)
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': image_obs, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    
    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find('Your task is to: ')
            
            if task_start != -1:
                self.tasks.append(obs[task_start + len('Your task is to: '):].strip())
            else:
                raise ValueError("Task description not found in text observation.")
        

    def build_text_obs(self, text_obs: List[str], admissible_actions: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(f"'{s}'" for s in admissible_actions[i] if s != 'help')

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                success['success_rate'].append(won_value)
                
                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]
        
        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }
    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == 'rgb_array'
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            observations = {
                'text': self.build_text_obs(infos, init=True), 
                'image': obs,   
                'anchor': obs
            }
        else:
            self.pre_text_obs = obs
            observations = {
                'text': self.build_text_obs(infos, obs, init=True),
                'image': None,
                'anchor': obs
            }
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        self.memory.store({'text_obs': self.pre_text_obs, 'action': [self.ACTION_LOOKUP[act] for act in actions]})
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode='tiny_rgb_array')
            next_observations = {
                'text': self.build_text_obs(infos),  
                'image': next_obs,
                'anchor': next_obs 
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                'text': self.build_text_obs(infos, next_obs),  
                'image': None, 
                'anchor': next_obs 
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos, text_obs: List[str]=None, init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = SOKOBAN_VISUAL_TEMPLATE if self.is_multi_modal \
                 else SOKOBAN_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(infos), 'image': obs, 'anchor': obs.copy()}
        
        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)
        
        # add text observation to next_observations
        next_observations['text'] = self.build_text_obs(infos)
        next_observations['anchor'] = next_observations['image'].copy()

        return next_observations, rewards, dones, infos


    def build_text_obs(self, infos: Tuple[Dict]=None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if 'ezpoints' in self.config.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif 'points24' in self.config.env_name.lower():
                text_formula = ''.join(str(element) for element in infos[i]['Formula']) if infos[i] is not None else ''
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif 'numberline' in self.config.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        # infos = [None] * self.envs.num_envs
        observations = {'text': self.build_text_obs(obs, infos, init=True), 
                        'image': None, 
                        'anchor': obs.copy()
                        }
        self.pre_text_obs = obs
        self.memory.reset(batch_size = len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)

        next_obs = self.format_obs(next_obs)

        self.memory.store({'text_obs': self.pre_text_obs, 'action': actions})
        self.pre_text_obs = next_obs

        next_observations = {
            'text': self.build_text_obs(next_obs, infos),
            'image': None,
            'anchor': next_obs.copy()
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1]=='Instruction:'
            tasks.append(parts[2])
        return tasks
    
    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index+1:])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs
    
    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions
            
    def build_text_obs(self, text_obs: List[str], infos: List[List[str]], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                    self.config.env.history_length,
                    obs_key="text_obs",
                    action_key="action")
            
        for i in range(len(text_obs)):
            
            available_actions = self.format_avail_actions(infos[i]['available_actions'])
            reformatted_available_actions = "\n".join(f"'{s}'," for s in available_actions)

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item['active_masks']:
                info = total_infos[batch_idx][i]
                won_value = float(info['won'])
                score_value = float(info['task_score'])
                success['success_rate'].append(won_value)
                success['webshop_task_score (not success_rate)'].append(score_value)
                return

class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
    
    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()
        
        self.supervisors = [info['supervisor'] for info in infos]
        self.memory.reset(batch_size = len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {'text': full_text_obs, 'image': None, 'anchor': text_obs}, infos
    
    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({'text_obs': text_obs, 'action': actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)
        
        # add action_valid to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])

        next_observations = {'text': full_text_obs, 'image': None, 'anchor': text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos
    

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                    )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length:]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"
                
                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                        supervisor_first_name=self.supervisors[i]['first_name'],
                        supervisor_last_name=self.supervisors[i]['last_name'],
                        supervisor_email=self.supervisors[i]['email'],
                        supervisor_phone_number=self.supervisors[i]['phone_number'],
                        task_description=self.tasks[i],
                        step_count=len(self.memory[i]),
                        history_length=valid_history_length,
                        action_history=action_history.strip(),
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs


class RouteEnvironmentManager(EnvironmentManagerBase):
    """EnvironmentManager for RL4CO routing envs (starting with TSP).
    """
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self.rl4co_env_name = getattr(config.env.rl4co, "env_name", "tsp").lower() if hasattr(config, "env") and getattr(config.env, "rl4co", None) else "tsp"
        self.return_topk_options = config.return_topk_options
        # Format reward configuration (optional)
        rl4co_cfg = getattr(config.env, "rl4co", {}) if hasattr(config, "env") else {}
        self.use_format_reward = getattr(rl4co_cfg, "use_format_reward", True)
        self.format_reward_weight = getattr(rl4co_cfg, "format_reward_weight", 0.05)
        self.feasibility_reward_weight = getattr(rl4co_cfg, "feasibility_reward_weight", 0.15)
        self.use_conditional_reward = getattr(rl4co_cfg, "use_conditional_reward", True)
        self.feasibility_threshold = getattr(rl4co_cfg, "feasibility_threshold", 0.9)
        self.normalize_env_reward = getattr(rl4co_cfg, "normalize_env_reward", True)
        self.env_reward_range = getattr(rl4co_cfg, "env_reward_range", None)
        self.fixed_scale_reference = getattr(rl4co_cfg, "fixed_scale_reference", None)

        super().__init__(envs, projection_f, config)

    def reset(self, kwargs=None) -> Dict[str, Any]:
        # Some env implementations accept kwargs, others do not.
        try:
            res = self.envs.reset(kwargs=kwargs) if kwargs is not None else self.envs.reset()
        except TypeError:
            res = self.envs.reset()

        # Accept different return signatures:
        # (text_obs_list, image_obs_list, infos) or (text_obs_list, infos)
        if isinstance(res, tuple) and len(res) == 3:
            text_obs_list, image_obs_list, infos = res
        elif isinstance(res, tuple) and len(res) == 2:
            text_obs_list, infos = res
            image_obs_list = None
        else:
            # Fallback: treat whole as text_obs
            text_obs_list = res
            image_obs_list = None
            infos = [{} for _ in range(len(text_obs_list) if hasattr(text_obs_list, "__len__") else 1)]

        batch_size = len(text_obs_list) if hasattr(text_obs_list, "__len__") else 1
        self.memory.reset(batch_size=batch_size)
        self.pre_text_obs = [""] * batch_size

        text_obs_list = self.build_text_obs(text_obs_list)
        observations = {"text": text_obs_list, "image": image_obs_list, "anchor": text_obs_list}
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        res = self.envs.step(actions)

        # Accept multiple possible signatures:
        # (text_obs_list, image_obs_list, rewards, dones, infos)
        # (text_obs_list, rewards, dones, infos)
        if isinstance(res, tuple) and len(res) == 5:
            next_text_obs, image_obs_list, env_rewards, dones, infos = res
        elif isinstance(res, tuple) and len(res) == 4:
            next_text_obs, env_rewards, dones, infos = res
            image_obs_list = None
        else:
            # Unexpected shape
            raise ValueError("Unexpected return shape from routing envs.step")

        # Store history and update textual observations
        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        text_obs = next_text_obs
        self.pre_text_obs = text_obs

        env_rewards = to_numpy(env_rewards)

        # Compute format reward if enabled
        format_bonuses = None
        if self.use_format_reward:
            format_bonuses = np.array([1.0 if v == 1 else 0.0 for v in valids], dtype=np.float32)
            final_rewards = env_rewards + self.format_reward_weight * format_bonuses
            rewards = final_rewards
        else:
            rewards = env_rewards

        # Enrich infos
        for i, info in enumerate(infos):
            if info is None:
                infos[i] = {}
                info = infos[i]
            info["is_action_valid"] = to_numpy(valids[i])
            if self.use_format_reward and format_bonuses is not None:
                info["format_bonus"] = float(format_bonuses[i])
                info["env_reward"] = float(env_rewards[i])

        dones = to_numpy(dones) if dones is not None else None
        text_obs = self.build_text_obs(text_obs)
        next_observations = {"text": text_obs, "image": image_obs_list, "anchor": text_obs}
        return next_observations, rewards, dones, infos
    
    def build_text_obs(self, text_obs) -> List[str]:
        postprocess_text_obs = []
        Template = RL4CO_TSP_TEMPLATE if self.return_topk_options else RL4CO_TSP_TEMPLATE_NO_HIS
        for i in range(len(text_obs)):
            obs = Template.format(
                    text_obs = text_obs[i]
                )
            postprocess_text_obs.append(obs)
        return postprocess_text_obs
        

def make_envs(config):
    """
    Create enviroments 
    """ 
    group_n = config.group_n

    if "search" in config.env_name.lower():
        from agent_system.environments.env_package.search import build_search_envs, search_projection
        _envs = build_search_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_config=config.env)
        _val_envs = build_search_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_config=config.env)

        projection_f = partial(search_projection)
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env_name.lower():
        from agent_system.environments.env_package.gym_cards import build_gymcards_envs, gym_projection
        _envs = build_gymcards_envs(env_name=config.env_name, seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True)
        _val_envs = build_gymcards_envs(env_name=config.env_name, seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False)
        
        projection_f = partial(gym_projection, env_name=config.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env_name.lower():
        from agent_system.environments.env_package.alfworld import build_alfworld_envs, alfworld_projection
        if config.env_name == 'alfworld/AlfredThorEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        elif config.env_name == 'alfworld/AlfredTWEnv':
            alf_config_path = os.path.join(os.path.dirname(__file__), 'env_package/alfworld/configs/config_tw.yaml')
        else:
            raise ValueError(f"Unsupported environment: {config.env_name}")

        env_kwargs = {
            'eval_dataset': config.env.alfworld.eval_dataset, # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(alf_config_path, config.env.seed, config.data.train_batch_size, group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_alfworld_envs(alf_config_path, config.env.seed + 1000, config.data.val_batch_size, 1, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env_name.lower():
        from agent_system.environments.env_package.sokoban import build_sokoban_envs, sokoban_projection
        env_kwargs = {
            'dim_room': config.env.sokoban.dim_room,
            'num_boxes': config.env.sokoban.num_boxes,
            'max_steps': config.env.max_steps,
            'search_depth': config.env.sokoban.search_depth
        }
        _envs = build_sokoban_envs(config.env.seed, config.data.train_batch_size, group_n, mode=config.env.sokoban.mode, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_sokoban_envs(config.env.seed + 1000, config.data.val_batch_size, 1, mode=config.env.sokoban.mode, is_train=False, env_kwargs=env_kwargs)
        
        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env_name.lower():
        from agent_system.environments.env_package.webshop import build_webshop_envs, webshop_projection
        if config.env.webshop.use_small:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle_1000.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2_1000.json')
        else:
            file_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_shuffle.json')
            attr_path = os.path.join(os.path.dirname(__file__), 'env_package/webshop/webshop/data/items_ins_v2.json')
        env_kwargs = {
                    'observation_mode': 'text', 
                    'num_products': None, 
                    'human_goals': config.env.webshop.human_goals,
                    'file_path': file_path,
                    'attr_path': attr_path
                    }
        _envs = build_webshop_envs(seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, is_train=True, env_kwargs=env_kwargs)
        _val_envs = build_webshop_envs(seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, is_train=False, env_kwargs=env_kwargs)

        projection_f = partial(webshop_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time
        time.sleep((config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1) # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env_name.lower():
        from agent_system.environments.env_package.appworld import build_appworld_envs, appworld_projection
        _envs = build_appworld_envs(dataset_name='train', seed=config.env.seed, env_num=config.data.train_batch_size, group_n=group_n, start_server_id=0)
        _val_envs = build_appworld_envs(dataset_name='test_normal', seed=config.env.seed + 1000, env_num=config.data.val_batch_size, group_n=1, start_server_id=config.data.train_batch_size*group_n)
        
        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif config.env_name.lower().startswith("ml4co-kit/"):
        from agent_system.environments.env_package.ml4co_kit import (
            build_ml4cokit_routing_envs,
            build_ml4cokit_scheduling_envs,
            ml4cokit_projection,
            ml4cokit_scheduling_projection,
        )

        sub_env = config.env_name.lower().split("/", 1)[1]
        if sub_env in ("tsp", "cvrp", "op"):
            ml4co_cfg = getattr(config.env, "ml4co_kit", getattr(config.env, "rl4co", {}))
            device = getattr(ml4co_cfg, "device", "cpu")
            generator_params = _to_container(getattr(ml4co_cfg, "generator_params", {}), resolve=True)
            rl4co_kwargs = _to_container(getattr(ml4co_cfg, "rl4co_kwargs", {}), resolve=True)

            _envs = build_ml4cokit_routing_envs(
                env_name=sub_env,
                seed=config.seed,
                env_num=config.train_batch_size,
                group_n=group_n,
                device=device,
                generator_params=generator_params,
                rl4co_kwargs=rl4co_kwargs,
            )
            _val_envs = build_ml4cokit_routing_envs(
                env_name=sub_env,
                seed=config.env.seed + 1000,
                env_num=config.data.val_batch_size,
                group_n=1,
                device=device,
                generator_params=generator_params,
                rl4co_kwargs=rl4co_kwargs,
            )

            projection_f = partial(ml4cokit_projection, env_name=sub_env)
            envs = ML4COKitRoutingEnvironmentManager(_envs, projection_f, config, env_name=sub_env)
            val_envs = ML4COKitRoutingEnvironmentManager(_val_envs, projection_f, config, env_name=sub_env)
            return envs, val_envs
        elif sub_env in ("jssp", "pfsp"):
            ml4co_cfg = getattr(config.env, "ml4co_kit", getattr(config.env, "rl4co_scheduling", {}))
            device = getattr(ml4co_cfg, "device", "cpu")
            generator_params = _to_container(getattr(ml4co_cfg, "generator_params", {}), resolve=True)
            rl4co_kwargs = _to_container(getattr(ml4co_cfg, "rl4co_kwargs", {}), resolve=True)

            _envs = build_ml4cokit_scheduling_envs(
                env_name=sub_env,
                seed=config.env.seed,
                env_num=config.data.train_batch_size,
                group_n=group_n,
                device=device,
                generator_params=generator_params,
                rl4co_kwargs=rl4co_kwargs,
            )
            _val_envs = build_ml4cokit_scheduling_envs(
                env_name=sub_env,
                seed=config.env.seed + 1000,
                env_num=config.data.val_batch_size,
                group_n=1,
                device=device,
                generator_params=generator_params,
                rl4co_kwargs=rl4co_kwargs,
            )

            projection_f = partial(ml4cokit_scheduling_projection, env_name=sub_env)
            envs = ML4COKitSchedulingEnvironmentManager(_envs, projection_f, config, env_name=sub_env)
            val_envs = ML4COKitSchedulingEnvironmentManager(_val_envs, projection_f, config, env_name=sub_env)
            return envs, val_envs
        else:
            raise ValueError(f"Unsupported ml4co-kit env: {sub_env}")
    elif "rl4co_scheduling" in config.env_name.lower():
        from agent_system.environments.env_package.rl4co_scheduling import (
            build_rl4co_scheduling_envs,
            rl4co_scheduling_projection,
        )

        sched_env_name = config.env_name
        sched_device = config.device
        generator_params = config.generator_params
        rl4co_kwargs = config.rl4co_kwargs
        return_topk_options = config.return_topk_options

        _envs = build_rl4co_scheduling_envs(
            env_name=sched_env_name,
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            device=sched_device,
            generator_params=generator_params,
            rl4co_kwargs=rl4co_kwargs,
            return_topk_options=return_topk_options,
        )
        _val_envs = build_rl4co_scheduling_envs(
            env_name=sched_env_name,
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            device=sched_device,
            generator_params=generator_params,
            rl4co_kwargs=rl4co_kwargs,
            return_topk_options=return_topk_options,
        )

        projection_f = partial(
            rl4co_scheduling_projection,
            env_name=sched_env_name,
        )
        envs = RL4COSchedulingEnvironmentManager(_envs, projection_f, config)
        val_envs = RL4COSchedulingEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "rl4co" in config.env_name.lower():
        from agent_system.environments.env_package.rl4co import (
            build_route_envs,
            route_projection,
            route_projection_selected,
        )

        # Resolve rl4co-specific config (with sensible defaults)
        rl4co_env_name = config.env_name.split("/")[1]
        rl4co_device = config.device
        env_nums = config.train_batch_size
        return_topk_options = config.return_topk_options

        num_locs = np.random.randint(20,40, env_nums).tolist()

        generator_params = SimpleNamespace(
            num_loc=num_locs, 
            min_loc=0.0, 
            max_loc=1.0, 
            min_prize=1.0, 
            max_prize=2.0
        )
        generator_params = _to_container(generator_params)

        _envs = build_route_envs(
            env_name=rl4co_env_name,
            seed=config.seed,
            env_num=config.train_batch_size,
            group_n=group_n,
            device=rl4co_device,
            generator_params=generator_params,
            return_topk_options=return_topk_options,
        )
        _val_envs = build_route_envs(
            env_name=rl4co_env_name,
            seed=config.seed + 1000,
            env_num=config.val_batch_size,
            group_n=1,
            device=rl4co_device,
            generator_params=generator_params,
            return_topk_options=return_topk_options,
        )
        if return_topk_options > 0:
            projection_f = partial(
                route_projection_selected,
                env_name=rl4co_env_name,
            )
        else:
            projection_f = partial(
                route_projection,
                env_name=rl4co_env_name,
            )
        envs = RouteEnvironmentManager(_envs, projection_f, config)
        val_envs = RouteEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)


if __name__ == "__main__":
    """
    Manual quick-play supporting current backbones:
    - rl4co/*     : step-by-step routing (tsp/cvrp/op)
    - ml4co-kit/* : one-shot routing (tsp/cvrp/op)
    """
    import time

    env_name = "rl4co/tsp"  # change to rl4co/tsp|cvrp|op or ml4co-kit/*
    
    cfg = {
        "train_batch_size": 3,
        "val_batch_size": 1,
        "env_name": env_name,
        "seed": 0,
        "group_n": 3,
        "device":"cpu",
        "return_topk_options": 3,
    }
    cfg = OmegaConf.create(cfg)
    print(f"[Manual test] env_name = {env_name}")
    t0 = time.time()
    envs, _ = make_envs(cfg)
    print("Init time:", time.time() - t0)

    obs, infos = envs.reset(kwargs={})
    print("\n" + "=" * 80)
    print("Initial text observation (first env):")
    print("=" * 80)
    print(obs["text"][0])
    print("=" * 80)

    max_steps = 100
    for step_idx in range(max_steps):
        print("\n" + "=" * 80)
        print(f"Step {step_idx + 1}")
        print("=" * 80)
        batch_size = len(obs["text"])
        print(obs["text"][0])

        text_actions = input("\nEnter action(s) (comma-separated): ").strip()
        if not text_actions:
            text_actions = "0"
        acts = [a.strip() for a in text_actions.split(",")]
        if len(acts) < batch_size:
            acts.extend([acts[-1]] * (batch_size - len(acts)))

        obs, rewards, dones, infos = envs.step(acts)
        print(f"\nRewards: {rewards}")
        print(f"Dones: {dones}")

        if infos:
            print("\nReward Breakdown:")
            for i, info in enumerate(infos):
                env_reward = info.get("env_reward", rewards[i])
                if "format_reward" in info:
                    print(f"  Env {i}: Format={info.get('format_reward',0):.4f}, "
                            f"Feasibility={info.get('feasibility_reward',0):.4f}, "
                            f"Env={info.get('scaled_env_reward',0):.4f}, "
                            f"Final={rewards[i]:.4f}")
                elif "format_bonus" in info:
                    print(f"  Env {i}: FormatBonus={info.get('format_bonus',0):.4f}, "
                            f"FeasibilityBonus={info.get('feasibility_bonus',0):.4f}, "
                            f"Env={env_reward:.4f}, Final={rewards[i]:.4f}")
                else:
                    print(f"  Env {i}: Env={env_reward:.4f}, Final={rewards[i]:.4f}")

        done_flag = dones.all()
        if done_flag:
            print("\n" + "=" * 80)
            print("Episode finished!")
            print("=" * 80)
            break
