"""Format reward calculation module.

This module provides functions to compute format rewards based on:
1. Format correctness: whether the action follows the required format
2. Solution feasibility: whether the solution satisfies problem constraints
3. Environment reward: the original reward from the environment
"""

from typing import List, Dict, Any, Optional, Tuple
import torch
import numpy as np


def check_tsp_route_feasibility(
    route: List[int],
    num_nodes: int,
    start_node: int = 0,
) -> bool:
    """Check if a TSP route is feasible.
    
    A feasible TSP route should:
    - Visit all nodes exactly once (except possibly returning to start)
    - Start from the start_node (usually 0)
    - Contain valid node indices
    
    Args:
        route: List of node indices representing the route
        num_nodes: Total number of nodes in the problem
        start_node: Expected starting node (default: 0)
    
    Returns:
        True if the route is feasible, False otherwise
    """
    if not route:
        return False
    
    # Check if all indices are valid
    if any(idx < 0 or idx >= num_nodes for idx in route):
        return False
    
    # Extract unique nodes visited (excluding potential return to start)
    # For TSP, we typically expect: [start, node1, node2, ..., nodeN, start]
    # or just [start, node1, node2, ..., nodeN]
    unique_nodes = set(route)
    
    # If route ends at start, remove the last occurrence
    if len(route) > 1 and route[-1] == start_node:
        route_core = route[:-1]
    else:
        route_core = route
    
    # Check if we visit all nodes exactly once
    if len(set(route_core)) != len(route_core):
        return False  # Duplicate visits
    
    # Check if we visit all required nodes
    if len(set(route_core)) < num_nodes:
        return False  # Not all nodes visited
    
    return True


def check_cvrp_route_feasibility(
    routes: List[List[int]],
    num_nodes: int,
    depot: int = 0,
) -> bool:
    """Check if CVRP routes are feasible.
    
    A feasible CVRP solution should:
    - Each route starts and ends at depot
    - All customer nodes are visited exactly once
    - All node indices are valid
    
    Args:
        routes: List of routes, each route is a list of node indices
        num_nodes: Total number of nodes (including depot)
        depot: Depot node index (default: 0)
    
    Returns:
        True if routes are feasible, False otherwise
    """
    if not routes:
        return False
    
    visited_nodes = set()
    
    for route in routes:
        if not route:
            continue
        
        # Check if route starts and ends at depot
        if route[0] != depot or route[-1] != depot:
            return False
        
        # Check if all indices are valid
        if any(idx < 0 or idx >= num_nodes for idx in route):
            return False
        
        # Collect customer nodes (excluding depot)
        customer_nodes = [n for n in route if n != depot]
        
        # Check for duplicates within route
        if len(customer_nodes) != len(set(customer_nodes)):
            return False
        
        # Check for duplicates across routes
        for node in customer_nodes:
            if node in visited_nodes:
                return False
            visited_nodes.add(node)
    
    # Check if all customer nodes are visited (depot is node 0)
    expected_customers = set(range(1, num_nodes))
    if visited_nodes != expected_customers:
        return False
    
    return True


def check_schedule_feasibility(
    schedule: List[int],
    num_jobs: Optional[int] = None,
) -> bool:
    """Check if a scheduling solution is feasible.
    
    For JSSP/FFSP, a basic feasibility check:
    - All indices are non-negative
    - If num_jobs is provided, all indices should be < num_jobs
    
    Note: Full feasibility checking would require checking constraints
    against the actual problem instance, which is complex. This is a
    basic check that can be enhanced later.
    
    Args:
        schedule: List of job/operation indices
        num_jobs: Optional total number of jobs
    
    Returns:
        True if schedule passes basic checks, False otherwise
    """
    if not schedule:
        return False
    
    # Check if all indices are non-negative
    if any(idx < 0 for idx in schedule):
        return False
    
    # If num_jobs is provided, check bounds
    if num_jobs is not None:
        if any(idx >= num_jobs for idx in schedule):
            return False
    
    return True


def normalize_env_rewards(
    env_rewards: np.ndarray,
    reward_range: Optional[Tuple[float, float]] = None,
    method: str = "minmax",
) -> np.ndarray:
    """Normalize environment rewards to [0, 1] range.
    
    Since environment rewards are typically negative (e.g., -7 to -9),
    this function converts them to a [0, 1] range for combination with
    format/feasibility rewards.
    
    Args:
        env_rewards: Original environment rewards (typically negative)
        reward_range: Optional (min, max) range for normalization.
                     If None, uses min/max of current batch
        method: Normalization method ("minmax" or "exp")
    
    Returns:
        Normalized rewards in [0, 1] range
    """
    env_rewards = np.asarray(env_rewards)
    
    if method == "minmax":
        if reward_range is None:
            # Use batch statistics
            min_reward = np.min(env_rewards)
            max_reward = np.max(env_rewards)
        else:
            min_reward, max_reward = reward_range
        
        # Handle case where all rewards are the same
        if max_reward == min_reward:
            return np.ones_like(env_rewards, dtype=np.float32)
        
        # Normalize to [0, 1]
        normalized = (env_rewards - min_reward) / (max_reward - min_reward)
        return np.clip(normalized, 0, 1).astype(np.float32)
    
    elif method == "exp":
        # Use exponential function to map negative rewards to [0, 1]
        # Better rewards (less negative) get higher scores
        # Assuming rewards are typically in range [-20, 0]
        if reward_range is None:
            min_reward = -20.0
            max_reward = 0.0
        else:
            min_reward, max_reward = reward_range
        
        # Shift to positive range
        shifted = env_rewards - min_reward
        # Normalize
        normalized = shifted / (max_reward - min_reward)
        return np.clip(normalized, 0, 1).astype(np.float32)
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def compute_format_reward(
    valids: List[int],
    actions: Any,
    env_rewards: np.ndarray,
    env_name: str,
    env_type: str = "routing",  # "routing" or "scheduling"
    format_reward_weight: float = 0.05,
    feasibility_reward_weight: float = 0.15,
    num_nodes: Optional[int] = None,
    num_jobs: Optional[int] = None,
    start_node: int = 0,
    # New parameters for conditional reward mechanism
    use_conditional_reward: bool = True,
    feasibility_threshold: float = 0.9,
    normalize_env_reward: bool = True,
    env_reward_range: Optional[Tuple[float, float]] = None,
    fixed_scale_reference: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute format reward using conditional reward mechanism (Scheme B).
    
    Conditional Reward Mechanism:
    1. Format reward: Base reward, always given
    2. Feasibility reward: Only given if format is correct
    3. Environment reward: Full weight only if feasibility >= threshold
    
    This prevents reward hacking by requiring:
    - Format correctness for feasibility reward
    - High feasibility for full environment reward weight
    
    Args:
        valids: List of validity flags (1 if format is correct, 0 otherwise)
        actions: Parsed actions (routes for routing, schedules for scheduling)
        env_rewards: Original rewards from environment (numpy array, typically negative)
        env_name: Environment name (e.g., "tsp", "cvrp", "jssp")
        env_type: Type of environment ("routing" or "scheduling")
        format_reward_weight: Weight for format correctness bonus (default: 0.05)
        feasibility_reward_weight: Weight for feasibility bonus (default: 0.15)
        num_nodes: Number of nodes (for routing problems)
        num_jobs: Number of jobs (for scheduling problems)
        start_node: Starting node index (for routing problems)
        use_conditional_reward: Whether to use conditional reward mechanism (default: True)
        feasibility_threshold: Threshold for high feasibility (default: 0.9)
        normalize_env_reward: Whether to normalize env rewards to [0, 1] (default: True)
        env_reward_range: Optional (min, max) range for normalization
        fixed_scale_reference: Optional fixed reference value for scaling (avoids batch instability)
    
    Returns:
        Tuple of (final_rewards, info_dict) where info_dict contains breakdown
    """
    batch_size = len(valids)
    env_rewards = np.asarray(env_rewards).flatten()
    
    if len(env_rewards) != batch_size:
        # Handle case where env_rewards might be 2D
        env_rewards = env_rewards.reshape(-1)[:batch_size]
    
    format_bonuses = np.zeros(batch_size, dtype=np.float32)
    feasibility_bonuses = np.zeros(batch_size, dtype=np.float32)
    is_feasible = np.zeros(batch_size, dtype=bool)
    
    # Compute format and feasibility bonuses
    for i in range(batch_size):
        # Format correctness bonus (0 or 1)
        if valids[i] == 1:
            format_bonuses[i] = 1.0
        
        # Feasibility bonus (0 or 1)
        if env_type == "routing":
            if isinstance(actions, list) and len(actions) > i:
                route = actions[i]
                if isinstance(route, list):
                    if env_name == "tsp":
                        feasible = check_tsp_route_feasibility(
                            route, num_nodes or len(route), start_node
                        )
                    elif env_name == "cvrp":
                        # For CVRP, actions might be a list of routes (multi-route) or single route
                        if route and isinstance(route[0], list):
                            # Multi-route case: route is List[List[int]]
                            feasible = check_cvrp_route_feasibility(
                                route, num_nodes or max(max(r) for r in route if r) + 1, start_node
                            )
                        else:
                            # Single route case - treat as TSP-like (depot -> customers -> depot)
                            feasible = check_tsp_route_feasibility(
                                route, num_nodes or len(route), start_node
                            )
                    else:
                        # OP or other routing problems - use basic check
                        feasible = (
                            len(route) > 0
                            and all(0 <= idx < (num_nodes or 1000) for idx in route)
                        )
                    
                    is_feasible[i] = feasible
                    if feasible:
                        feasibility_bonuses[i] = 1.0
        
        elif env_type == "scheduling":
            if isinstance(actions, list) and len(actions) > i:
                schedule = actions[i]
                if isinstance(schedule, list):
                    feasible = check_schedule_feasibility(schedule, num_jobs)
                    is_feasible[i] = feasible
                    if feasible:
                        feasibility_bonuses[i] = 1.0
    
    # Compute final rewards using conditional reward mechanism
    if use_conditional_reward:
        # Normalize environment rewards to [0, 1] if needed
        if normalize_env_reward:
            normalized_env = normalize_env_rewards(
                env_rewards,
                reward_range=env_reward_range,
                method="minmax"
            )
        else:
            # Use original env rewards (will be scaled)
            normalized_env = env_rewards
        
        # Conditional reward mechanism:
        # 1. Format reward: Base reward, always given
        format_reward = format_reward_weight * format_bonuses
        
        # 2. Feasibility reward: Only given if format is correct
        feasibility_reward = (
            feasibility_reward_weight * 
            feasibility_bonuses * 
            format_bonuses  # Condition: format must be correct
        )
        
        # 3. Environment reward: Full weight only if feasibility >= threshold
        # Low feasibility gets reduced weight (similar to LLMCoSolver)
        feasibility_mask = feasibility_bonuses >= feasibility_threshold
        
        # Calculate remaining weight for environment reward
        remaining_weight = 1.0 - format_reward_weight - feasibility_reward_weight
        
        # High feasibility: full weight; Low feasibility: 10% weight
        env_reward_weight = (
            feasibility_mask * remaining_weight +
            (1 - feasibility_mask) * 0.1 * remaining_weight
        )
        
        # Scale environment reward
        if normalize_env_reward:
            # Normalized env reward is already in [0, 1]
            env_reward = normalized_env * env_reward_weight
        else:
            # Scale original env reward using fixed reference or batch mean
            if fixed_scale_reference is not None:
                scale_factor = fixed_scale_reference
            else:
                # Use batch mean absolute reward
                scale_factor = np.mean(np.abs(env_rewards)) if len(env_rewards) > 0 else 1.0
            
            # Normalize and scale
            normalized_env = normalize_env_rewards(
                env_rewards,
                reward_range=env_reward_range,
                method="minmax"
            )
            env_reward = normalized_env * env_reward_weight * scale_factor
        
        # Final reward: sum of all components
        final_rewards = format_reward + feasibility_reward + env_reward
        
        # Store detailed info
        info = {
            "format_bonuses": format_bonuses,
            "feasibility_bonuses": feasibility_bonuses,
            "is_feasible": is_feasible,
            "format_correct": np.array(valids, dtype=bool),
            "env_rewards": env_rewards,
            "normalized_env_rewards": normalized_env if normalize_env_reward else None,
            "format_reward": format_reward,
            "feasibility_reward": feasibility_reward,
            "env_reward": env_reward,
            "env_reward_weight": env_reward_weight,
            "feasibility_mask": feasibility_mask,
        }
    
    else:
        # Original simple additive mechanism (for backward compatibility)
        if fixed_scale_reference is not None:
            format_scale = format_reward_weight * fixed_scale_reference
            feasibility_scale = feasibility_reward_weight * fixed_scale_reference
        else:
            # Use batch mean absolute reward
            mean_abs_reward = np.mean(np.abs(env_rewards)) if len(env_rewards) > 0 else 1.0
            format_scale = format_reward_weight * mean_abs_reward
            feasibility_scale = feasibility_reward_weight * mean_abs_reward
        
        final_rewards = (
            env_rewards +
            format_scale * format_bonuses +
            feasibility_scale * feasibility_bonuses
        )
        
        info = {
            "format_bonuses": format_bonuses,
            "feasibility_bonuses": feasibility_bonuses,
            "is_feasible": is_feasible,
            "format_correct": np.array(valids, dtype=bool),
            "env_rewards": env_rewards,
        }
    
    return final_rewards, info

