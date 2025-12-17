import json
import re
from typing import List, Tuple, Any


def _parse_int_sequence(text: str) -> List[int]:
    """Parse a sequence of integers from a string."""
    return [int(x) for x in re.findall(r"-?\d+", text)]


def _normalize_cvrp_routes(seq: List[int]) -> List[List[int]]:
    """Split a flat sequence with depot markers (0) into CVRP routes."""
    if not seq:
        return []
    routes: List[List[int]] = []
    current: List[int] = []
    for idx in seq:
        if idx == 0:
            if current:
                current.append(0)
                routes.append(current)
                current = []
            current.append(0)
        else:
            current.append(idx)
    if current:
        if current[-1] != 0:
            current.append(0)
        routes.append(current)
    return routes


def _safe_list(obj: Any) -> List[int]:
    """Convert obj to list[int] if possible."""
    if isinstance(obj, list):
        try:
            return [int(x) for x in obj]
        except Exception:
            return []
    if isinstance(obj, str):
        return _parse_int_sequence(obj)
    return []


def ml4cokit_projection(
    text_actions: List[str],
    env_name: str,
) -> Tuple[List[Any], List[int]]:
    """Parse text actions for ML4CO-Kit routing environments."""
    env_name = env_name.lower()
    parsed_actions: List[Any] = []
    valids: List[int] = []

    for raw in text_actions:
        action = None
        valid = 0

        # try:
        #     data = json.loads(raw)
        # except Exception:
        #     data = raw
        # 兼容Route: [0, 29, 57, 76, 51, 18, 54, 55, 39, 10, 43, 81, 47, 15, 33, 12, 34, 7, 65, 45, 59, 40, 32, 8, 73, 67, 26, 80, 25, 38, 6, 68, 19, 4, 21, 61, 48, 5, 13, 72, 83, 53, 16, 44, 31, 2, 77, 74, 60, 9, 37, 86, 64, 85, 23, 78, 30, 50, 46, 66, 69, 3, 22, 1, 41, 75, 14, 20, 36, 11, 63, 27, 84, 42, 49, 71, 17, 28, 58, 52, 79, 24, 70, 82, 62, 56, 35, 0], Objective: 7591.213",
        try:
            data = json.loads(raw)
        except Exception:
            print("Raw action is not valid JSON, trying to extract route info.")
            route_match = re.search(r"Route:\s*\[([-\d,\s]+)\]", raw)
            if route_match:
                route_str = route_match.group(1)
                data = {"route": route_str}
            else:
                data = raw

        if isinstance(data, dict):
            if env_name == "cvrp":
                if "routes" in data:
                    routes = []
                    for r in data.get("routes", []):
                        r_list = _safe_list(r)
                        if r_list and r_list[0] != 0:
                            r_list = [0] + r_list
                        if r_list and r_list[-1] != 0:
                            r_list = r_list + [0]
                        if r_list:
                            routes.append(r_list)
                    if routes:
                        action = routes
                elif "route" in data:
                    route_seq = _safe_list(data.get("route"))
                    routes = _normalize_cvrp_routes(route_seq)
                    if routes:
                        action = routes
            else:
                if "route" in data:
                    route_list = _safe_list(data.get("route"))
                    if route_list:
                        action = route_list
        elif isinstance(data, str):
            seq = _parse_int_sequence(data)
            if env_name == "cvrp":
                routes = _normalize_cvrp_routes(seq)
                if routes:
                    action = routes
            else:
                if seq:
                    action = seq
        elif isinstance(data, list):
            # Already a list structure
            if env_name == "cvrp" and data and isinstance(data[0], list):
                try:
                    action = [[int(x) for x in r] for r in data]
                except Exception:
                    action = None
            else:
                try:
                    action = [int(x) for x in data]
                except Exception:
                    action = None

        if action is not None:
            valid = 1

        parsed_actions.append(action if action is not None else [])
        valids.append(valid)

    return parsed_actions, valids


def ml4cokit_scheduling_projection(
    text_actions: List[str],
    env_name: str,
) -> Tuple[List[List[int]], List[int]]:
    """Parse text actions for ML4CO-Kit scheduling environments."""
    parsed_actions: List[List[int]] = []
    valids: List[int] = []

    for raw in text_actions:
        action: List[int] = []
        valid = 0

        try:
            data = json.loads(raw)
        except Exception:
            data = raw

        if isinstance(data, dict) and "schedule" in data:
            action = _safe_list(data.get("schedule"))
        elif isinstance(data, str):
            action = _parse_int_sequence(data)
        elif isinstance(data, list):
            action = _safe_list(data)

        if action:
            valid = 1

        parsed_actions.append(action)
        valids.append(valid)

    return parsed_actions, valids

