import json
from typing import Any, List, Tuple
import re


def route_projection(
    actions: List[str],
    env_name: str | None = None,
) -> Tuple[Any, List[int]]:
    """Parse text actions for RL4CO routing envs.
    """
    valids: List[int] = []

    parsed_actions: List[int] = []
    for a in actions:
        s = a.strip()
        # If user/model provided an explicit boxed answer \box{...}, prefer it.
        m = re.search(r"\\boxed\{([^}]*)\}", s)
        if m:
            inner = m.group(1).strip()
            # Try to parse integer directly
            try:
                idx = int(inner)
                parsed_actions.append(idx)
                valids.append(1)
            except Exception:
                # Could not parse boxed answer -> treat as invalid
                print("invalid answer")
                parsed_actions.append(0)
                valids.append(0)
        else:
            print("invalid answer")
            parsed_actions.append(0)
            valids.append(0)

    return parsed_actions, valids