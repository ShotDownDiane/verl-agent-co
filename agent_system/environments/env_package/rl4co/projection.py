import json
from typing import Any, List, Tuple


def _parse_int_sequence(text: str) -> List[int]:
    """Utility: parse a whitespace/comma separated sequence of ints."""
    tokens = text.replace(",", " ").split()
    seq: List[int] = []
    for tok in tokens:
        try:
            seq.append(int(tok))
        except Exception:
            continue
    return seq


def rl4co_projection(
    actions: List[str],
    one_step: bool = False,
    env_name: str | None = None,
) -> Tuple[Any, List[int]]:
    """Parse text actions for RL4CO routing envs.

    - step-by-step mode (one_step=False): each action is a single integer index.
    - one-step mode (one_step=True): each action encodes a full route, either as:
        * JSON: {"route": [0, 5, 2, ...]}
        * or plain string: "0 5 2 3 ..."
    """
    valids: List[int] = []

    if not one_step:
        parsed_actions: List[int] = []
        for a in actions:
            s = a.strip()
            try:
                idx = int(s)
                parsed_actions.append(idx)
                valids.append(1)
            except Exception:
                parsed_actions.append(0)
                valids.append(0)
        return parsed_actions, valids

    # one-step: parse full routes
    routes: List[List[int]] = []
    for a in actions:
        s = a.strip()
        route: List[int] = []
        ok = 0
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "route" in obj:
                if isinstance(obj["route"], list):
                    route = [int(x) for x in obj["route"]]
                    ok = 1
                elif isinstance(obj["route"], str):
                    route = _parse_int_sequence(obj["route"])
                    ok = 1 if route else 0
        except Exception:
            # not JSON, treat as plain sequence
            route = _parse_int_sequence(s)
            ok = 1 if route else 0

        if not route:
            # fallback: dummy route [0]
            route = [0]
        routes.append(route)
        valids.append(ok)

    return routes, valids


