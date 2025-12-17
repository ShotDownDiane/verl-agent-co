import json
from typing import Any, List, Tuple


def _parse_int_sequence(text: str) -> List[int]:
    tokens = text.replace(",", " ").split()
    seq: List[int] = []
    for tok in tokens:
        try:
            seq.append(int(tok))
        except Exception:
            continue
    return seq


def rl4co_scheduling_projection(
    actions: List[str],
    one_step: bool = False,
    env_name: str | None = None,
) -> Tuple[Any, List[int]]:
    """Parse text actions for scheduling envs (JSSP/FFSP).

    - step-by-step: each action is an integer job/operation index.
    - one-step: each action encodes a full schedule as a job index sequence:
        * JSON: {"schedule": [0, 1, 3, ...]}
        * or plain string: "0 1 3 2 ..."
    """
    valids: List[int] = []

    if not one_step:
        parsed_actions: List[int] = []
        for a in actions:
            s = a.strip()
            # strip simple prefixes
            for prefix in ["op_", "task_", "job_", "operation_"]:
                if s.lower().startswith(prefix):
                    s = s[len(prefix) :]
                    break
            try:
                idx = int(s)
                parsed_actions.append(idx)
                valids.append(1)
            except Exception:
                parsed_actions.append(0)
                valids.append(0)
        return parsed_actions, valids

    # one-step: parse full schedules
    schedules: List[List[int]] = []
    for a in actions:
        s = a.strip()
        seq: List[int] = []
        ok = 0
        try:
            obj = json.loads(s)
            if isinstance(obj, dict) and "schedule" in obj:
                if isinstance(obj["schedule"], list):
                    seq = [int(x) for x in obj["schedule"]]
                    ok = 1
                elif isinstance(obj["schedule"], str):
                    seq = _parse_int_sequence(obj["schedule"])
                    ok = 1 if seq else 0
        except Exception:
            seq = _parse_int_sequence(s)
            ok = 1 if seq else 0

        if not seq:
            seq = [0]
        schedules.append(seq)
        valids.append(ok)

    return schedules, valids


