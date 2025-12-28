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

def route_projection_selected(
    actions: List[str],
    env_name: str | None = None
) -> Tuple[List[int], List[int]]:
    """
    Parse text actions based on the specific mode.
    """
    valids: List[int] = []
    parsed_actions: List[int] = []

    for a in actions:
        s = a.strip()
        m = re.search(r"\\boxed\{([^{}]*)\}", s)
        
        # 1. 如果没找到 box，直接判错
        if not m:
            parsed_actions.append(0)
            valids.append(0)
            continue

        content = m.group(1).strip()
        
        # --- 模式 A: 解析选项 (A/B/C -> 0/1/2) ---
        # 清理可能的 "Option A", "Choice B" 等前缀
        clean_content = re.sub(r"^(Option|Choice)\s+", "", content, flags=re.IGNORECASE).strip()
        
        # 严格限制只能是单个字母
        if len(clean_content) == 1 and clean_content.isalpha():
            idx = ord(clean_content.upper()) - 65  # A=0, B=1...
            if 0 <= idx < 26:
                parsed_actions.append(idx)
                valids.append(1)
            else:
                # 字母超出合理范围
                parsed_actions.append(0)
                valids.append(0)
        else:
            # 解析失败 (例如输出了 "A and B" 或 "Node 1")
            parsed_actions.append(0)
            valids.append(0)

    return parsed_actions, valids