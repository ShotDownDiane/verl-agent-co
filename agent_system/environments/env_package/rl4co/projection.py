import json
from typing import Any, List, Tuple
import re


def co_projection(
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

import re
from typing import List, Tuple

def co_projection_selected(
    actions: List[str],
    env_name: str | None = None
) -> Tuple[List[int], List[int]]:
    """
    Parse text actions with strict structural validation.
    
    Criteria for valid=1:
    1. Must contain <Observation>...</Observation> block.
    2. Must contain <Thought>...</Thought> block.
    3. Must contain <Decision>...</Decision> block.
    4. The <Decision> block must contain \boxed{X}.
    5. X must be a valid single uppercase letter.
    """
    valids: List[int] = []
    parsed_actions: List[int] = []
    
    # 定义提取 Decision 块的正则（支持跨行）
    decision_block_pattern = re.compile(r"<Decision>(.*?)</Decision>", re.DOTALL | re.IGNORECASE)
    # 定义提取 boxed 内容的正则
    box_pattern = re.compile(r"\\boxed\{([^{}]*)\}")

    for a in actions:
        # --- 1. 结构完整性检查 (三段式验证) ---
        # 必须同时包含三个核心标签对
        has_obs = "<Observation>" in a and "</Observation>" in a
        has_thought = "<Thought>" in a and "</Thought>" in a
        # Decision 标签将在下面通过正则严格提取，这里先做简单检查
        has_decision_tags = "<Decision>" in a and "</Decision>" in a

        if not (has_obs and has_thought and has_decision_tags):
            parsed_actions.append(0)
            valids.append(0)
            continue

        # --- 2. 提取 Decision 块内容 ---
        # 我们只从 Decision 标签内部提取答案，防止模型在 Thought 里幻觉出 \boxed{}
        decision_match = decision_block_pattern.search(a)
        
        if not decision_match:
            parsed_actions.append(0)
            valids.append(0)
            continue
            
        decision_content = decision_match.group(1).strip()

        # --- 3. 提取 Boxed 答案 ---
        box_match = box_pattern.search(decision_content)
        
        if not box_match:
            parsed_actions.append(0)
            valids.append(0)
            continue

        content = box_match.group(1).strip()
        
        # --- 4. 解析选项 (A -> 0, B -> 1) ---
        # 清理可能的前缀 (虽然 boxed 内通常很干净)
        clean_content = re.sub(r"^(Option|Choice)\s+", "", content, flags=re.IGNORECASE).strip()

        if len(clean_content) == 1 and clean_content.isalpha():
            # 强制转换为大写，处理 'a' 的情况
            idx = ord(clean_content.upper()) - 65 
            if 0 <= idx < 50: # 假设最多50个选项，防止异常字符
                parsed_actions.append(idx)
                valids.append(1) # 只有闯过所有关卡，这里才给 1
            else:
                parsed_actions.append(0)
                valids.append(0)
        else:
            parsed_actions.append(0)
            valids.append(0)

    return parsed_actions, valids