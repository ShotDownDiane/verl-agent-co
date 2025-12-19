from typing import List, Tuple
import re

# def poi_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
#     """
#     将 LLM 输出中的 <action>标签内容提取为合法的 A–E 动作。
    
#     actions: LLM 的原始输出列表，每个元素类似：
#         "<think>…</think><action>A</action>"
    
#     返回：
#         processed_actions: 清洗后的动作标识列表，如 ["A", "C", ...] 或原样文本
#         valids: 有效性标志列表，与 processed_actions 对应，1 表示合法，0 表示非法
#     """
#     valid_actions = {"A","B","C","D","E"}
#     processed = []
#     valids = []
#     for raw in actions:
#         act = raw.lower()
#         val = 0
#         # 正则提取 <action> 标签
#         m = re.search(r"<answer>(.*?)</answer>", act, flags=re.DOTALL)
#         token = ""
#         if m:
#             content = m.group(1).strip()
#             # 尝试用 json 解析（比 eval 安全）
#             try:
#                 parsed = json.loads(content)
#                 if isinstance(parsed, dict) and "answer" in parsed:
#                     token = parsed["answer"].strip().upper()
#                 else:
#                     token = str(parsed).strip().upper()
#             except Exception:
#                 # 如果不是合法 JSON，尝试正则提取 answer
#                 m2 = re.search(r'"answer"\s*:\s*"([a-e])"', content)
#                 if m2:
#                     token = m2.group(1).strip().upper()
#                 else:
#                     token = ""

#             if token in valid_actions:
#                 processed.append(token)
#                 val = 1
#             else:
#                 processed.append(token)  # 非法但提取
#         else:
#             processed.append("0")  # 无 action 标签

#         # 检查 think 是否存在
#         if "<think>" not in raw or "</think>" not in raw:
#             val = 0

#         processed[-1] = processed[-1] or ""  # 防 None
#         valids.append(val)
#     return processed, valids

from typing import List
import re

def poi_projection(actions: List[str]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().upper()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # check <think>...</think>
        think_start_idx = original_str.find("<think>")
        think_end_idx = original_str.find("</think>")
        if think_start_idx == -1 or think_end_idx == -1:
            valids[i] = 0

    return actions, valids

