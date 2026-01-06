import os
import sys
import json
import re
import math
import time
import random
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

# Add project root to path to import VLMAgent
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
try:
    from examples.prompt_agent.vlm_agent import VLMAgent
    from examples.prompt_agent.llm_agent import LLMAgent
except ImportError:
    print("Warning: VLMAgent not found. Mocking for development.")
    class VLMAgent:
        def __init__(self, **kwargs): pass
        def generate(self, text, image, **kwargs): return "Mock VLM Response: I see the nodes. The target is to the North-East."

# ==========================================
# Global Constants & Data
# ==========================================
NODE_COORDS: Dict[int, Tuple[float, float]] = {}

# ==========================================
# Module 1: Geometry Engine
# ==========================================

@dataclass
class SpatialRelation:
    target_id: int
    direction: str         # N, NE, E, SE, S, SW, W, NW
    distance: float        # Euclidean distance
    is_nearest: bool
    normalized_dist: float # Distance normalized by max possible distance (sqrt(2))

class GeometryEngine:
    """
    God-view engine to calculate spatial truths.
    """
    def __init__(self):
        self.compass_sectors = {
            'E': (337.5, 22.5),
            'NE': (22.5, 67.5),
            'N': (67.5, 112.5),
            'NW': (112.5, 157.5),
            'W': (157.5, 202.5),
            'SW': (202.5, 247.5),
            'S': (247.5, 292.5),
            'SE': (292.5, 337.5)
        }

    def _get_coords(self, node_id: int) -> Tuple[float, float]:
        return NODE_COORDS[node_id]

    def calculate_bearing_angle(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """Calculates angle in degrees (0-360). 0=East, 90=North."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.degrees(math.atan2(dy, dx))
        return (angle + 360) % 360

    def get_compass_direction(self, start: Tuple[float, float], end: Tuple[float, float]) -> str:
        angle = self.calculate_bearing_angle(start, end)
        # Adjust to standard compass if needed, but let's stick to standard math -> compass mapping
        # 0 (East) -> 'E'
        # 90 (North) -> 'N'
        # etc.
        
        # Check sectors
        for direction, (low, high) in self.compass_sectors.items():
            # Handle wrap-around for East (337.5 - 360 & 0 - 22.5)
            if direction == 'E':
                if angle >= 337.5 or angle < 22.5:
                    return direction
            else:
                if low <= angle < high:
                    return direction
        return 'Unknown'

    def analyze(self, current_pos: Tuple[float, float], candidate_ids: List[int], gt_node_id: int) -> Dict[str, Any]:
        """
        Analyzes spatial relations between current position, candidates, and GT.
        """
        gt_coords = self._get_coords(gt_node_id)
        
        # 1. Calculate relations for GT
        gt_dist = math.hypot(gt_coords[0] - current_pos[0], gt_coords[1] - current_pos[1])
        gt_dir = self.get_compass_direction(current_pos, gt_coords)
        
        # 2. Calculate relations for all candidates
        candidates_meta = []
        min_dist = float('inf')
        
        for cid in candidate_ids:
            c_coords = self._get_coords(cid)
            dist = math.hypot(c_coords[0] - current_pos[0], c_coords[1] - current_pos[1])
            direction = self.get_compass_direction(current_pos, c_coords)
            candidates_meta.append({
                'id': cid,
                'dist': dist,
                'dir': direction
            })
            if dist < min_dist:
                min_dist = dist
                
        # 3. Identify Distinctive Features
        # Is GT the nearest?
        is_nearest = abs(gt_dist - min_dist) < 1e-6
        
        # Is GT unique in its direction?
        same_dir_count = sum(1 for c in candidates_meta if c['dir'] == gt_dir)
        is_unique_dir = (same_dir_count == 1) and (gt_node_id in candidate_ids) # Assuming GT is in candidates
        
        return {
            'gt_id': gt_node_id,
            'gt_direction': gt_dir,
            'gt_distance': gt_dist,
            'is_nearest': is_nearest,
            'is_unique_direction': is_unique_dir,
            'candidates_meta': candidates_meta,
            'gt_coords': gt_coords,
            'current_pos': current_pos
        }

# ==========================================
# Module 2: Perception Loop
# ==========================================

from typing import Dict, Any, Tuple, Optional, List

class PerceptionModule:
    def __init__(self, vlm_agent):
        self.vlm = vlm_agent
        # 扩展方位词表，增加容错性
        self.direction_map = {
            'N': ['north', 'top', 'upper', 'above', 'up'],
            'S': ['south', 'bottom', 'lower', 'below', 'down'],
            'E': ['east', 'right'],
            'W': ['west', 'left'],
            'NE': ['north-east', 'top-right', 'upper-right'],
            'NW': ['north-west', 'top-left', 'upper-left'],
            'SE': ['south-east', 'bottom-right', 'lower-right'],
            'SW': ['south-west', 'bottom-left', 'lower-left']
        }
        # 简单的反义词映射，用于 Feedback
        self.antonyms = {
            'N': 'South/Bottom', 'S': 'North/Top',
            'E': 'West/Left', 'W': 'East/Right'
        }

    def verify_description(self, description: str, spatial_facts: Dict[str, Any]) -> Tuple[bool, str]:
        """
        验证逻辑：
        1. ID Check: 是否提到了正确的 GT ID？
        2. Direction Check: 是否包含了正确的方位词？
        3. Negation Check: 防止 "Not North" 被判定为 True (简单版)
        """
        desc_lower = description.lower()
        gt_id = str(spatial_facts['gt_id']).lower() # 确保 ID 是字符串
        gt_dir = spatial_facts['gt_direction']
        
        # --- 1. Identity Check (关键修复: 确保它在描述正确的对象) ---
        # 检查 GT ID 是否出现在描述中。
        # 注意：如果 ID 是简单的字母 'A', 可能会误匹配单词中的 a。
        # 建议在 prompt 里要求输出 "Candidate A" 这样带前缀的格式，或者 ID 本身较长。
        if f"candidate {gt_id}" not in desc_lower and f"option {gt_id}" not in desc_lower and gt_id not in desc_lower.split():
             return False, f"Description failed to explicitly mention the correct target ID: '{spatial_facts['gt_id']}'."

        # --- 2. Direction Check ---
        expected_keywords = self.direction_map.get(gt_dir, [])
        dir_found = False
        
        # 扫描关键词
        matched_kw = ""
        for kw in expected_keywords:
            if kw in desc_lower:
                # 简单的语境检查：确保不是 "not north"
                # (这只是一个简单的 heuristic，更复杂的需要 NLP dependency parsing，但通常够用)
                idx = desc_lower.find(kw)
                if idx > 3 and "not " in desc_lower[idx-4:idx]:
                    continue # 跳过否定的方位
                
                dir_found = True
                matched_kw = kw
                break
        
        if not dir_found:
            return False, f"Description incorrect. It failed to identify that {gt_id} is to the {gt_dir} (Keywords expected: {expected_keywords})."

        # --- 3. Consistency/Hallucination Check (可选) ---
        # 如果 GT 在 North，但描述里疯狂出现 South，可能是幻觉
        # 暂时略过，避免过度约束，先把上面两个过了就行

        return True, "Verified"

    def run_perception_loop(self, image_b64: str, spatial_facts: Dict[str, Any], max_retries: int = 3) -> Optional[str]:
        """
        Input: 
            spatial_facts: {'gt_id': 'H', 'gt_direction': 'NW', 'is_nearest': False, ...}
        """
        gt_id = spatial_facts['gt_id']
        gt_dir = spatial_facts['gt_direction']
        
        # 构造提示词 (Hints)
        dir_hint = f"It is located to the {self.direction_map[gt_dir][0]} (or {self.direction_map[gt_dir][1]})."
        feature_hint = "It is the closest candidate." if spatial_facts.get('is_nearest') else "It is clearly visible."
        
        system_instruction = (
            "You are a precise spatial perception engine. "
            "Your job is NOT to solve the problem, but to GROUND the provided truth into the image. "
            "Describe the spatial facts strictly based on the image provided."
        )
        
        # [关键修改] Prompt 策略：直接告诉它答案，让它去图里找并在视觉上确认
        user_prompt = (
            f"Please focus on **Candidate {gt_id}** in the provided image. (Assume {gt_id} is the correct choice).\n"
            f"1. Locate **Candidate {gt_id}** explicitly.\n"
            f"2. Describe its position relative to the Current Position/Center.\n"
            f"3. Describe its immediate surroundings (is it isolated or in a cluster?).\n"
            f"4. Compare it briefly with other nearby candidates to explain its uniqueness."
        )

        for attempt in range(max_retries):
            print(f"--- Perception Attempt {attempt + 1} for ID {gt_id} ---")
            
            # 这里的 generate 需要你根据实际模型 API 适配
            response = self.vlm.generate(
                system_prompt=system_instruction,
                text=user_prompt, 
                image=image_b64
            )
            
            # print(f"VLM Output: {response}") 
            
            # Verify
            is_valid, reason = self.verify_description(response, spatial_facts)
            
            if is_valid:
                print("  -> Verification Passed.")
                return response
            
            # Feedback Loop
            print(f"  -> Failed: {reason}")
            
            # [关键修改] 动态调整 Prompt，给予更强的引导
            user_prompt = (
                f"Your previous description was rejected. Reason: {reason}\n"
                f"Let me give you the ground truth: **Candidate {gt_id} is located to the {gt_dir}**.\n"
                f"Please look at the image again. Find the specific marker for {gt_id} in that direction.\n"
                f"Describe ONLY Candidate {gt_id} and confirm its location matches the ground truth."
            )
            
        print(f"!!! Validation Failed after {max_retries} retries for Step ID {gt_id} !!!")
        return None

# ==========================================
# Module 3: Logic Injection
# ==========================================

from typing import Dict, Any

class LogicInjectionModule:
    def __init__(self, llm_agent):
        # 假设 llm_agent 可以是同一个 VLM 实例，也可以是专门的文本模型 (GPT-4o)
        self.llm = llm_agent

    def inject_logic(self, 
                     verified_desc: str, 
                     spatial_facts: Dict[str, Any], 
                     trajectory_action: str, 
                     problem_context: str = "spatial optimization") -> str:
        """
        Generate the 'Thought' part by grounding the reasoning in geometric facts.
        
        Args:
            verified_desc: The text output from PerceptionModule.
            spatial_facts: The raw geometric data calculated in Module 1 (e.g., {'is_nearest': True, 'gt_direction': 'NW'}).
            trajectory_action: The ID of the optimal node.
            problem_context: e.g., "Facility Location", "TSP", or general description.
        """
        
        gt_id = str(trajectory_action)
        
        # 1. 动态构建 "逻辑锚点" (Logic Anchors)
        # 根据数学事实，提示 LLM 应该用什么样的逻辑来解释
        logic_hints = []
        if spatial_facts.get('is_nearest'):
            logic_hints.append("emphasize minimizing immediate travel cost/distance")
        if spatial_facts.get('is_unique_direction'):
            logic_hints.append("highlight the directional exploration value")
        if spatial_facts.get('in_dense_cluster'):
             logic_hints.append("mention the potential for serving multiple nearby targets (high density)")
        elif spatial_facts.get('is_isolated'):
             logic_hints.append("mention capturing an uncovered/remote area")
             
        hint_str = "; ".join(logic_hints) if logic_hints else "focus on general spatial feasibility"

        # 2. 构建 Prompt
        # 关键点：Explicitly mapping Facts -> Decision
        system_instruction = (
            "You are an Operations Research expert annotating a dataset. "
            "Your task is to provide a strictly logical, concise rationalization for a known optimal spatial decision. "
            "Do NOT use phrases like 'I think' or 'Maybe'. State the reason as a fact."
        )

        user_prompt = (
            f"**Context:** Solving a {problem_context} problem.\n"
            f"**Observation:** {verified_desc}\n"
            f"**Decision:** The optimal action is Node {gt_id}.\n\n"
            f"**Task:** Write the <Thought> component for a Chain-of-Thought process.\n"
            f"Explain WHY Node {gt_id} is the best choice based on the observation.\n"
            f"**Reasoning constraints:**\n"
            f"- {hint_str}.\n" # 动态注入逻辑提示
            f"- Keep it extremely concise (1 short sentence, max 25 words).\n"
            f"- Use 'Because' or implies causality directly.\n"
            f"- Do not explicitly mention 'spatial facts' or 'data', just use the logic.\n"
        )
        
        try:
            # 这里的 temperature 建议设低一点 (0.1)，保证逻辑稳定性
            response = self.llm.generate(
                system_prompt=system_instruction,
                text=user_prompt,
                temperature=0.7
            )
            
            # Post-processing to clean up format
            clean_response = response.strip().replace("Thought:", "").replace("<Thought>", "")
            return clean_response

        except Exception as e:
            print(f"Logic Injection Failed: {e}")
            # Fallback Logic: 使用最安全的通用逻辑
            return f"Node {gt_id} is selected because it offers the most favorable spatial trade-off based on the current distribution."

# ==========================================
# Module 4: Refinement
# ==========================================

import re

class RefinementModule:
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def assemble_cot(self, verified_desc: str, logic_reasoning: str, trajectory_action: str) -> str:
        """
        Assembles the final CoT.
        Optimization: Uses LLM to polish content ONLY, then wraps tags via Python to ensure 100% format validity.
        """
        
        # 1. 准备 Decision 字符串 (Python 处理比 LLM 更稳)
        decision_str = f"\\boxed{{{trajectory_action}}}"
        
        # 2. 构造 Prompt：只让 LLM 负责“洗”文本，不让它负责“包”标签
        # 这样可以避免 LLM 忘记闭合标签或搞错 XML 格式
        system_instruction = (
            "You are a concise technical editor. "
            "Rewrite the provided Observation and Thought to be brief, factual, and strictly aligned. "
            "Remove all filler phrases like 'Based on the image', 'I can see', etc. "
            "Output exactly two lines separated by a '|||' delimiter."
        )

        user_prompt = (
            f"Input Data:\n"
            f"1. [Raw Observation]: {verified_desc}\n"
            f"2. [Raw Thought]: {logic_reasoning}\n\n"
            f"Task:\n"
            f"- Rewrite [Raw Observation] to be a dense, geometric summary (max 30 words).\n"
            f"- Rewrite [Raw Thought] to directly explain the decision based on that summary (max 20 words).\n"
            f"- Ensure NO redundancy between them.\n"
            f"- Output format: <Refined Observation> ||| <Refined Thought>"
        )

        try:
            # 3. LLM Generation
            response = self.llm.generate(
                system_prompt=system_instruction, # 保持与之前模块参数命名一致
                text=user_prompt,               # 保持与之前模块参数命名一致
                max_tokens=150,                        # 给够 token，防止截断
                temperature=0.1                        # 低温保证格式稳定
            )
            
            # 4. 解析与封装 (Python 逻辑)
            # 尝试按分隔符分割
            if "|||" in response:
                parts = response.split("|||")
                obs_clean = parts[0].strip()
                thought_clean = parts[1].strip()
            else:
                # Fallback: 如果 LLM 没按格式输出，尝试用换行符或其他启发式分割
                # 或者直接使用原始文本 (Safety net)
                print(f"Refinement Warning: Separator missing in response: '{response}'")
                obs_clean = verified_desc
                thought_clean = logic_reasoning

            # 5. 最终组装 (绝对安全的 XML 格式)
            final_cot = (
                f"<Observation> {obs_clean} </Observation>\n"
                f"<Thought> {thought_clean} </Thought>\n"
                f"<Decision> {decision_str} </Decision>"
            )
            return final_cot

        except Exception as e:
            print(f"Refinement Module Failed: {e}")
            # Fallback to pure rule-based cleaning
            # 简单的规则清洗
            clean_obs = verified_desc.replace("I can see", "").replace("The image shows", "").strip()
            clean_thought = logic_reasoning.replace("Based on", "").strip()
            
            return (
                f"<Observation> {clean_obs} </Observation>\n"
                f"<Thought> {clean_thought} </Thought>\n"
                f"<Decision> {decision_str} </Decision>"
            )

# ==========================================
# Module 5: Main Pipeline
# ==========================================
option2idx = {
    'A': 0,
    'B': 1,
    'C': 2,
    'D': 3,
    'E': 4,
    'F': 5,
    'G': 6,
    'H': 7,
    'I': 8,
    'J': 9,
    'K': 10,
    'L': 11,
    'M': 12,
    'N': 13,
    'O': 14,
    'P': 15,
    'Q': 16,
    'R': 17,
    'S': 18,
    'T': 19,
    'U': 20,
    'V': 21,
    'W': 22,
    'X': 23,
    'Y': 24,
    'Z': 25
}


def process_single_trajectory(
    trajectory_steps: List[Dict[str, Any]], 
    node_coords: Dict[int, List[float]],
    vlm_agent: VLMAgent, 
    llm_agent: VLMAgent,
    debug_image_dir: str = None
) -> List[Dict[str, Any]]:
    """
    Processes a single trajectory (sequence of steps) to generate CoT data.
    
    Args:
        trajectory_steps: List of step dicts: {'obs': str, 'trajectory': str, 'image': str, 'candidates': list, ...}
        node_coords: Dictionary of node coordinates {id: [x, y]}
        vlm_agent: Agent for Perception Module
        llm_agent: Agent for Logic Injection Module
        debug_image_dir: Directory containing images (if needed for VLM)
        
    Returns:
        List of processed step dicts with 'cot' field added.
    """
    # Initialize Modules
    geo_engine = GeometryEngine()
    perception_module = PerceptionModule(vlm_agent)
    logic_module = LogicInjectionModule(llm_agent)
    refinement_module = RefinementModule(llm_agent)
    
    processed_steps = []
    
    # Track history for context if needed
    history_actions = []
    global NODE_COORDS
    NODE_COORDS = {int(k): tuple(v) for k, v in node_coords.items()}
    prev_node = 0

    for step_idx, step_data in enumerate(trajectory_steps):
        # Access data from step dict (User requested structure: step["trajectory"])
        obs_text = step_data.get('obs', '')
        # Parse \boxed{A} to get action index
        action_raw = str(step_data['trajectory'])
        
        # Determine action_id (Node ID)
        # Try to parse candidates if available in step
        candidates_list = step_data.get('candidates', [])
        
        if candidates_list and "\\boxed{" in action_raw:
             #将\boxed{A} 变成0
            action_idx = option2idx[action_raw.replace("\\boxed{", "").replace("}", "")]
            # Simple extraction assuming action is "Node X" or just "X"
            if action_idx < len(candidates_list):
                action_id = candidates_list[action_idx]
            else:
                action_id = 0 # Fallback
        else:
             # If action_raw is already a Node ID
             try:
                 action_id = int(action_raw)
             except:
                 action_id = 0

        # 2. Determine Current Position
        # If node 0 is not in coords, use (0.5, 0.5)
        current_pos = NODE_COORDS.get(prev_node, (0.5, 0.5))
        prev_node = action_id
        
        # 3. Module 1: Geometry Analysis
        spatial_facts = geo_engine.analyze(current_pos, candidates_list, action_id)
        
        # 4. Prepare Image (Base64)
        image_b64 = step_data.get('image', None)
        
        # 5. Module 2: Perception Loop
        verified_desc = perception_module.run_perception_loop(image_b64, spatial_facts)
        
        # 6. Module 3: Logic Injection
        logic_reasoning = logic_module.inject_logic(verified_desc, spatial_facts, action_raw)
        
        # 7. Module 4: Refinement
        final_cot = refinement_module.assemble_cot(verified_desc, logic_reasoning, action_raw)
        
        # Store Result
        new_step = step_data.copy()
        new_step['cot'] = final_cot
        new_step['spatial_facts'] = spatial_facts # Optional: Keep for debugging
        processed_steps.append(new_step)
        
        history_actions.append(action_raw)
        print(f"Step {step_idx} processed. GT Action: {action_raw}")

    return processed_steps


def main(input_file: str, output_file: str, loc_file: str = None, debug_img_dir: str = None):
        
    # Load Trajectories
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    # Initialize Agents (Mock or Real)
    # In real usage, these would be initialized with model paths
    # vlm_agent = VLMAgent(model_path="...") 
    # llm_agent = VLMAgent(model_path="...")
    
    try:
        print("Initializing Real VLMAgent connected to https://api.siliconflow.cn/v1...")
        # Use the configuration as specified in Qwen3_single_worker_tsp_cot.py
        vlm_agent = VLMAgent(
            api_key="sk-mlqqrnvqurprnmxxpatvhllckaogtckcajwxehrngcysjgmo",
            api_base_url="https://api.siliconflow.cn/v1",
            model_name="Qwen/Qwen3-VL-32B-Thinking"
        )
        response = vlm_agent.generate(system_prompt="Hello, I am a student.", text="Hello, I am a student.")
        print(response)
        print("✓ Real VLMAgent initialized.")

        # Initialize Mock Agent for Logic Injection Module
        llm_agent = LLMAgent(
            api_key="sk-mlqqrnvqurprnmxxpatvhllckaogtckcajwxehrngcysjgmo",
            api_base_url="https://api.siliconflow.cn/v1",
            model_name="Qwen/Qwen3-30B-A3B-Thinking-2507"
        )
        response = llm_agent.generate(system_prompt="Hello, I am a student.", text="Hello, I am a student.")
        print(response)
        print("✓ Mock LLMAgent initialized.")
    except Exception as e:
        print(f"Failed to initialize real VLMAgent: {e}")
        print("Falling back to MockAgent.")
        class MockAgent:
            def generate(self, text, image=None, **kwargs):
                if "Verified Environment Description" in text: # LLM Logic
                     return "Based on verified desc, it is the nearest neighbor in the NE sector."
                elif "Please format the following information" in text: # Refinement Logic
                     return "<Observation> Verified </Observation>\n<Thought> Reasoned </Thought>\n<Decision> \\boxed{0} </Decision>"
                else: # VLM Perception
                     return "The target is in the North-East direction."
        agent = MockAgent()
     
    all_processed_data = []
    
    if isinstance(data, list) and len(data) > 0:
        print(f"Loaded {len(data)} trajectories from {input_file}")
        
        # Process first trajectory for testing/verification
        traj_idx = 0
        traj_data = data[traj_idx]
        
        # 1. Prepare Node Coords (Mock if missing)
        node_coords = traj_data.get("node_coords", {})
        if not node_coords:
            node_coords = {}
            for i in range(300):
                x = random.uniform(0.0, 1.0)
                y = random.uniform(0.0, 1.0)
                node_coords[i] = [x, y]
        
        # 2. Prepare Candidates (Mock if missing)
        # Check if candidates exist and match length
        traj_len = len(traj_data.get("trajectory", []))
        candidates_matrix = traj_data.get("candidates", [])
        if not candidates_matrix or len(candidates_matrix) != traj_len:
            candidates_matrix = []
            for i in range(traj_len):
                candidate = np.random.randint(0, 300, 12).tolist()
                candidates_matrix.append(candidate)
        
        # 3. Reshape into List[Dict] (The user's requested structure: traj_data[step]["trajectory"])
        trajectory_steps = []
        obs_list = traj_data.get("obs_list", [""] * traj_len)
        image_list = traj_data.get("image_list", [None] * traj_len)
        
        for i in range(traj_len):
            step_dict = {
                "trajectory": traj_data["trajectory"][i], # The Action
                "obs": obs_list[i] if i < len(obs_list) else "",
                "image": image_list[i] if image_list and i < len(image_list) else None,
                "candidates": candidates_matrix[i],
                "step_idx": i,
                "trajectory_idx": traj_idx # Add context
            }
            trajectory_steps.append(step_dict)

        # Run Pipeline
        processed_steps = process_single_trajectory(
            trajectory_steps,
            node_coords,
            vlm_agent, # VLM Agent
            llm_agent, # LLM Agent
            debug_img_dir
        )
        
        # Flatten: Extend the main list with these steps
        all_processed_data.extend(processed_steps)
        
        # Save Output
        with open(output_file, 'w') as f:
            json.dump(all_processed_data, f, indent=2)
        print(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    # Default paths
    input_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/flp_agent_output.json"
    output_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/flp_cot_dataset.json"
    
    # Run
    main(input_path, output_path)
