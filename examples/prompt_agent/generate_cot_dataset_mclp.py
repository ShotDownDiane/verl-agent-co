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

import dataclasses

class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if hasattr(o, 'to_json'):
            return o.to_json()
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)

# ==========================================
# Global Constants & Data
# ==========================================
NODE_COORDS: Dict[int, Tuple[float, float]] = {}

option2idx = {
    'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14,
    'P': 15, 'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21,
    'W': 22, 'X': 23, 'Y': 24, 'Z': 25
}
idx2option = {v: k for k, v in option2idx.items()}

class DualLogger:
    """Redirects stdout to both terminal and a file."""
    def __init__(self, filepath, stream=sys.stdout):
        self.terminal = stream
        self.log = open(filepath, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# ==========================================
# Module 1: Geometry Engine
# ==========================================

import math
import numpy as np
from scipy.spatial import ConvexHull
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Literal, Optional

# ==========================================
# Data Structures for Structured Facts
# ==========================================

@dataclass
class GeneralFactors:
    """宏观态势因子：用于描述 '我在哪' 和 '局势如何'"""
    # Localization
    norm_pos: Tuple[float, float]    # (0.0-1.0) 归一化坐标
    quadrant: str                    # e.g., "North-West", "South-East"
    depot_direction: str             # 起点在我的哪个方位
    depot_dist_ratio: float          # 离起点的相对距离 (0-1)
    
    # Situational Awareness
    progress_ratio: float            # 任务进度 (0.0-1.0)
    distribution_type: str           # "Clustered", "Uniform", "Mixed"
    is_in_dense_region: bool         # 当前是否处于高密度区

    def to_json(self):
        return dataclasses.asdict(self)

@dataclass
class MCLPFactors:
    """MCLP 微观因子：用于支撑 '最大覆盖' 策略"""
    marginal_gain: int               # 净增益：选该点能新覆盖多少需求
    raw_density: int                 # 原始密度：半径 R 内总共有多少点
    redundancy_ratio: float          # 重叠率：(原始密度 - 净增益) / 原始密度
    is_corner_case: bool             # 是否属于孤立点 (Corner Case)

    def to_json(self):
        return dataclasses.asdict(self)

@dataclass
class CandidateMeta:
    """单个候选点的完整几何元数据"""
    id: int
    label: str                       # e.g., "A", "B"
    coords: Tuple[float, float]
    dist: float
    direction: str                   # 视觉方位 (N, NW...)
    angle: float                     # 绝对角度 (0-360)
    mclp_factors: Optional[MCLPFactors] = None

    def to_json(self):
        return dataclasses.asdict(self)

# ==========================================
# Core Engine
# ==========================================

class TaskContextManager:
    """
    管理不同运筹任务的'世界观'。
    """
    PROFILES = {
        "MCLP": {
            "role_def": "You are a strategic planner solving the Maximal Covering Location Problem (MCLP).",
            "objective": "Select facility locations to maximize the total number of covered customers within a fixed service radius.",
            "visual_focus": "Focus on high-density clusters, coverage overlaps, and isolated points.",
            "strategy_keywords": ["Max Coverage", "Redundancy Reduction", "Corner Case Handling"]
        }
    }

    @classmethod
    def get_system_prompt(cls, task_type: str) -> str:
        profile = cls.PROFILES.get(task_type, cls.PROFILES["MCLP"])
        return (
            f"{profile['role_def']}\n"
            f"**Objective**: {profile['objective']}\n"
            f"**Visual Priority**: {profile['visual_focus']}\n"
            f"**Key Strategies**: {', '.join(profile['strategy_keywords'])}.\n"
            f"Act strictly according to these rules."
        )

class GeometryEngine:
    def __init__(self, 
                 coords: np.ndarray, 
                 depot_idx: int = 0, 
                 global_bounds: Tuple[float, float, float, float] = None):
        """
        Args:
            coords: Shape (N, 2) array of all node coordinates.
            depot_idx: Index of the start/depot node.
            global_bounds: Optional (min_x, min_y, max_x, max_y) for static normalization.
        """
        self.coords = coords
        self.coords = np.array(list(coords.values()))
        self.depot_idx = depot_idx
        self.total_nodes = self.coords.shape[0]
        
        # 1. 预计算全局边界 (用于宏观定位)
        if global_bounds:
            self.min_x, self.min_y, self.max_x, self.max_y = global_bounds
        else:
            self.min_x, self.min_y = np.min(self.coords, axis=0)
            self.max_x, self.max_y = np.max(self.coords, axis=0)
        
        self.span_x = max(self.max_x - self.min_x, 1e-6)
        self.span_y = max(self.max_y - self.min_y, 1e-6)
        
        # 2. 预计算全局分布类型 (Clustered/Uniform)
        self.global_distribution, self.global_avg_nnd = self._analyze_global_distribution()

        # 方位角定义
        self.compass_sectors = {
            'N': 90, 'NE': 45, 'E': 0, 'SE': 315,
            'S': 270, 'SW': 225, 'W': 180, 'NW': 135
        }

    def _get_coords(self, idx: int) -> np.ndarray:
        return self.coords[idx]

    # --- Global Analysis Helpers ---

    def _analyze_global_distribution(self) -> Tuple[str, float]:
        """计算 Clark-Evans Index 判定全局分布类型"""
        if self.total_nodes < 5:
            return "Uniform", 0.5
            
        nbrs = NearestNeighbors(n_neighbors=2).fit(self.coords)
        distances, _ = nbrs.kneighbors(self.coords)
        mean_nnd = np.mean(distances[:, 1]) # Observed Mean Distance
        
        area = self.span_x * self.span_y
        density = self.total_nodes / area
        expected_nnd = 0.5 / np.sqrt(density) # Expected Mean for Random
        
        # Aggregation Index R
        R = mean_nnd / expected_nnd
        
        dist_type = "Mixed"
        if R < 0.7: dist_type = "Clustered"
        elif R > 1.2: dist_type = "Uniform"
        
        return dist_type, mean_nnd

    def _get_quadrant_desc(self, pos: np.ndarray) -> str:
        """返回象限描述 (用于 CoT 定位)"""
        nx = (pos[0] - self.min_x) / self.span_x
        ny = (pos[1] - self.min_y) / self.span_y
        ns = "South" if ny < 0.5 else "North"
        we = "West" if nx < 0.5 else "East"
        return f"{ns}-{we}"

    def _check_density(self, current_pos: np.ndarray) -> bool:
        """判断当前是否在高密度区"""
        # 简单 heuristic: 检查最近的 3 个点的距离是否显著小于全局平均
        dists = np.linalg.norm(self.coords - current_pos, axis=1)
        dists.sort()
        # 跳过自己 (index 0)
        local_avg = np.mean(dists[1:4]) if len(dists) > 4 else 0
        return local_avg < (self.global_avg_nnd * 0.8)

    # --- Geometric Math Helpers ---

    def calculate_bearing(self, start: np.ndarray, end: np.ndarray) -> float:
        """0=East, 90=North, 0-360 degrees"""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        angle = math.degrees(math.atan2(dy, dx))
        return (angle + 360) % 360

    def get_compass_dir(self, start: np.ndarray, end: np.ndarray) -> str:
        angle = self.calculate_bearing(start, end)
        best_dir = 'Unknown'
        min_diff = 360.0
        for direction, center_angle in self.compass_sectors.items():
            diff = abs(angle - center_angle)
            diff = min(diff, 360 - diff)
            if diff < min_diff:
                min_diff = diff
                best_dir = direction
        return best_dir

    def get_angular_diff(self, a1: float, a2: float) -> float:
        diff = abs(a1 - a2)
        return min(diff, 360 - diff)

    # --- Main Analysis Pipelines ---

    def analyze_general(self, current_idx: int, visited_count: int) -> GeneralFactors:
        """生成通用宏观因子"""
        curr_pos = self._get_coords(current_idx)
        depot_pos = self._get_coords(self.depot_idx)
        
        nx = (curr_pos[0] - self.min_x) / self.span_x
        ny = (curr_pos[1] - self.min_y) / self.span_y
        
        return GeneralFactors(
            norm_pos=(nx, ny),
            quadrant=self._get_quadrant_desc(curr_pos),
            depot_direction=self.get_compass_dir(curr_pos, depot_pos),
            depot_dist_ratio=np.linalg.norm(curr_pos - depot_pos) / np.sqrt(self.span_x**2 + self.span_y**2),
            progress_ratio=visited_count / self.total_nodes,
            distribution_type=self.global_distribution,
            is_in_dense_region=self._check_density(curr_pos)
        )

    def analyze_mclp_micro(self, 
                          covered_mask: np.ndarray,
                          candidate_ids: List[int],
                          radius: float) -> Dict[int, MCLPFactors]:
        """
        Generate MCLP micro factors for all candidates.
        Core Logic: Net Gain, Raw Density, Redundancy, Corner Case
        """
        if not candidate_ids:
            return {}

        all_coords = self.coords
        results = {}
        cand_coords_list = [self._get_coords(cid) for cid in candidate_ids]
        cand_coords = np.array(cand_coords_list) # (M, 2)
        
        # We need to calculate factors for each candidate
        # 1. Raw Density: Points within radius
        # 2. Marginal Gain: Points within radius AND NOT covered
        
        # To do this efficiently:
        # Distance matrix between candidates and ALL nodes
        # shape: (M, N)
        # This might be heavy if N is large. M is usually small (20-30). N ~ 50-100.
        # So (30, 100) is fine.
        
        # Expand dims for broadcasting
        # cand_coords: (M, 1, 2)
        # all_coords: (1, N, 2)
        dists = np.linalg.norm(cand_coords[:, np.newaxis, :] - all_coords[np.newaxis, :, :], axis=2)
        
        # Boolean mask of coverage by candidate: (M, N)
        covered_by_cand = dists <= radius
        
        for i, cid in enumerate(candidate_ids):
            # Boolean array for this candidate
            coverage_mask = covered_by_cand[i] # (N,)
            
            # 1. Raw Density
            raw_density = int(np.sum(coverage_mask))
            
            # 2. Marginal Gain (Net Gain)
            # Count points that are covered by this candidate AND NOT currently covered
            # currently_covered is passed as covered_mask (N,)
            newly_covered_mask = coverage_mask & (~covered_mask)
            marginal_gain = int(np.sum(newly_covered_mask))
            
            # 3. Redundancy Ratio
            # (Raw Density - Marginal Gain) / Raw Density
            # If Raw Density is 0 (shouldn't happen for candidate itself usually, but possible if radius is tiny), result 0
            if raw_density > 0:
                redundancy_ratio = (raw_density - marginal_gain) / raw_density
            else:
                redundancy_ratio = 0.0
                
            # 4. Is Corner Case
            # Definition: Isolated point, hard to be covered incidentally.
            # Heuristic: 
            # - Marginal Gain is small (but > 0)
            # - Raw Density is small (e.g. < 3)
            # - Maybe check if it's far from the "centroid" of the map?
            # Let's stick to the user's description: "很难被顺带覆盖的孤立点"
            # Low Raw Density implies it's not in a cluster.
            is_corner_case = (raw_density <= 2) and (marginal_gain > 0)
            
            results[cid] = MCLPFactors(
                marginal_gain=marginal_gain,
                raw_density=raw_density,
                redundancy_ratio=float(redundancy_ratio),
                is_corner_case=is_corner_case
            )
            
        return results

    # --- Master Analysis Interface ---

    def analyze_step(self, 
                     current_idx: int, 
                     candidate_ids: List[int], 
                     gt_node_id: int, 
                     covered_mask: np.ndarray = None, # New Argument
                     radius: float = 0.1,             # New Argument
                     visited_count: int = 0) -> Dict[str, Any]:
        """
        Main entry point for generating all geometric truths for a step.
        """
        curr_pos = self._get_coords(current_idx)
        gt_pos = self._get_coords(gt_node_id)
        
        # 1. Macro Analysis
        general_facts = self.analyze_general(current_idx, visited_count)
        
        # 2. Micro Analysis (MCLP)
        # Filter out padding (-1)
        valid_cands = [c for c in candidate_ids if c != -1]
        
        # Default to all False if None
        if covered_mask is None:
            covered_mask = np.zeros(self.total_nodes, dtype=bool)

        # Switch to MCLP Analysis
        mclp_facts_map = self.analyze_mclp_micro(covered_mask, valid_cands, radius)
        
        # 3. Assemble Candidates Metadata
        candidates_meta = []
        min_dist = float('inf') 
        
        for i, cid in enumerate(valid_cands):
            c_pos = self._get_coords(cid)
            dist = np.linalg.norm(c_pos - curr_pos)
            if dist < min_dist: min_dist = dist
            
            label = idx2option.get(i, f"UNK_{i}")

            candidates_meta.append(CandidateMeta(
                id=cid,
                label=label,
                coords=tuple(c_pos),
                dist=float(dist),
                direction=self.get_compass_dir(curr_pos, c_pos),
                angle=self.calculate_bearing(curr_pos, c_pos),
                mclp_factors=mclp_facts_map[cid] # Enable MCLP
            ))

        # 4. GT Specifics
        gt_dist = np.linalg.norm(gt_pos - curr_pos)
        is_nearest = abs(gt_dist - min_dist) < 1e-6
        
        # Temptation Logic for MCLP
        # Greedy Choice = Highest Marginal Gain
        sorted_by_gain = sorted(candidates_meta, key=lambda x: x.mclp_factors.marginal_gain, reverse=True)
        
        temptation_id = None
        if len(sorted_by_gain) > 0:
            best_gain_cand = sorted_by_gain[0]
            if best_gain_cand.id != gt_node_id:
                temptation_id = best_gain_cand.id
            elif len(sorted_by_gain) > 1:
                # GT is the best, so next best?
                temptation_id = sorted_by_gain[1].id

        return {
            "general": general_facts,
            "candidates": candidates_meta,
            "gt_id": gt_node_id,
            "gt_stats": {
                "dist": gt_dist,
                "dir": self.get_compass_dir(curr_pos, gt_pos),
                "is_nearest": is_nearest,
                "mclp_factors": mclp_facts_map.get(gt_node_id) # MCLP
            },
            "temptation_id": temptation_id
        }
# ==========================================
# Module 2: Perception Loop
# ==========================================
from typing import Dict, Any, Tuple, Optional, List
import re

class PerceptionModule:
    def __init__(self, vlm_agent):
        self.vlm = vlm_agent
        
        # --- 1. 词表扩展 ---
        self.direction_map = {
            'N': ['north', 'top', 'upper', 'above'],
            'S': ['south', 'bottom', 'lower', 'below'],
            'E': ['east', 'right'],
            'W': ['west', 'left'],
            'NE': ['north-east', 'top-right', 'upper-right'],
            'NW': ['north-west', 'top-left', 'upper-left'],
            'SE': ['south-east', 'bottom-right', 'lower-right'],
            'SW': ['south-west', 'bottom-left', 'lower-left']
        }
        
        # 几何特征关键词 (MCLP Specific)
        # Cluster (簇), Overlap (重叠), Isolated (孤立), Coverage (覆盖)
        self.cluster_keywords = ['cluster', 'group', 'dense', 'crowded', 'pack', 'cloud', 'concentration', 'density']
        self.overlap_keywords = ['overlap', 'redundant', 'covered', 'shared', 'double', 'intersect']
        self.isolated_keywords = ['isolated', 'corner', 'remote', 'alone', 'outlier', 'orphan']
        self.coverage_keywords = ['cover', 'reach', 'service', 'serve', 'range', 'capture']
        
        # New keywords based on user request
        self.unserved_keywords = ['unserved', 'new', 'fresh', 'uncovered', 'capture', 'gain']
        self.circle_keywords = ['circle', 'radius', 'zone', 'area', 'range']

    def _check_keywords(self, text: str, keywords: List[str]) -> bool:
        text_lower = text.lower()
        return any(k in text_lower for k in keywords)

    def construct_grounding_prompt(self, facts: Dict[str, Any]) -> str:
        """
        根据 GeometryEngine 的输出，动态构建“引导式”Prompt。
        目标：告诉 VLM 真值，让它翻译成视觉描述。
        """
        gt_id = facts['gt_id']
        gt_stats = facts['gt_stats']
        mclp_factors = gt_stats['mclp_factors'] # dataclass
        general = facts['general'] # dataclass
        temptation_id = facts.get('temptation_id')

        # Helper to format ID
        def get_fmt(cid):
            cand = next((c for c in facts['candidates'] if c.id == cid), None)
            lbl = cand.label if cand else "?"
            return f"Option {lbl} [Node {cid}]"

        gt_str = get_fmt(gt_id)

        # 1. 基础定位
        prompt = (
            f"Please generate a precise visual observation for **{gt_str}**.\n"
            f"Context: Global distribution is {general.distribution_type}. "
            f"We are currently in the {general.quadrant} quadrant.\n\n"
            f"Ground Truths to Describe:\n"
            f"1. **Location**: {gt_str} is to the **{gt_stats['dir']}**.\n"
        )

        # 2. 几何特征引导 (MCLP Specifics - User Requested Questions)
        
        # Visual Density: "Does a circle centered here capture many red dots?"
        if mclp_factors.raw_density > 3:
            prompt += f"2. **Visual Density**: (Answer to 'Does a circle centered here capture many red dots?') -> Yes, it centers on a **dense group** of unserved points.\n"
        else:
            prompt += f"2. **Visual Density**: (Answer to 'Does a circle centered here capture many red dots?') -> It captures a small or moderate number of points.\n"

        # Separation: "Is this circle distinct from existing green circles, or does it overlap significantly?"
        if mclp_factors.redundancy_ratio < 0.2:
            prompt += f"3. **Separation**: (Answer to 'Is this circle distinct from existing green circles?') -> Yes, it is **distinct** with minimal overlap.\n"
        elif mclp_factors.redundancy_ratio > 0.5:
            prompt += f"3. **Separation**: (Answer to 'Does it overlap significantly?') -> Yes, it shares significant space with existing circles (**Overlap**).\n"
        else:
             prompt += f"3. **Separation**: (Answer to 'Is there overlap?') -> There is partial overlap with existing circles.\n"

        # Boundary: "Is it covering the core of the cluster or just clipping the edge?"
        if mclp_factors.marginal_gain >= 4:
             prompt += f"4. **Boundary**: (Answer to 'Is it covering the core?') -> It covers the **core** of the cluster (High Gain).\n"
        elif mclp_factors.is_corner_case:
             prompt += f"4. **Boundary**: (Answer to 'Is it covering the core?') -> It targets an **isolated outlier** (Corner Case).\n"
        else:
             prompt += f"4. **Boundary**: (Answer to 'Is it covering the core?') -> It clips the edge or covers a smaller group.\n"

        # 3. 诱惑点对比 (最重要的逻辑！)
        if not gt_stats['is_nearest'] and temptation_id is not None:
            temp_str = get_fmt(temptation_id)
            prompt += (
                f"5. **Comparison**: Explicitly mention that **{temp_str}** might seem attractive (e.g., higher raw density), "
                f"but explain that {gt_str} is chosen for its strategic value (e.g., better net gain or unique coverage).\n"
            )
        
        prompt += "\nOutput a concise but detailed observation paragraph confirming these visual facts."
        return prompt

    def verify_description(self, description: str, facts: Dict[str, Any]) -> Tuple[bool, str]:
        """
        全方位验证逻辑：ID -> 方位 -> 几何特征 -> 对比逻辑
        """
        desc_lower = description.lower()
        gt_id = str(facts['gt_id'])
        gt_stats = facts['gt_stats']
        mclp_factors = gt_stats['mclp_factors']
        temptation_id = facts.get('temptation_id')

        # --- 1. Identity Check ---
        if not re.search(rf"\b(candidate|node|option)\s*{re.escape(gt_id)}\b", desc_lower):
             return False, f"Failed to explicitly mention target 'Candidate {gt_id}'."

        # --- 2. Direction Check ---
        gt_dir = gt_stats['dir']
        expected_kws = self.direction_map.get(gt_dir, [])
        if not self._check_keywords(desc_lower, expected_kws):
            return False, f"Failed to identify location '{gt_dir}'. Expected keywords: {expected_kws}"

        # --- 3. Geometric Feature Check (MCLP) ---
        # A. High Density / Cluster Check
        if mclp_factors.raw_density > 3:
            if not self._check_keywords(desc_lower, self.cluster_keywords):
                return False, f"Missed density feature: Target is in a 'Cluster/Group', but description didn't mention it."

        # B. Corner Case Check
        if mclp_factors.is_corner_case:
            if not self._check_keywords(desc_lower, self.isolated_keywords):
                return False, f"Missed isolation feature: Target is a 'Corner Case/Isolated', but description missed it."

        # C. Overlap / Separation Check
        if mclp_factors.redundancy_ratio > 0.5:
             if not self._check_keywords(desc_lower, self.overlap_keywords):
                 return False, f"Missed overlap feature: Target has significant overlap, but description didn't mention it."
        elif mclp_factors.redundancy_ratio < 0.2 and mclp_factors.raw_density > 3:
             # If efficient, maybe check for 'distinct' or 'unserved' or 'minimal overlap'
             # Let's check for lack of 'heavy overlap' words or presence of positive efficiency words?
             # For now, let's just ensure they don't say it's "highly redundant".
             pass 

        # --- 4. Temptation/Comparison Check (Crucial for SFT) ---
        if temptation_id is not None and not gt_stats['is_nearest']:
            temp_id_str = str(temptation_id)
            if not re.search(rf"\b{re.escape(temp_id_str)}\b", desc_lower):
                return False, f"Failed to compare with the greedy temptation 'Candidate {temptation_id}'."

        return True, "Verified"

    def construct_reflexion_prompt(self, gt_id: int, previous_response: str, missing_reason: str, label: str = "?") -> str:
        """
        构造'反思'提示词：展示错误答案 + 强行注入真值
        """
        gt_str = f"Option {label} [Node {gt_id}]"
        return (
            f"### Review of your previous output:\n"
            f"**Your Draft**: \"{previous_response}\"\n"
            f"**Critique**: This description is incomplete. {missing_reason}\n\n"
            
            f"### New Task:\n"
            f"Please **REWRITE** the observation for {gt_str}.\n"
            f"1. Keep the correct location details from your draft.\n"
            f"2. **MANDATORY**: Integrate the missing fact mentioned in the critique.\n"
            f"3. Ensure the tone remains objective and observational."
        )

    def run_perception_loop(self, image_b64: str, spatial_facts: Dict[str, Any], max_retries: int = 3) -> Optional[str]:
        gt_id = spatial_facts['gt_id']
        
        # Find Label for Reflexion
        gt_cand = next((c for c in spatial_facts['candidates'] if c.id == gt_id), None)
        gt_label = gt_cand.label if gt_cand else "?"

        # 1. 第一次尝试：标准引导 Prompt
        current_prompt = self.construct_grounding_prompt(spatial_facts)
        task_system_prompt = TaskContextManager.get_system_prompt("FLP")
        system_instruction = (
            f"{task_system_prompt}\n\n"
            "Your specific sub-task is: Ground the geometric truths provided by the engine into a visual description. "
            "Use the terminology appropriate for this specific problem type."
        )

        # 用于存储对话历史（如果模型支持多轮对话更好，单轮就用 Prompt 拼接）
        # 这里演示 Prompt 拼接方式，适配性更广
        
        for attempt in range(max_retries):
            print(f"--- Attempt {attempt + 1} ---")
            
            response = self.vlm.generate(
                system_prompt=system_instruction,
                text=current_prompt, 
                image=image_b64,
                max_tokens=2048
            )
            print(f"Prompt: {current_prompt}")
            print("************")
            print(response)
            # 2. 验证
            is_valid, reason = self.verify_description(response, spatial_facts)
            
            if is_valid:
                return response # 成功
            
            # 3. 失败：进入 Reflexion 模式
            # print(f"  -> Failed: {reason}")
            
            # 这里的关键是：不要让它重头猜，而是基于它刚才的回答进行修改
            # 下一次循环使用的 Prompt 变更为 'Critique Prompt'
            current_prompt = self.construct_reflexion_prompt(
                gt_id=gt_id,
                previous_response=response, # 把它的错误答案喂回去
                missing_reason=reason,      # 把你的 python 验证结果喂回去
                label=gt_label
            )
            
        return None # 彻底失败

# ==========================================
# Module 3: Logic Injection
# ==========================================
from typing import Dict, Any, Optional


class LogicInjectionModule:
    """
    Dedicated Logic Injection for FLP (Facility Location Problem).
    Orchestrates the 'Strategic Narrative' by fusing:
    1. Visual Perception (from VLM)
    2. Hard Data / Dashboard (from Text Observation)
    3. Geometric Rules (from GeometryEngine)
    """
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def _get_node_fmt(self, node_id: int, facts: Dict[str, Any]) -> str:
        """Helper: Option A [Node 3]"""
        cand = next((c for c in facts['candidates'] if c.id == node_id), None)
        lbl = cand.label if cand else "?"
        return f"Option {lbl} [Node {node_id}]"

    def _select_strategic_narrative(self, facts: Dict[str, Any]) -> Dict[str, str]:
        """
        [Director]: Selects the MCLP strategy script based on geometric factors.
        Implements 3 Scripts: Greedy Sweep, Complementary Expansion, Diminishing Returns.
        """
        gt_id = facts['gt_id']
        gt_stats = facts['gt_stats']
        mclp_factors = gt_stats['mclp_factors'] # MCLP Specific
        temptation_id = facts.get('temptation_id')
        
        gt_str = self._get_node_fmt(gt_id, facts)
        gain = mclp_factors.marginal_gain
        redundancy = mclp_factors.redundancy_ratio
        
        # Helper for Temptation string
        temp_str = "the temptation point"
        if temptation_id is not None:
            temp_str = self._get_node_fmt(temptation_id, facts)

        # 🎭 Script C: Diminishing Returns / Rejection (Late Stage / Negative Sample)
        # Condition: GT is NOT the visually nearest/densest, meaning we skipped a "Honey Pot".
        if not gt_stats['is_nearest'] and temptation_id is not None:
             narrative = {
                "strategy_name": "Diminishing Returns",
                "reasoning_focus": (
                    f"Visually, {temp_str} is in a dense area. However... "
                    f"Dashboard reveals a low Marginal Gain because most nodes are already served (High Overlap). "
                    f"In contrast, {gt_str} offers a clean Gain of {gain}."
                ),
                "data_citation": f"...Dashboard reveals low Gain for {temp_str} vs Gain {gain} for {gt_str}...",
                "conflict_handling": f"Rejecting this Redundant Move ({temp_str}) in favor of {gt_str}."
            }

        # 🎭 Script B: Complementary Expansion (Mid Stage)
        # Condition: Moderate Gain but Very Low Redundancy (Efficient expansion)
        elif redundancy < 0.1: # Strict non-overlap
             narrative = {
                "strategy_name": "Complementary Expansion",
                "reasoning_focus": (
                    f"{gt_str} lies adjacent to the existing coverage, targeting the remaining fringe nodes. "
                    f"Strategy: Non-Overlapping Expansion to extend the service frontier."
                ),
                "data_citation": f"It offers a Gain of {gain} with near-zero redundancy.",
                "conflict_handling": "Zero redundancy expansion takes precedence."
            }

        # 🎭 Script A: Greedy Sweep (Early Stage) - DEFAULT
        # Condition: High Gain (or default fallback)
        else:
            narrative = {
                "strategy_name": "Greedy Sweep",
                "reasoning_focus": (
                    f"Visually, {gt_str} centers on a dense, unserved cluster. "
                    f"Strategy: Maximal Coverage to secure the highest demand block."
                ),
                "data_citation": f"Dashboard confirms a massive Marginal Gain of {gain}.",
                "conflict_handling": "Coverage maximization is the primary objective."
            }

        return narrative

    def inject_logic(self, 
                     verified_desc: str, 
                     text_dashboard: str, 
                     spatial_facts: Dict[str, Any], 
                     problem_context: str = "MCLP") -> str:
        """
        Generates the 'Thought' component with strict Data Fusion requirements.
        """
        gt_id = spatial_facts['gt_id']
        gt_str = self._get_node_fmt(gt_id, spatial_facts)
        
        narrative = self._select_strategic_narrative(spatial_facts)
        
        system_instruction = (
            "You are an expert Maximal Covering Location Problem (MCLP) solver. "
            "Synthesize the **Visual Observation** and the **Dashboard Data** into a coherent thought. "
            "Your reasoning must be grounded in both geometry (Visual) and numbers (Dashboard)."
        )

        user_prompt = (
            f"**Task**: Generate the <Thought> rationale for choosing **{gt_str}**.\n\n"
            
            f"**Input 1: Visual Observation** (The View out the Window)\n"
            f"\"{verified_desc}\"\n\n"
            
            f"**Input 2: Dashboard Data** (The Instruments)\n"
            f"```text\n{text_dashboard}\n```\n\n"
            
            f"**Input 3: Strategic Directive** (The Mission)\n"
            f"- **Strategy**: {narrative['strategy_name']}\n"
            f"- **Core Logic**: {narrative['reasoning_focus']}\n"
            f"- **Conflict/Trade-off**: {narrative['conflict_handling']}\n\n"
            
            f"**Requirements for <Thought>**:\n"
            f"1. **The Anchor**: Start by citing the Visual Observation (e.g., 'Visually, {gt_str} is...').\n"
            f"2. **The Verification**: You MUST cite the **'Marginal Gain'** (or Net Gain) from the text data to justify why {gt_str} is better than others.\n"
            f"   (Expected Citation: \"{narrative.get('data_citation', 'Dashboard confirms...')}\")\n"
            f"3. **The Strategy**: Explicitly mention the Strategy **'{narrative['strategy_name']}'**.\n"
            f"4. Max 200 words."
        )
        
        try:
            # Generate
            response = self.llm.generate(
                system_prompt=system_instruction,
                text=user_prompt,
                temperature=0.3,
                max_tokens=2048
            )
            return response.strip()

        except Exception as e:
            print(f"Logic Injection Failed: {e}")
            return f"Following the {narrative['strategy_name']} strategy, {gt_str} is the optimal choice."

# ==========================================
# Module 4: Refinement
# ==========================================
import re
from typing import Dict, Any

class RefinementModule:
    """
    Dedicated FLP Editor.
    Role: Site Selection Analyst / Network Planner.
    Goal: Polish Observation (Visual Geometry) and Thought (Coverage Strategy).
    """
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def _get_target_info(self, facts: Dict[str, Any]) -> tuple[str, str]:
        """
        Standardizes ID format.
        """
        gt_id = facts['gt_id']
        cand = next((c for c in facts['candidates'] if c.id == gt_id), None)
        label = cand.label if cand else "?" 
        
        display_fmt = f"Option {label} [Node {gt_id}]"
        boxed_content = label 
        
        return display_fmt, boxed_content

    def assemble_cot(self, 
                     verified_desc: str, 
                     logic_reasoning: str, 
                     facts: Dict[str, Any]) -> str:
        """
        Assembles the final MCLP Chain-of-Thought.
        
        Args:
            verified_desc: Output from PerceptionModule (Visuals: Cluster, Overlap...)
            logic_reasoning: Output from LogicInjectionModule (Logic: Gain, Redundancy...)
            facts: The master data dict
        """
        
        # 1. 准备标准格式
        target_fmt, boxed_val = self._get_target_info(facts)
        decision_str = f"\\boxed{{{boxed_val}}}"
        
        # 2. 构造 Prompt：定义 "Network Architect" 角色
        system_instruction = (
            "You are an expert Network Architect. "
            "Your task is to refine the reasoning for optimal facility placement (MCLP). "
            "Tone: Professional, Technical, Focused on 'Securing Coverage' and 'Efficiency'."
        )

        user_prompt = (
            f"### Raw Input Data\n"
            f"1. [Raw Observation]: {verified_desc}\n"
            f"2. [Raw Thought]: {logic_reasoning}\n"
            f"3. [Target Identity]: {target_fmt}\n\n"
            
            f"### Refinement Task\n"
            f"Rewrite the inputs into two distinct parts separated by '|||'.\n\n"
            
            f"**Part 1: <Observation>** (Spatial Context)\n"
            f"- Describe the geometric relationship (e.g., 'dense cluster', 'fringe area').\n"
            f"- **MANDATORY**: Use the exact format **'{target_fmt}'**.\n"
            f"- Max 40 words. No 'I can see'.\n\n"
            
            f"**Part 2: <Thought>** (Architectural Reasoning)\n"
            f"- Adopt the **Network Architect** persona using phrases like:\n"
            f"  - 'Securing coverage' (for high gain)\n"
            f"  - 'Minimizing signal overlap' (for low redundancy)\n"
            f"  - 'Capturing demand weight' (for density)\n"
            f"- **CRITICAL**: You MUST preserve ALL numbers (Marginal Gain, Radius, Density) from the Raw Thought.\n"
            f"- Max 200 words.\n\n"
            
            f"### Output Example\n"
            f"Option C [Node 45] identifies a dense unserved cluster. ||| By selecting this site, we are securing coverage for the high-demand zone. Dashboard confirms a Marginal Gain of 15, capturing significant demand weight while minimizing signal overlap.\n\n"
            
            f"### Your Output:"
        )

        try:
            # 3. LLM Generation
            response = self.llm.generate(
                system_prompt=system_instruction,
                text=user_prompt,
                max_tokens=2048,
                temperature=0.5 # Low temp for strict formatting
            )
            
            # 4. Parsing
            parts = re.split(r'\s*\|\|\|\s*', response.strip())
            
            if len(parts) >= 2:
                obs_clean = parts[0].strip()
                thought_clean = parts[1].strip()
            else:
                print(f"Refinement Warning: Separator missing. Using raw inputs.")
                obs_clean = verified_desc.strip()
                thought_clean = logic_reasoning.strip()

            # 5. Safety Cleaning
            obs_clean = re.sub(r'^(Observation|Part 1):?', '', obs_clean, flags=re.IGNORECASE).strip()
            thought_clean = re.sub(r'^(Thought|Part 2):?', '', thought_clean, flags=re.IGNORECASE).strip()

            final_cot = (
                f"<Observation> {obs_clean} </Observation>\n"
                f"<Thought> {thought_clean} </Thought>\n"
                f"<Decision> {decision_str} </Decision>"
            )
            return final_cot

        except Exception as e:
            print(f"Refinement Failed: {e}")
            return (
                f"<Observation> {verified_desc} </Observation>\n"
                f"<Thought> {logic_reasoning} </Thought>\n"
                f"<Decision> {decision_str} </Decision>"
            )
# ==========================================
# Module 5: Main Pipeline
# ==========================================


def process_single_trajectory(
    trajectory_steps: List[Dict[str, Any]], 
    node_coords: Dict[int, List[float]],
    vlm_agent: VLMAgent, 
    llm_agent: LLMAgent,
    debug_image_dir: str = None,
    radius: float = 0.15 # Default radius
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
    geo_engine = GeometryEngine(coords=node_coords)
    perception_module = PerceptionModule(vlm_agent)
    logic_module = LogicInjectionModule(llm_agent)
    refinement_module = RefinementModule(llm_agent)
    
    processed_steps = []
    
    # Track history for context if needed
    history_actions = []
    global NODE_COORDS
    NODE_COORDS = {int(k): tuple(v) for k, v in node_coords.items()}
    prev_node = -1
    current_idx = 0
    
    # Initialize Covered Mask
    covered_mask = np.zeros(geo_engine.total_nodes, dtype=bool)

    for step_idx, step_data in enumerate(trajectory_steps):
        # Access data from step dict (User requested structure: step["trajectory"])
        obs_text = step_data.get('obs', '')
        # Parse \boxed{A} to get action index
        action_raw = str(step_data['trajectory'])
        
        # Determine action_id (Node ID)
        # Try to parse candidates if available in step
        candidates_list = step_data.get('candidates', [])
        
        #将\boxed{A} 变成0
        action_idx = option2idx[action_raw.replace("\\boxed{", "").replace("}", "")]
        # Simple extraction assuming action is "Node X" or just "X"
        if action_idx < len(candidates_list):
            action_id = candidates_list[action_idx]
        else:
            action_id = 0 # Fallback

        # 2. Determine Current Position
        # If node 0 is not in coords, use (0.5, 0.5)
        # current_pos = NODE_COORDS.get(prev_node, (0.5, 0.5))
        # 3. Module 1: Geometry Analysis
        spatial_facts = geo_engine.analyze_step(
            current_idx=current_idx,
            candidate_ids=candidates_list,
            gt_node_id=action_id,
            covered_mask=covered_mask.copy(), # Pass copy to be safe
            radius=radius,
            visited_count=step_idx
        )

        # Update State (Covered Mask)
        action_pos = geo_engine._get_coords(action_id)
        dists = np.linalg.norm(geo_engine.coords - action_pos, axis=1)
        new_covered = dists <= radius
        covered_mask = covered_mask | new_covered
        
        prev_node = current_idx
        current_idx = action_id
        
        # 4. Prepare Image (Base64)
        image_b64 = step_data.get('image', None)
        
        # 5. Module 2: Perception Loop
        try:
            verified_desc = perception_module.run_perception_loop(image_b64, spatial_facts)
        except Exception as e:
            print(f"Step {step_idx} failed perception loop: {e}")
            continue
        
        if verified_desc is None:
            print(f"Step {step_idx} failed perception loop.")
            continue
        
        # 6. Module 3: Logic Injection
        logic_reasoning = logic_module.inject_logic(verified_desc, obs_text, spatial_facts, problem_context="MCLP")
        
        # 7. Module 4: Refinement
        final_cot = refinement_module.assemble_cot(verified_desc=verified_desc, logic_reasoning=logic_reasoning, facts=spatial_facts)
        
        # Store Result
        new_step = step_data.copy()
        new_step['cot'] = final_cot
        # new_step['spatial_facts'] = spatial_facts # Optional: Keep for debugging
        processed_steps.append(new_step)
        
        history_actions.append(action_raw)
        print(f"Step {step_idx} processed. GT Action: {action_raw}")
    return processed_steps


def main(input_file: str, output_file: str, loc_file: str = None, debug_img_dir: str = None):
    # Setup Logging to file
    log_file = "generation_process_mclp.log"
    sys.stdout = DualLogger(log_file, sys.stdout)
    sys.stderr = DualLogger(log_file, sys.stderr)
    print(f"Logging initialized. Output redirected to {log_file}")
        
    # Load Trajectories
    with open(input_file, 'r') as f:
        data = json.load(f)
        
    # Initialize Agents (Mock or Real)
    # In real usage, these would be initialized with model paths
    # vlm_agent = VLMAgent(model_path="...") 
    # llm_agent = VLMAgent(model_path="...")
    
    try:
        api_base_url = "https://www.dmxapi.cn/v1"
        api_key = "sk-D0lymL9RZaorYDWERN5Ob8dsqVpdZHpQMJIiz8sRd5n7ZofZ"
        # api_base_url = "https://api.siliconflow.cn/v1"
        # api_key = "sk-mlqqrnvqurprnmxxpatvhllckaogtckcajwxehrngcysjgmo"

        print(f"Initializing Real VLMAgent connected to {api_base_url}...")
        # Use the configuration as specified in Qwen3_single_worker_tsp_cot.py
        vlm_agent = VLMAgent(
            api_key=api_key,
            api_base_url=api_base_url,
            model_name="glm-4.6V"
        )
        # response = vlm_agent.generate(system_prompt="Hello, I am a student.", text="Hello, I am a student.")
        # print(response)
        # print("✓ Real VLMAgent initialized.")

        # Initialize Mock Agent for Logic Injection Module
        llm_agent = LLMAgent(
            api_key=api_key,
            api_base_url=api_base_url,
            model_name="glm-4.7"
        )
        # response = llm_agent.generate(system_prompt="Hello, I am a student.", text="Hello, I am a student.")
        # print(response)
        print("✓ Real LLMAgent initialized.")
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
    print(f"Loaded {len(data)} trajectories from {input_file}")
    for traj_idx, traj_data in enumerate(data):
        print(f"Processing trajectory {traj_idx}/{len(data)}")
        # Process first trajectory for testing/verification
        # traj_idx = 0
        # traj_data = data[traj_idx]
        
        # 1. Prepare Node Coords (Mock if missing)
        node_coords = traj_data.get("node_coords", {})
        
        # 2. Prepare Candidates (Mock if missing)
        # Check if candidates exist and match length
        traj_len = len(traj_data.get("trajectory", []))
        candidates_matrix = traj_data.get("candidates", [])
        
        # 3. Reshape into List[Dict] (The user's requested structure: traj_data[step]["trajectory"])
        trajectory_steps = []
        obs_list = traj_data.get("obs_list", [""] * traj_len)
        image_list = traj_data.get("image_list", [None] * traj_len)
        
        for i in range(len(traj_data["trajectory"])):
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

        if traj_idx % 2 == 0 and traj_idx > 0:
            print(f"Processed {traj_idx}/{len(data)} trajectories")
            with open(output_file, 'w') as f:
                json.dump(all_processed_data, f, indent=2, cls=EnhancedJSONEncoder)
            print(f"Saved processed data to {output_file}")

        
    # Save Output
    with open(output_file, 'w') as f:
        json.dump(all_processed_data, f, indent=2, cls=EnhancedJSONEncoder)
    print(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    # Default paths
    input_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/mclp_agent_output.json"
    output_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/mclp_cot_dataset.json"
    
    # Run
    main(input_path, output_path)
