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
class FLPFactors:
    """FLP 微观因子：用于支撑 '最大覆盖' 和 '最小距离' 策略"""
    marginal_gain: float             # 边际收益：选该点后，总距离减少了多少
    local_density: int               # 局部密度：该点半径 R 内有多少个客户
    max_min_distance: float          # Gap Filling：该点离“最近的已建站点”有多远
    centroid_alignment: float        # 该点是否接近其周围邻居的几何重心

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
    flp_factors: Optional[FLPFactors] = None

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
        "FLP": {
            "role_def": "You are a supply chain analyst solving the Facility Location Problem (FLP).",
            "objective": "Select facility locations to minimize the sum of distances from all customers to their nearest facility.",
            "visual_focus": "Focus on the geometric median, service gaps, and high-density clusters.",
            "strategy_keywords": ["Max Coverage", "Gap-Filling", "Density Centroid", "Voronoi Partition"]
        }
    }

    @classmethod
    def get_system_prompt(cls, task_type: str) -> str:
        profile = cls.PROFILES.get(task_type, cls.PROFILES["FLP"])
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

    def analyze_flp_micro(self, 
                          current_open_facilities: List[int],
                          candidate_ids: List[int]) -> Dict[int, FLPFactors]:
        """
        Generate FLP micro factors for all candidates.
        Core Logic: Voronoi, Gain, Density
        """
        if not candidate_ids:
            return {}

        # 0. Pre-computation
        # Identify all customer nodes (assuming all nodes are customers)
        all_coords = self.coords
        
        # Calculate current min distances if there are open facilities
        current_min_dists = np.full(self.total_nodes, 1e9) # Use large number instead of inf
        if current_open_facilities:
            open_coords = self.coords[current_open_facilities]
            # dists: (N_customers, N_open)
            dists = np.linalg.norm(all_coords[:, np.newaxis, :] - open_coords[np.newaxis, :, :], axis=2)
            current_min_dists = np.min(dists, axis=1)
            current_total_cost = np.sum(current_min_dists)
        else:
            # First step: assume 'infinite' cost or just 0 baseline to maximize reduction?
            # To be consistent with "Reduction", we can assume a virtual facility very far away.
            # But simpler: For first step, Gain = -Total_Cost (minimize Cost = maximize Gain)
            # We will handle the "Gain" interpretation carefully.
            current_total_cost = 1e9 * self.total_nodes

        # Local Density Radius R (Heuristic: 3 * Global Avg NND)
        density_radius = self.global_avg_nnd * 3.0
        
        results = {}
        cand_coords_list = [self._get_coords(cid) for cid in candidate_ids]
        cand_coords = np.array(cand_coords_list)
        
        # Pre-calculate neighbors for centroid alignment (k=6, incl self)
        k = min(6, self.total_nodes)
        nbrs_engine = NearestNeighbors(n_neighbors=k).fit(self.coords)
        
        for i, cid in enumerate(candidate_ids):
            c_pos = cand_coords[i]
            
            # 1. Marginal Gain (Virtual Selection Simulation)
            dists_to_c = np.linalg.norm(all_coords - c_pos, axis=1)
            new_min_dists = np.minimum(current_min_dists, dists_to_c)
            new_total_cost = np.sum(new_min_dists)
            
            if not current_open_facilities:
                # Special case for first facility: Gain is just inverse of cost
                # Let's define Gain as "Cost Savings relative to Worst Case"
                # Or simply: Gain = -new_total_cost (so higher is better)
                # But to keep it positive and scaled, maybe just store new_total_cost and let logic handle it?
                # User asked for "Marginal Gain: 选该点后，总距离减少了多少？"
                # Technically undefined for first step.
                # Let's set it to a normalized score based on centrality?
                # For now: Gain = (Max Possible Cost - Actual Cost)
                marginal_gain = (self.span_x + self.span_y) * self.total_nodes - new_total_cost
            else:
                marginal_gain = current_total_cost - new_total_cost
            
            # 2. Max-Min Distance (Gap Filling)
            # Distance to nearest OPEN facility.
            # If no open facilities, this is undefined (or infinite).
            if current_open_facilities:
                dist_to_nearest_open = current_min_dists[cid]
            else:
                dist_to_nearest_open = self.span_x + self.span_y # Max
            
            # 3. Local Density
            count = np.sum(dists_to_c < density_radius)
            local_density = int(count)
            
            # 4. Centroid Alignment
            _, indices = nbrs_engine.kneighbors([c_pos])
            neighbor_indices = indices[0][1:] # Exclude self
            if len(neighbor_indices) > 0:
                neighbor_coords = self.coords[neighbor_indices]
                centroid = np.mean(neighbor_coords, axis=0)
                centroid_alignment = np.linalg.norm(c_pos - centroid)
            else:
                centroid_alignment = 0.0
            
            results[cid] = FLPFactors(
                marginal_gain=float(marginal_gain),
                local_density=local_density,
                max_min_distance=float(dist_to_nearest_open),
                centroid_alignment=float(centroid_alignment)
            )
            
        return results

    # --- Master Analysis Interface ---

    def analyze_step(self, 
                     current_idx: int, 
                     candidate_ids: List[int], 
                     gt_node_id: int, 
                     prev_node_idx: Optional[int] = None,
                     current_open_facilities: List[int] = None, # New Argument
                     visited_count: int = 0) -> Dict[str, Any]:
        """
        Main entry point for generating all geometric truths for a step.
        """
        curr_pos = self._get_coords(current_idx)
        gt_pos = self._get_coords(gt_node_id)
        
        # 1. Macro Analysis
        general_facts = self.analyze_general(current_idx, visited_count)
        
        # 2. Micro Analysis (FLP)
        # Filter out padding (-1)
        valid_cands = [c for c in candidate_ids if c != -1]
        
        # Default to empty list if None
        if current_open_facilities is None:
            current_open_facilities = []

        # Switch to FLP Analysis
        flp_facts_map = self.analyze_flp_micro(current_open_facilities, valid_cands)
        
        # 3. Assemble Candidates Metadata
        candidates_meta = []
        min_dist = float('inf') # Still relevant for "temptation" (nearest neighbor)? 
        # In FLP, "nearest" might mean "nearest to open facilities" or "highest gain"?
        # User mentioned "min_dist_to_open".
        # But here 'min_dist' was used to find the geographically closest candidate to *current_idx*.
        # In FLP, current_idx might be the last added facility.
        # If we want to keep "Temptation" logic, maybe "Greedy Temptation" is the one with highest Gain?
        # Or geographically closest to last one?
        # Usually FLP greedy is "Max Gain".
        
        # Let's keep calculating dist to current_idx for visualization/context
        
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
                flp_factors=flp_facts_map[cid] # Enable FLP
            ))

        # 4. GT Specifics
        gt_dist = np.linalg.norm(gt_pos - curr_pos)
        is_nearest = abs(gt_dist - min_dist) < 1e-6
        
        # Temptation Logic for FLP
        # Maybe the one with highest Marginal Gain?
        # Or if we want to simulate "Greedy" vs "Strategic", FLP Greedy IS Max Gain.
        # So maybe "Temptation" is the Greedy choice (Max Gain), and if GT is not Max Gain, we explain why.
        # Let's find candidate with max marginal_gain
        sorted_by_gain = sorted(candidates_meta, key=lambda x: x.flp_factors.marginal_gain, reverse=True)
        
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
                "flp_factors": flp_facts_map.get(gt_node_id)
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
        
        # 几何特征关键词 (FLP Specific)
        # Centroid (重心), Cluster (簇), Dense Core (高密核心), Gap (空隙), Voronoi Region (辖区), Coverage (覆盖)
        self.centroid_keywords = ['centroid', 'center', 'middle', 'core', 'central']
        self.gap_keywords = ['gap', 'void', 'empty', 'sparse', 'orphan', 'isolated', 'far']
        self.coverage_keywords = ['cover', 'reach', 'service', 'serve', 'range']
        self.density_keywords = ['cluster', 'group', 'dense', 'crowded', 'pack', 'cloud']

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
        flp_factors = gt_stats['flp_factors'] # dataclass
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

        # 2. 几何特征引导 (FLP Specifics)
        # Centrality: "Is this node visually in the middle of a dense cloud of red dots?"
        if flp_factors.local_density > 3 or flp_factors.centroid_alignment < 1.0:
            prompt += f"2. **Centrality**: It is visually located in the **middle/core** of a **dense cluster** of points.\n"
        
        # Separation: "Is it far away from existing Blue Squares (Open Facilities)?"
        # Assuming max_min_distance represents distance to nearest open facility
        if flp_factors.max_min_distance > 5.0: # Threshold needs tuning based on scale
             prompt += f"3. **Separation**: It is positioned **far away** from any existing facilities (Blue Squares), filling a service **gap**.\n"

        # Potential: "Does it cover a region that is currently 'orphaned' (far from any facility)?"
        if flp_factors.marginal_gain > 0: # Assuming positive gain means coverage improvement
            prompt += f"4. **Potential**: It covers a region that is currently **unserved/orphaned**, providing significant **coverage** gain.\n"

        # 3. 诱惑点对比 (最重要的逻辑！)
        if not gt_stats['is_nearest'] and temptation_id is not None:
            temp_str = get_fmt(temptation_id)
            prompt += (
                f"5. **Comparison**: Explicitly mention that **{temp_str}** might seem attractive (e.g., closer or denser), "
                f"but explain that {gt_str} is chosen for its strategic value (e.g., covering a larger gap or serving more people).\n"
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
        flp_factors = gt_stats['flp_factors']
        temptation_id = facts.get('temptation_id')

        # --- 1. Identity Check ---
        # 使用正则确保匹配完整的 ID (避免匹配到 text 中的数字)
        if not re.search(rf"\b(candidate|node|option)\s*{re.escape(gt_id)}\b", desc_lower):
             return False, f"Failed to explicitly mention target 'Candidate {gt_id}'."

        # --- 2. Direction Check ---
        gt_dir = gt_stats['dir']
        expected_kws = self.direction_map.get(gt_dir, [])
        if not self._check_keywords(desc_lower, expected_kws):
            return False, f"Failed to identify location '{gt_dir}'. Expected keywords: {expected_kws}"

        # --- 3. Geometric Feature Check (FLP) ---
        # A. Centrality/Density Check
        if flp_factors.local_density > 3:
            if not self._check_keywords(desc_lower, self.density_keywords + self.centroid_keywords):
                return False, f"Missed density feature: Target is in a 'Dense Core/Cluster', but description didn't mention it."

        # B. Gap/Separation Check
        if flp_factors.max_min_distance > 5.0:
            if not self._check_keywords(desc_lower, self.gap_keywords):
                return False, f"Missed gap feature: Target fills a 'Service Gap/Void', but description missed it."

        # --- 4. Temptation/Comparison Check (Crucial for SFT) ---
        if temptation_id is not None and not gt_stats['is_nearest']:
            temp_id_str = str(temptation_id)
            if not re.search(rf"\b{re.escape(temp_id_str)}\b", desc_lower):
                return False, f"Failed to compare with the greedy temptation 'Candidate {temptation_id}'. Evaluation requires 'Reject Greedy' logic."

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
        [Director]: Selects the FLP strategy script based on geometric factors.
        """
        gt_id = facts['gt_id']
        gt_stats = facts['gt_stats']
        flp_factors = gt_stats['flp_factors'] # FLP Specific
        
        gt_str = self._get_node_fmt(gt_id, facts)
        
        # Default: 🎭 Script C: Marginal Optimization (Fine-tuning)
        # Scenario: Late game or no obvious clusters
        narrative = {
            "strategy_name": "Marginal Optimization",
            "reasoning_focus": (
                f"The major clusters are covered. {gt_str} offers the best remaining Marginal Gain. "
                "Strategy: Fine-tuning to reduce residual total cost."
            ),
            "conflict_handling": "Global efficiency outweighs local heuristics."
        }

        # --- 🎭 Script A: Greedy Capture (Greedy Coverage) ---
        # Scenario: High Density + High Gain (Early game or massive cluster)
        # Condition: High local density (>3) AND decent gain (positive)
        if flp_factors.local_density > 3 and flp_factors.marginal_gain > 0:
            narrative = {
                "strategy_name": "Greedy Capture",
                "reasoning_focus": (
                    f"Visually, {gt_str} sits at the Centroid of a large unserved cluster. "
                    f"Strategy: Major Cluster Capture to maximize immediate coverage."
                ),
                "conflict_handling": "Prioritize high-demand clusters over scattered points."
            }

        # --- 🎭 Script B: Gap Filling ---
        # Scenario: Service void, far from existing facilities
        # Condition: High Max-Min Distance (> 5.0) - overriding density if needed
        elif flp_factors.max_min_distance > 5.0:
             narrative = {
                "strategy_name": "Gap Filling",
                "reasoning_focus": (
                    f"Visually, this region is a service void (far from existing facilities). "
                    f"Strategy: Gap Filling to minimize the maximum service distance."
                ),
                "conflict_handling": "Equity and worst-case minimization outweigh pure density."
            }

        return narrative

    def inject_logic(self, 
                     verified_desc: str, 
                     text_dashboard: str, 
                     spatial_facts: Dict[str, Any], 
                     problem_context: str = "FLP") -> str:
        """
        Generates the 'Thought' component with strict Data Fusion requirements.
        """
        gt_id = spatial_facts['gt_id']
        gt_str = self._get_node_fmt(gt_id, spatial_facts)
        
        narrative = self._select_strategic_narrative(spatial_facts)
        
        system_instruction = (
            "You are an expert Facility Location Problem (FLP) solver. "
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
            f"2. **The Verification**: You MUST cite the **'Expected Total Distance Reduction'** (or similar Gain value) from the text data to justify why {gt_str} is better than others.\n"
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
        Assembles the final FLP Chain-of-Thought.
        
        Args:
            verified_desc: Output from FLPPerceptionModule (Visuals: Centroid, Gap...)
            logic_reasoning: Output from FLPLogicInjectionModule (Logic: Gain, Cost...)
            facts: The master data dict
        """
        
        # 1. 准备标准格式
        target_fmt, boxed_val = self._get_target_info(facts)
        decision_str = f"\\boxed{{{boxed_val}}}"
        
        # 2. 构造 Prompt：定义 "Urban Planner" 角色
        system_instruction = (
            "You are an expert Network Planner & Site Selection Analyst. "
            "Your task is to refine the reasoning for placing a facility. "
            "Tone: Strategic, Analytical, Maximizing Utility."
        )

        user_prompt = (
            f"### Raw Input Data\n"
            f"1. [Raw Observation]: {verified_desc}\n"
            f"2. [Raw Thought]: {logic_reasoning}\n"
            f"3. [Target Identity]: {target_fmt}\n\n"
            
            f"### Refinement Task\n"
            f"Rewrite the inputs into two distinct parts separated by '|||'.\n\n"
            
            f"**Part 1: <Observation>** (Geometric & Spatial Context)\n"
            f"- Focus on: **Cluster Centroids**, **High-Density Zones**, or **Service Voids (Gaps)**.\n"
            f"- **MANDATORY**: Use the exact format **'{target_fmt}'**.\n"
            f"- Max 40 words. No 'I can see'.\n\n"
            
            f"**Part 2: <Thought>** (Strategic Value)\n"
            f"- Explain the selection strategy: '**Greedy Capture**' (for centroids) or '**Gap Filling**' (for voids).\n"
            f"- **CRITICAL**: You MUST preserve any specific numbers (e.g., 'Gain 50.2') mentioned in the Raw Thought.\n"
            f"- Focus on the consequence: 'maximizing coverage', 'minimizing total distance', 'serving the outlier'.\n"
            f"- Max 50 words.\n\n"
            
            f"### Output Example\n"
            f"Option C [Node 45] sits at the geometric centroid of the dense southern cluster. ||| Executing Greedy Capture, we select this site as Dashboard data confirms it maximizes Marginal Gain (102.4), anchoring the network in the highest demand zone.\n\n"
            
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
    current_open_facilities = []

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
            prev_node_idx=prev_node,
            current_open_facilities=list(current_open_facilities)
        )

        prev_node = current_idx
        current_idx = action_id
        current_open_facilities.append(action_id)
        
        # 4. Prepare Image (Base64)
        image_b64 = step_data.get('image', None)
        
        # 5. Module 2: Perception Loop
        verified_desc = perception_module.run_perception_loop(image_b64, spatial_facts)
        if verified_desc is None:
            print(f"Step {step_idx} failed perception loop.")
            continue
        
        # 6. Module 3: Logic Injection
        logic_reasoning = logic_module.inject_logic(verified_desc, obs_text, spatial_facts)
        
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
    log_file = "generation_process_flp.log"
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
            # save 一下防止丢失
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
    input_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/flp_agent_output.json"
    output_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/flp_cot_dataset.json"
    
    # Run
    main(input_path, output_path)
