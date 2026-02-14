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
class TSPFactors:
    """TSP 微观因子：用于支撑 '剥洋葱' 和 '不回头' 策略"""
    is_on_hull: bool                 # 是否在局部点阵的凸包边缘 (Perimeter-Peeling)
    outlier_score: float             # 距离局部重心的偏离度 (>1.0 为显著离群)
    isolation_score: float           # 局部隔离度 (距离最近邻居的距离 / 特征半径)
    angular_sweep_score: float       # 角度顺滑度 (180=直线, 0=急转弯)
    is_bridge_candidate: bool        # (启发式) 是否适合作为跳出当前簇的桥梁

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
    tsp_factors: TSPFactors

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
        "TSP": {
            "role_def": "You are a spatial reasoning expert solving the Traveling Salesperson Problem (TSP).",
            "objective": "Minimize the total closed-loop tour length.",
            "visual_focus": "Focus on convex hulls, cluster perimeters, and nearest neighbors.",
            "strategy_keywords": ["Perimeter-Peeling", "Convex Hull", "Nearest Neighbor", "Backtracking Avoidance"]
        },
        "CVRP": {
            "role_def": "You are a logistics expert solving the Capacitated Vehicle Routing Problem (CVRP).",
            "objective": "Optimize routes for a fleet with limited capacity, starting and ending at the Depot.",
            "visual_focus": "Focus on the Depot location, angular sectors (sweep), and distance from the depot.",
            "strategy_keywords": ["Angular Sweep", "Depot Return", "Capacity Preservation", "Petal Routing"]
        },
        "MCLP": {
            "role_def": "You are an urban planner solving the Maximum Covering Location Problem (MCLP).",
            "objective": "Maximize the total demand covered by placing limited facilities within a fixed radius.",
            "visual_focus": "Focus on uncovered gaps, density centroids, and overlapping circles.",
            "strategy_keywords": ["Gap-Filling", "Density Centroid", "Marginal Gain", "Overlap Minimization"]
        },
        "FLP": {
            "role_def": "You are a supply chain analyst solving the Facility Location Problem (FLP).",
            "objective": "Select facility locations to minimize the sum of distances from all customers to their nearest facility.",
            "visual_focus": "Focus on the geometric median and minimizing total dispersion.",
            "strategy_keywords": ["K-Means Center", "Demand Weighted Distance", "Voronoi Partition"]
        }
    }

    @classmethod
    def get_system_prompt(cls, task_type: str) -> str:
        profile = cls.PROFILES.get(task_type, cls.PROFILES["TSP"])
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

    def analyze_tsp_micro(self, 
                          current_idx: int, 
                          candidate_ids: List[int], 
                          prev_node_idx: Optional[int] = None) -> Dict[int, TSPFactors]:
        """
        生成 TSP 微观因子 (针对所有候选点)
        Returns: Dict[candidate_id, TSPFactors]
        """
        curr_pos = self._get_coords(current_idx)
        cand_points = np.array([self._get_coords(cid) for cid in candidate_ids])
        
        if len(candidate_ids) == 0:
            return {}

        # 1. 凸包计算 (Convex Hull)
        hull_mask = {cid: False for cid in candidate_ids}
        if len(candidate_ids) >= 3:
            try:
                hull = ConvexHull(cand_points)
                # hull.vertices 是 cand_points 的 index
                for v_idx in hull.vertices:
                    hull_mask[candidate_ids[v_idx]] = True
            except:
                # 共线或降维，所有点视为边缘
                for cid in candidate_ids: hull_mask[cid] = True
        else:
             for cid in candidate_ids: hull_mask[cid] = True

        # 2. 局部重心 (Local Centroid)
        local_centroid = np.mean(cand_points, axis=0)
        # 特征半径 (用于归一化)
        dists_to_center = np.linalg.norm(cand_points - local_centroid, axis=1)
        avg_radius = np.mean(dists_to_center) + 1e-6

        # 3. 逐点计算
        results = {}
        for i, cid in enumerate(candidate_ids):
            c_pos = cand_points[i]
            
            # A. Outlier Score
            d_center = dists_to_center[i]
            outlier_score = d_center / avg_radius
            
            # B. Isolation (Nearest neighbor excluding current pos)
            # 简单起见，计算到其他候选点的最小距离
            other_points = cand_points[np.arange(len(cand_points)) != i]
            if len(other_points) > 0:
                nn_dist = np.min(np.linalg.norm(other_points - c_pos, axis=1))
            else:
                nn_dist = avg_radius # Max isolation if alone
            isolation = nn_dist / avg_radius

            # C. Angular Sweep
            sweep_score = 180.0 # Default straight
            if prev_node_idx is not None:
                prev_pos = self._get_coords(prev_node_idx)
                angle_in = self.calculate_bearing(prev_pos, curr_pos)
                angle_out = self.calculate_bearing(curr_pos, c_pos)
                # Deviation from straight line (180 deg difference)
                # Ideal: angle_out = angle_in. No. Ideal depends on strategy (spiral vs zigzag)
                # Let's use smoothness: How much direction changed. 
                # 0 change = straight. 180 change = U-turn.
                diff = self.get_angular_diff(angle_in, angle_out)
                # Score: 180 (Straight) -> 0 (U-Turn)
                sweep_score = 180 - abs(diff) # Waiting: If in=90, out=90, diff=0. This is straight.
                # Actually, bearing is absolute. If I move East (0), then keep moving East (0), diff is 0.
                # Straight line means diff is 0. U-turn means diff is 180.
                # Let's standardize: 1.0 = Straight, 0.0 = U-turn
                sweep_score = 1.0 - (diff / 180.0) 

            results[cid] = TSPFactors(
                is_on_hull=hull_mask[cid],
                outlier_score=outlier_score,
                isolation_score=isolation,
                angular_sweep_score=sweep_score,
                # Heuristic: Bridge candidates are usually far from centroid OR specific directional logic
                is_bridge_candidate=(outlier_score > 1.2) 
            )
        
        return results

    # --- Master Analysis Interface ---

    def analyze_step(self, 
                     current_idx: int, 
                     candidate_ids: List[int], 
                     gt_node_id: int, 
                     prev_node_idx: Optional[int] = None,
                     visited_count: int = 0) -> Dict[str, Any]:
        """
        Main entry point for generating all geometric truths for a step.
        """
        curr_pos = self._get_coords(current_idx)
        gt_pos = self._get_coords(gt_node_id)
        
        # 1. Macro Analysis
        general_facts = self.analyze_general(current_idx, visited_count)
        
        # 2. Micro Analysis (TSP)
        # Filter out padding (-1)
        valid_cands = [c for c in candidate_ids if c != -1]

        tsp_facts_map = self.analyze_tsp_micro(current_idx, valid_cands, prev_node_idx)
        
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
                tsp_factors=tsp_facts_map[cid]
            ))

        # 4. GT Specifics
        gt_dist = np.linalg.norm(gt_pos - curr_pos)
        is_nearest = abs(gt_dist - min_dist) < 1e-6
        
        # Identify "Temptation Node" (Nearest neighbor that is NOT GT)
        temptation_id = None
        sorted_cands = sorted(candidates_meta, key=lambda x: x.dist)
        if len(sorted_cands) > 0 and sorted_cands[0].id != gt_node_id:
            temptation_id = sorted_cands[0].id
        elif len(sorted_cands) > 1 and sorted_cands[0].id == gt_node_id:
            temptation_id = sorted_cands[1].id # Next closest
        return {
            "general": general_facts,
            "candidates": candidates_meta,
            "gt_id": gt_node_id,
            "gt_stats": {
                "dist": gt_dist,
                "dir": self.get_compass_dir(curr_pos, gt_pos),
                "is_nearest": is_nearest,
                "tsp_factors": tsp_facts_map.get(gt_node_id)
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
        
        # 几何特征关键词 (用于验证 VLM 是否“看到”了这些属性)
        self.hull_keywords = ['hull', 'boundary', 'edge', 'perimeter', 'outer', 'limit', 'farthest']
        self.isolation_keywords = ['isolated', 'alone', 'outlier', 'far', 'distant', 'separate', 'remote']
        self.dense_keywords = ['cluster', 'group', 'dense', 'crowded', 'pack']

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
        tsp_factors = gt_stats['tsp_factors'] # dataclass
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

        # 2. 几何特征引导 (TSP Specifics)
        if tsp_factors.is_on_hull:
            prompt += f"2. **Topology**: It is located on the **outer boundary (convex hull)** of the local cluster.\n"
        else:
            prompt += f"2. **Topology**: It is an **internal** node inside the cluster.\n"

        if tsp_factors.outlier_score > 1.0 or tsp_factors.isolation_score > 0.8:
            prompt += f"3. **Density**: It appears **isolated/outlier** compared to the dense center.\n"
        
        # 3. 诱惑点对比 (最重要的逻辑！)
        if not gt_stats['is_nearest'] and temptation_id is not None:
            temp_str = get_fmt(temptation_id)
            prompt += (
                f"4. **Comparison**: Explicitly mention that **{temp_str}** is closer "
                f"but explain that {gt_str} is chosen for its strategic position (e.g., boundary/isolation).\n"
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
        tsp_factors = gt_stats['tsp_factors']
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

        # --- 3. Geometric Feature Check (New!) ---
        # A. Hull/Boundary Check
        if tsp_factors.is_on_hull:
            if not self._check_keywords(desc_lower, self.hull_keywords):
                return False, f"Missed topological feature: Target is on the 'Convex Hull/Boundary', but description didn't mention it."

        # B. Isolation Check
        if tsp_factors.outlier_score > 1.2: # 显著离群
            if not self._check_keywords(desc_lower, self.isolation_keywords):
                return False, f"Missed density feature: Target is an 'Outlier/Isolated', but description missed it."

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
        task_system_prompt = TaskContextManager.get_system_prompt("TSP")
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
    Dedicated Logic Injection for TSP (Multimodal Version).
    
    Orchestrates the 'Strategic Narrative' by fusing:
    1. Visual Topology (Hull, Cluster, Outlier) - from VLM
    2. Exact Metrics (Distance Cost) - from Text Dashboard
    3. Geometric Rules - from GeometryEngine
    """
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def _get_node_fmt(self, node_id: int, facts: Dict[str, Any]) -> str:
        """Helper: Option A [Node 3]"""
        cand = next((c for c in facts['candidates'] if c.id == node_id), None)
        lbl = cand.label if cand else "?"
        return f"Option {lbl} [Node {node_id}]"

    def _get_candidate_dist(self, node_id: int, facts: Dict[str, Any]) -> str:
        """Helper: Get formatted distance string"""
        cand = next((c for c in facts['candidates'] if c.id == node_id), None)
        return f"{cand.dist:.2f}" if cand else "Unknown"

    def _select_strategic_narrative(self, facts: Dict[str, Any]) -> Dict[str, str]:
        """
        [导演中心]：TSP 专用剧本分发器 (增强版)
        将具体的距离数值 (Hard Numbers) 写入剧本，强制 LLM 进行成本/收益对比。
        """
        gt_id = facts['gt_id']
        gt_stats = facts['gt_stats']
        tsp_factors = gt_stats['tsp_factors'] # TSPFactors dataclass
        general = facts['general']
        temptation_id = facts.get('temptation_id')
        
        gt_str = self._get_node_fmt(gt_id, facts)
        gt_dist = self._get_candidate_dist(gt_id, facts)
        
        # 默认剧本 (Greedy)
        narrative = {
            "strategy_name": "Greedy Efficiency",
            "reasoning_focus": f"Selecting {gt_str} minimizes immediate travel cost (Dist: {gt_dist}).",
            "conflict_handling": "None"
        }

        # =========================================================
        # 🎭 Playbook A: Perimeter Peeling (凸包剥离)
        # =========================================================
        # 条件：在凸包上 + 且不是最近的 (需要牺牲距离)
        if tsp_factors.is_on_hull and tsp_factors.outlier_score > 0.8:
            narrative = {
                "strategy_name": "Perimeter-Peeling",
                "reasoning_focus": (
                    f"{gt_str} is on the outer boundary (Hull). "
                    "Clearing it now maintains the topological perimeter."
                ),
                "conflict_handling": "Prioritize boundary geometry over minimal distance."
            }
            
            # 冲突处理：如果有更近的点 (Temptation)
            if temptation_id is not None and not gt_stats['is_nearest']:
                temp_str = self._get_node_fmt(temptation_id, facts)
                temp_dist = self._get_candidate_dist(temptation_id, facts)
                
                narrative["conflict_handling"] = (
                    f"REJECT GREEDY: Visually {temp_str} is closer. "
                    f"**Cite Dashboard**: {temp_str} costs only {temp_dist}, while target costs {gt_dist}. "
                    f"However, we accept this higher cost to prevent future backtracking."
                )

        # =========================================================
        # 🎭 Playbook B: Outlier Clearance (孤立点清理)
        # =========================================================
        elif tsp_factors.isolation_score > 1.2:
            narrative = {
                "strategy_name": "Outlier Clearance",
                "reasoning_focus": (
                    f"This node is spatially isolated. Leaving it for later would incur a massive detour penalty."
                ),
                "conflict_handling": (
                    f"Strategically visiting the outlier now (Dist: {gt_dist}) "
                    "is cheaper than returning to it later."
                )
            }

        # =========================================================
        # 🎭 Playbook C: End-Game Loop (回环)
        # =========================================================
        elif general.progress_ratio > 0.85:
            narrative = {
                "strategy_name": "Loop Closing",
                "reasoning_focus": (
                    f"Moving towards {general.depot_direction} to align with the start node "
                    "for the final tour closure."
                ),
                "conflict_handling": "Directional bias outweighs local proximity."
            }

        return narrative

    def inject_logic(self, 
                     verified_desc: str, 
                     text_dashboard: str,  # <--- [NEW] 接收文本观察
                     spatial_facts: Dict[str, Any], 
                     problem_context: str = "TSP") -> str:
        """
        Generates the 'Thought' component.
        Enforces: Visual Topology + Numeric Cost Verification.
        """
        gt_id = spatial_facts['gt_id']
        gt_str = self._get_node_fmt(gt_id, spatial_facts)
        
        # 1. 获取剧本
        narrative = self._select_strategic_narrative(spatial_facts)
        
        # 2. 构建 Prompt
        system_instruction = (
            "You are an expert TSP Solver (Traveling Salesperson Problem). "
            "Synthesize the **Visual Observation** and the **Dashboard Data** into a coherent thought. "
            "Your reasoning must balance Geometry (Visual) and Cost (Numbers)."
        )

        user_prompt = (
            f"**Task**: Generate the <Thought> rationale for choosing **{gt_str}**.\n\n"
            
            f"**Input 1: Visual Observation** (Topology)\n"
            f"\"{verified_desc}\"\n\n"
            
            f"**Input 2: Dashboard Data** (Metrics)\n"
            f"```text\n{text_dashboard}\n```\n\n"
            
            f"**Input 3: Strategic Directive** (Strategy)\n"
            f"- **Strategy**: {narrative['strategy_name']}\n"
            f"- **Core Logic**: {narrative['reasoning_focus']}\n"
            f"- **Conflict/Trade-off**: {narrative['conflict_handling']}\n\n"
            
            f"**Requirements for <Thought>**:\n"
            f"1. **The Anchor**: Start by citing the Visual Observation (e.g., 'Visually, {gt_str} lies on the Hull...').\n"
            f"2. **The Verification**: Explicitly **cite the Distance** from the Dashboard Data to quantify the cost "
            f"(e.g., 'Dashboard confirms a travel cost of 0.45' or 'It is the nearest option at 0.12').\n"
            f"3. **The Strategy**: Conclude with the Strategy **'{narrative['strategy_name']}'**.\n"
            f"4. **Rejection Logic**: If rejecting a closer node (Conflict), contrast the distances (e.g., 'Accepting 0.45 over 0.12 to preserve structure').\n"
            f"5. Max 200 words. Telegraphic style."
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
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def assemble_cot(self, verified_desc: str, logic_reasoning: str, trajectory_action: str) -> str:
        """
        Assembles the final CoT with strict Information Decoupling.
        Goal: 
        - Observation = Pure Visual Facts (Geometry, Topology, Location)
        - Thought = Pure Strategic Reasoning (Strategy Name, Future Consequence, Trade-off)
        """
        
        # 1. 准备 Decision 字符串
        # 确保 action 是字符串形式
        decision_str = f"\\boxed{{{trajectory_action}}}"
        
        # 2. 构造 Prompt：核心是定义 "Role" 和 "Boundary"
        system_instruction = (
            "You are an expert AI Data Synthesizer. "
            "Your task is to refine raw texts into a strict 'Observation-Thought' pair for training. "
            "Constraint: Maximize information density. Eliminate redundancy between Observation and Thought."
        )

        user_prompt = (
            f"### Raw Input Data\n"
            f"1. [Raw Observation]: {verified_desc}\n"
            f"2. [Raw Thought]: {logic_reasoning}\n\n"
            
            f"### Refinement Task\n"
            f"Rewrite the inputs into two distinct parts separated by '|||'.\n\n"
            
            f"**Part 1: <Observation>** (The 'What')\n"
            f"- Summarize ONLY the geometric facts: location, quadrant, hull status, density, isolation.\n"
            f"- Remove phrases like 'I can see', 'In the image'.\n"
            f"- Max 100 words.\n\n"
            
            f"**Part 2: <Thought>** (The 'Why')\n"
            f"- Explain the decision based on the Strategy (e.g., 'Perimeter-Peeling', 'Reject Greedy').\n"
            f"- **CRITICAL**: Do NOT repeat the facts from Observation. Assume the user just read them.\n"
            f"- Focus on the *consequence* (e.g., 'to avoid backtracking', 'to close the loop').\n"
            f"- Max 100 words.\n\n"
            
            f"### Output Format Example\n"
            f"Candidate A is on the NW convex hull boundary. ||| Using the Perimeter-Peeling strategy, we clear this outlier first to maintain topological order.\n\n"
            
            f"### Your Output:"
        )

        try:
            # 3. LLM Generation
            response = self.llm.generate(
                system_prompt=system_instruction,
                text=user_prompt,
                max_tokens=2048,
                temperature=0.5 # 稍微提高一点点以允许措辞调整，但保持低位
            )
            
            # print(f"Refinement Raw: {response}") # Debug
            
            # 4. 鲁棒解析 (Regex Parsing)
            # 处理可能的换行符或空格: "Obs... \n ||| \n Thought..."
            parts = re.split(r'\s*\|\|\|\s*', response.strip())
            
            if len(parts) >= 2:
                obs_clean = parts[0].strip()
                thought_clean = parts[1].strip()
            else:
                # Fallback: 如果分隔符丢失，尝试智能分割或回退
                print(f"Refinement Warning: Separator '|||' missing. Using raw inputs.")
                obs_clean = verified_desc.strip()
                thought_clean = logic_reasoning.strip()

            # 5. 最终组装 (XML Wrapping)
            # 这里可以加一个额外的清洗，去掉可能残留的 "Observation:" 前缀
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
            # Fallback: 简单的字符串清洗
            clean_obs = verified_desc.replace("I can see", "").strip()
            return (
                f"<Observation> {clean_obs} </Observation>\n"
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
        )

        prev_node = current_idx
        current_idx = action_id
        
        # 4. Prepare Image (Base64)
        image_b64 = step_data.get('image', None)
        
        # 5. Module 2: Perception Loop
        print(f"Step {step_idx}")
        try:
            verified_desc = perception_module.run_perception_loop(image_b64, spatial_facts)
        except Exception as e:
            print(f"Step {step_idx} failed perception loop: {e}")
            continue
            
        if verified_desc is None:
            print(f"Step {step_idx} failed perception loop.")
            continue
        
        # 6. Module 3: Logic Injection
        logic_reasoning = logic_module.inject_logic(verified_desc, obs_text, spatial_facts)
        
        # 7. Module 4: Refinement
        final_cot = refinement_module.assemble_cot(verified_desc, logic_reasoning, action_raw)
        
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
    log_file = "generation_process_tsp.log"
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
        if traj_idx < 39:
            print(f"skip the {traj_idx}!!!")
            continue
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
    
    with open(output_file, 'w') as f:
        json.dump(all_processed_data, f, indent=2, cls=EnhancedJSONEncoder)
    print(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    # Default paths
    input_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/tsp_agent_output.json"
    output_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/tsp_cot_dataset.json"
    
    # Run
    main(input_path, output_path)
