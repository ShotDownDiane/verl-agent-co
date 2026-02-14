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

import math
import numpy as np
from sklearn.neighbors import NearestNeighbors
from dataclasses import dataclass
from typing import List, Dict, Tuple, Any, Optional

# ==========================================
# 1. CVRP 专用几何因子
# ==========================================

@dataclass
class GeneralFactors:
    """宏观态势因子 (通用)"""
    norm_pos: Tuple[float, float]
    quadrant: str
    depot_direction: str
    depot_dist_ratio: float
    progress_ratio: float
    distribution_type: str
    is_in_dense_region: bool

@dataclass
class CVRPFactors:
    """CVRP 微观因子：聚焦 '扇区扫描' 和 '容量管理'"""
    # --- 角度/扇区 (The Sweep) ---
    polar_angle: float            # 相对于 Depot 的极角 (0-360)
    sweep_order_rank: int         # 在当前候选集中，按逆时针扫描的顺位
    angle_from_current: float     # 与当前节点相对于 Depot 的夹角差 (衡量是否“顺路”)
    
    # --- 距离/位置 (The Radial Cost) ---
    dist_to_depot: float          # 离车场的距离
    is_furthest_in_sector: bool   # 是否是该扇区最远的点 (Seed Point，通常优先访问)
    
    # --- 容量/需求 (The Capacity) ---
    demand: float                 # 该点的需求量
    is_capacity_feasible: bool    # 剩余载重是否足够 (Current Load + Demand <= Cap)
    capacity_fill_ratio: float    # 访问后车辆载重率 (用于判断是否“正好填满”)

@dataclass
class CandidateMeta:
    id: int
    label: str
    coords: Tuple[float, float]
    dist: float
    direction: str
    cvrp_factors: CVRPFactors     # 替换为 CVRP 因子

# ==========================================
# 2. 核心引擎
# ==========================================

class GeometryEngine:
    def __init__(self, 
                 coords: np.ndarray, 
                 demands: np.ndarray, # [新增] 需求数组
                 depot_idx: int = 0, 
                 global_bounds: Tuple[float, float, float, float] = None):
        
        self.coords = np.array(list(coords.values()))
        self.demands = demands
        self.depot_idx = depot_idx
        self.total_nodes = len(coords)
        
        # 预计算全局边界 (保持不变)
        if global_bounds:
            self.min_x, self.min_y, self.max_x, self.max_y = global_bounds
        else:
            self.min_x, self.min_y = np.min(self.coords, axis=0)
            self.max_x, self.max_y = np.max(self.coords, axis=0)
        
        self.span_x = max(self.max_x - self.min_x, 1e-6)
        self.span_y = max(self.max_y - self.min_y, 1e-6)
        
        # 全局分布 (保持不变)
        self.global_distribution, self.global_avg_nnd = self._analyze_global_distribution()

        # 方位定义 (保持不变)
        self.compass_sectors = {
            'N': 90, 'NE': 45, 'E': 0, 'SE': 315,
            'S': 270, 'SW': 225, 'W': 180, 'NW': 135
        }

    # --- 基础工具方法 ---

    def _get_coords(self, idx: int) -> np.ndarray:
        return self.coords[idx]

    def _analyze_global_distribution(self) -> Tuple[str, float]:
        """判定全局分布类型 (Cluster/Uniform)"""
        # ... (逻辑同前，保持不变) ...
        # 为节省篇幅省略，可以直接复用之前的代码
        if self.total_nodes < 5: return "Uniform", 0.5
        nbrs = NearestNeighbors(n_neighbors=2).fit(self.coords)
        dists, _ = nbrs.kneighbors(self.coords)
        mean_nnd = np.mean(dists[:, 1])
        area = self.span_x * self.span_y
        expected = 0.5 / np.sqrt(self.total_nodes / area)
        R = mean_nnd / expected
        if R < 0.7: return "Clustered", mean_nnd
        elif R > 1.2: return "Uniform", mean_nnd
        return "Mixed", mean_nnd

    def _get_quadrant_desc(self, pos: np.ndarray) -> str:
        # ... (逻辑同前) ...
        nx = (pos[0] - self.min_x) / self.span_x
        ny = (pos[1] - self.min_y) / self.span_y
        ns = "South" if ny < 0.5 else "North"
        we = "West" if nx < 0.5 else "East"
        return f"{ns}-{we}"

    def calculate_polar_angle(self, center: np.ndarray, target: np.ndarray) -> float:
        """
        [CVRP 核心] 计算相对于 Depot 的极角 (0-360)
        0度 = 正东, 90度 = 正北 (逆时针方向)
        """
        dy = target[1] - center[1]
        dx = target[0] - center[0]
        angle = math.degrees(math.atan2(dy, dx))
        return (angle + 360) % 360

    def get_compass_dir(self, start: np.ndarray, end: np.ndarray) -> str:
        # ... (逻辑同前，用于生成 'NW', 'SE' 等可读方位) ...
        # 这里可以使用 calculate_polar_angle 的结果来查表
        angle = self.calculate_polar_angle(start, end)
        best_dir = 'Unknown'
        min_diff = 360.0
        for direction, center_angle in self.compass_sectors.items():
            diff = abs(angle - center_angle)
            diff = min(diff, 360 - diff)
            if diff < min_diff:
                min_diff = diff
                best_dir = direction
        return best_dir

    # --- CVRP Analysis Pipelines ---

    def analyze_general(self, current_idx: int, visited_count: int) -> GeneralFactors:
        """宏观因子生成 (保持不变)"""
        curr_pos = self._get_coords(current_idx)
        depot_pos = self._get_coords(self.depot_idx)
        
        nx = (curr_pos[0] - self.min_x) / self.span_x
        ny = (curr_pos[1] - self.min_y) / self.span_y
        
        return GeneralFactors(
            norm_pos=(nx, ny),
            quadrant=self._get_quadrant_desc(curr_pos),
            depot_direction=self.get_compass_dir(curr_pos, depot_pos),
            depot_dist_ratio=np.linalg.norm(curr_pos - depot_pos) / np.linalg.norm([self.span_x, self.span_y]),
            progress_ratio=visited_count / self.total_nodes,
            distribution_type=self.global_distribution,
            is_in_dense_region=False # 简化处理
        )

    def analyze_cvrp_micro(self, 
                           current_idx: int, 
                           candidate_ids: List[int], 
                           current_load: float, 
                           vehicle_capacity: float) -> Dict[int, CVRPFactors]:
        """
        [新增] CVRP 微观因子计算
        关注：极角扇区、载重限制、回场代价
        """
        depot_pos = self._get_coords(self.depot_idx)
        # 获取当前位置相对于 Depot 的极角（作为扫描基准）
        # 如果当前在 Depot，基准角度设为 0 (或者上一条路径的角度，这里简化为 0)
        curr_pos = self._get_coords(current_idx)
        
        if current_idx == self.depot_idx:
            current_polar = 0.0 # 开始新路径，默认从东边开始扫
        else:
            current_polar = self.calculate_polar_angle(depot_pos, curr_pos)

        results = {}
        
        # 1. 预计算所有候选点的极角，用于排序
        cand_angles = []
        for cid in candidate_ids:
            c_pos = self._get_coords(cid)
            angle = self.calculate_polar_angle(depot_pos, c_pos)
            cand_angles.append((cid, angle))
        
        # 按角度排序，模拟扫描线
        cand_angles.sort(key=lambda x: x[1])
        sorted_ids = [x[0] for x in cand_angles]

        # 2. 逐个计算因子
        for cid in candidate_ids:
            c_pos = self._get_coords(cid)
            angle = self.calculate_polar_angle(depot_pos, c_pos)
            
            # A. 扇区/扫描逻辑
            # 计算顺时针/逆时针的夹角差
            diff = angle - current_polar
            # 归一化到 [-180, 180] 或 [0, 360] 看业务逻辑，这里我们看“逆时针扫描”
            # 正值表示在当前角度的“前方”
            sweep_diff = (angle - current_polar + 360) % 360 
            
            # B. 距离/位置
            dist_to_depot = np.linalg.norm(c_pos - depot_pos)
            
            # C. 容量逻辑
            demand = self.demands[cid]
            rem_cap = vehicle_capacity - current_load
            is_feasible = (demand <= rem_cap + 1e-6)
            
            # 填满率：如果这个点正好能填满车，那是极好的（Fill Ratio 接近 1.0）
            # 如果不可行，Fill Ratio > 1.0
            fill_ratio = (current_load + demand) / (vehicle_capacity + 1e-6)

            results[cid] = CVRPFactors(
                polar_angle=angle,
                sweep_order_rank=sorted_ids.index(cid), # 简单的全局角度排序
                angle_from_current=sweep_diff,          # 越小说明越顺路 (Sweep Strategy)
                dist_to_depot=dist_to_depot,
                is_furthest_in_sector=False,            # 简化，暂不计算
                demand=float(demand),
                is_capacity_feasible=is_feasible,
                capacity_fill_ratio=fill_ratio
            )
            
        return results

    # --- Master Interface ---

    def analyze_step(self, 
                     current_idx: int, 
                     candidate_ids: List[int], 
                     gt_node_id: int, 
                     current_load: float,        # [CVRP新增]
                     vehicle_capacity: float,    # [CVRP新增]
                     visited_count: int = 0) -> Dict[str, Any]:
        
        curr_pos = self._get_coords(current_idx)
        gt_pos = self._get_coords(gt_node_id)
        
        # 1. Macro Analysis
        general_facts = self.analyze_general(current_idx, visited_count)
        
        # 2. Micro Analysis (CVRP)
        valid_cands = [c for c in candidate_ids if c != -1]
        cvrp_facts_map = self.analyze_cvrp_micro(
            current_idx, valid_cands, current_load, vehicle_capacity
        )
        
        # 3. Assemble Metadata
        candidates_meta = []
        min_dist = float('inf')
        
        # 用于 Label 映射 (Optional)
        idx2option = {i: chr(65+i) for i in range(len(valid_cands))} # A, B, C...

        for i, cid in enumerate(valid_cands):
            c_pos = self._get_coords(cid)
            dist = np.linalg.norm(c_pos - curr_pos)
            if dist < min_dist: min_dist = dist
            
            label = idx2option.get(i, f"?")

            candidates_meta.append(CandidateMeta(
                id=cid,
                label=label,
                coords=tuple(c_pos),
                dist=float(dist),
                direction=self.get_compass_dir(curr_pos, c_pos), # 视觉方位
                cvrp_factors=cvrp_facts_map[cid] # 核心因子
            ))

        # 4. GT Specifics
        gt_dist = np.linalg.norm(gt_pos - curr_pos)
        is_nearest = abs(gt_dist - min_dist) < 1e-6
        
        # Identify "Temptation" (e.g. closer but overload)
        temptation_id = None
        # ... (Temptation logic can be refined for CVRP: e.g. closer but infeasible)

        return {
            "general": general_facts,
            "candidates": candidates_meta,
            "gt_id": gt_node_id,
            "gt_stats": {
                "dist": gt_dist,
                "dir": self.get_compass_dir(curr_pos, gt_pos),
                "is_nearest": is_nearest,
                "cvrp_factors": cvrp_facts_map.get(gt_node_id)
            },
            "context": { # 传给 LogicInjection 用
                "load": current_load,
                "capacity": vehicle_capacity,
                "remaining": vehicle_capacity - current_load
            }
        }
# ==========================================
# Module 2: Perception Loop
# ==========================================
from typing import Dict, Any, Tuple, Optional, List
import re

import re
from typing import Dict, Any, Tuple, Optional, List

class PerceptionModule:
    """
    Dedicated Perception Module for Capacitated Vehicle Routing Problem (CVRP).
    Focuses on: Angular Sectors, Depot Proximity, and Capacity Constraints.
    """
    def __init__(self, vlm_agent):
        self.vlm = vlm_agent
        
        # --- 1. 方位词表 (基础导航) ---
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
        
        # --- 2. CVRP 专用术语表 (用于验证) ---
        self.keywords = {
            # 扇区/扫描逻辑
            "sweep": [
                'sector', 'wedge', 'slice', 'angle', 'sweep', 'rotation', 
                'counter-clockwise', 'clockwise', 'align', 'path flow', 'sequence'
            ],
            # 仓库/回程逻辑
            "depot": [
                'depot', 'station', 'hub', 'center', 'origin', 'start point', 
                'return', 'refill', 'replenish'
            ],
            # 容量/拒绝逻辑
            "capacity": [
                'capacity', 'load', 'demand', 'full', 'limit', 'overload', 
                'constraint', 'heavy', 'skip', 'reject'
            ],
            # 距离/贪心逻辑
            "proximity": [
                'close', 'near', 'adjacent', 'proximity', 'immediate', 'next'
            ]
        }

    def _get_node_fmt(self, node_id: int, facts: Dict[str, Any]) -> str:
        """Standard Format: Option A [Node 3]"""
        cand = next((c for c in facts['candidates'] if c.id == node_id), None)
        lbl = cand.label if cand else "?"
        # 特殊处理 Depot
        if node_id == 0:
            return f"The Depot (Option {lbl} [Node 0])"
        return f"Option {lbl} [Node {node_id}]"

    def _check_keywords(self, text: str, keyword_list: List[str]) -> bool:
        text_lower = text.lower()
        return any(k in text_lower for k in keyword_list)

    def construct_grounding_prompt(self, facts: Dict[str, Any]) -> str:
        """
        构建 CVRP 专用的视觉引导 Prompt。
        """
        gt_id = facts['gt_id']
        gt_stats = facts['gt_stats']
        cvrp = gt_stats['cvrp_factors']  # CVRPFactors dataclass
        general = facts['general']
        temptation_id = facts.get('temptation_id')
        gt_str = self._get_node_fmt(gt_id, facts)

        # 1. 场景设定 (Logistics Context)
        prompt = (
            f"You are a Logistics Vision Engine. The image shows a CVRP (Vehicle Routing) map.\n"
            f"**Context**: The Depot is at the center (Node 0). We are currently in the {general.quadrant} quadrant.\n"
            f"Your goal is to visually confirm why **{gt_str}** is the optimal next stop.\n\n"
            f"**Ground Truths to Describe**:\n"
        )

        # 2. 基础定位
        prompt += f"1. **Location**: {gt_str} is visually located to the **{gt_stats['dir']}** of the current agent.\n"

        # 3. CVRP 特征引导 (分情况讨论)
        
        # Case A: 回到车场 (Depot Return)
        if gt_id == 0:
            prompt += (
                f"2. **Role**: Identify this node as the **Depot**. "
                f"State that the vehicle is returning here (likely to refill capacity).\n"
            )
        
        # Case B: 正常的扇区扫描 (Angular Sweep)
        # 如果是普通客户节点，且角度顺滑
        elif abs(cvrp.angle_from_current) < 45: 
            prompt += (
                f"2. **Sector Alignment**: Describe how this node aligns with the current **angular sweep** (counter-clockwise motion). "
                f"It naturally follows the current path vector without erratic zig-zagging.\n"
            )
        
        # Case C: 跨扇区/远端 (Radical Jump) - 较少见，通常是因为要开启新扇区
        else:
            prompt += f"2. **Position**: It represents a strategic move to a new angular sector.\n"

        # 4. 对比与拒绝 (The Temptation Logic)
        if not gt_stats['is_nearest'] and temptation_id is not None:
            temp_str = self._get_node_fmt(temptation_id, facts)
            prompt += (
                f"3. **Comparison (Critical)**: Visually contrast it with the closer **{temp_str}**.\n"
                f"   - Explicitly mention that {temp_str} is **closer**.\n"
                f"   - Explain that we SKIP {temp_str} because of **capacity constraints** or **sector misalignment** "
                f"(i.e., {temp_str} might belong to a different cluster/wedge).\n"
            )
        
        prompt += "\n**Output Requirement**: A concise, professional observation using logistics terms (Sector, Depot, Proximity)."
        return prompt

    def verify_description(self, description: str, facts: Dict[str, Any]) -> Tuple[bool, str]:
        """
        CVRP 专用验证器
        """
        desc_lower = description.lower()
        gt_id = str(facts['gt_id'])
        gt_stats = facts['gt_stats']
        cvrp = gt_stats['cvrp_factors']
        temptation_id = facts.get('temptation_id')

        # --- 1. Identity Check ---
        # 必须提到 ID，如 "Option A" 或 "Node 3"
        # Depot 特殊检查
        if gt_id == '0':
            if 'depot' not in desc_lower and 'node 0' not in desc_lower:
                return False, "Failed to identify the target as 'Depot' or 'Node 0'."
        else:
            # 使用宽泛匹配，允许 Option X 或 Node Y
            if not re.search(rf"\b(candidate|node|option)\s*(\[)?{re.escape(gt_id)}(\])?\b", desc_lower):
                 return False, f"Failed to explicitly mention target 'Node {gt_id}'."

        # --- 2. Direction Check ---
        gt_dir = gt_stats['dir']
        expected_kws = self.direction_map.get(gt_dir, [])
        if not self._check_keywords(desc_lower, expected_kws):
            return False, f"Failed to identify visual location '{gt_dir}'. Expected keywords: {expected_kws}"

        # --- 3. CVRP Feature Check ---
        
        # A. Depot Check
        if gt_id == '0':
            if not self._check_keywords(desc_lower, self.keywords['depot']):
                return False, "Missed 'Depot/Refill' context for Node 0."
        
        # B. Sweep/Sector Check (对于非 Depot 点)
        # 如果是顺滑移动，必须提到 扇区/扫描/对齐
        elif abs(cvrp.angle_from_current) < 45:
            if not self._check_keywords(desc_lower, self.keywords['sweep']):
                return False, "Missed strategic feature: 'Sector/Sweep/Alignment' in description."

        # C. Temptation/Capacity Check
        # 如果拒绝了最近点，必须提到“为什么”
        if temptation_id is not None and not gt_stats['is_nearest']:
            temp_cand = next((c for c in facts['candidates'] if c.id == temptation_id), None)
            if temp_cand:
                temp_lbl = temp_cand.label
                # 检查是否提到了诱惑点
                hit_temp = (str(temptation_id) in desc_lower) or (temp_lbl.lower() in desc_lower)
                if not hit_temp:
                    return False, f"Failed to mention the closer alternative 'Option {temp_lbl} [Node {temptation_id}]'."
                
                # 检查是否给出了拒绝理由 (Capacity or Sector)
                # 只要出现了 capacity 或 sweep 类词汇即可
                has_reason = (self._check_keywords(desc_lower, self.keywords['capacity']) or 
                              self._check_keywords(desc_lower, self.keywords['sweep']))
                if not has_reason:
                    return False, "Mentioned the closer node but failed to explain rejection (Capacity/Sector reason)."

        return True, "Verified"

    def construct_reflexion_prompt(self, gt_id: int, previous_response: str, missing_reason: str, facts: Dict[str, Any]) -> str:
        """
        Reflexion Loop
        """
        gt_str = self._get_node_fmt(gt_id, facts)
        return (
            f"### Critique of Previous Output\n"
            f"**Draft**: \"{previous_response}\"\n"
            f"**Issue**: {missing_reason}\n\n"
            f"### Action\n"
            f"Rewrite the observation for **{gt_str}**.\n"
            f"- Correctly describe its location.\n"
            f"- **Fix the Issue**: Incorporate the missing logic (e.g., mention 'Sector' or 'Depot' or 'Capacity').\n"
            f"- Maintain the ID format: {gt_str}."
        )

    def run_perception_loop(self, image_b64: str, spatial_facts: Dict[str, Any], max_retries: int = 3) -> Optional[str]:
        gt_id = spatial_facts['gt_id']
        
        system_instruction = (
            "You are an expert Logistics AI solving a CVRP problem. "
            "Your task is to analyze the map and describe the next optimal move using "
            "precise geometric and logistics terminology (e.g., Angular Sector, Depot, Capacity)."
        )

        current_prompt = self.construct_grounding_prompt(spatial_facts)

        for attempt in range(max_retries):
            print(f"--- CVRP Perception Attempt {attempt + 1} ---")
            
            response = self.vlm.generate(
                system_prompt=system_instruction,
                text=current_prompt, 
                image=image_b64,
                max_tokens=2048
            )
            print("Perception Prompt:")
            print(current_prompt)
            print("************")
            print(f"Raw Response: {response}")
            
            is_valid, reason = self.verify_description(response, spatial_facts)
            
            if is_valid:
                return response
            
            # Reflexion
            current_prompt = self.construct_reflexion_prompt(
                gt_id=gt_id,
                previous_response=response,
                missing_reason=reason,
                facts=spatial_facts
            )
            
        return None

# ==========================================
# Module 3: Logic Injection
# ==========================================

from typing import Dict, Any, Optional


class LogicInjectionModule:
    """
    Dedicated Logic Injection for CVRP (Multimodal Version).
    
    Orchestrates the 'Strategic Narrative' by fusing:
    1. Visual Perception (from VLM)
    2. Hard Data / Dashboard (from Text Observation)
    3. Geometric Rules (from GeometryEngine)
    
    The goal is to generate CoTs that sound like a pilot checking both 
    the window (Visual) and the instruments (Dashboard).
    """
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def _get_node_fmt(self, node_id: int, facts: Dict[str, Any]) -> str:
        """Helper: Option A [Node 3] or The Depot [Node 0]"""
        if node_id == 0:
            return "The Depot [Node 0]"
        cand = next((c for c in facts['candidates'] if c.id == node_id), None)
        lbl = cand.label if cand else "?"
        return f"Option {lbl} [Node {node_id}]"

    def _get_candidate_data(self, node_id: int, facts: Dict[str, Any]) -> Any:
        """Helper: Fetch metadata (demand, dist) for a specific node"""
        return next((c for c in facts['candidates'] if c.id == node_id), None)

    def _select_strategic_narrative(self, facts: Dict[str, Any]) -> Dict[str, str]:
        """
        [导演中心]：CVRP 专用剧本分发器 (增强版)
        现在会将具体的数值 (Hard Numbers) 写入剧本，强制 LLM 引用。
        """
        gt_id = facts['gt_id']
        gt_stats = facts['gt_stats']
        cvrp = gt_stats['cvrp_factors']
        context = facts.get('context', {}) # {load, capacity, remaining}
        temptation_id = facts.get('temptation_id')
        
        gt_str = self._get_node_fmt(gt_id, facts)
        remaining_cap = context.get('remaining', 0.0)
        
        # 默认剧本
        narrative = {
            "strategy_name": "Greedy Logistics",
            "reasoning_focus": f"Selecting {gt_str} minimizes travel distance.",
            "conflict_handling": "None"
        }

        # =========================================================
        # 🎭 Playbook B: Depot Return (回库充能) - 引用剩余量
        # =========================================================
        if gt_id == 0:
            narrative = {
                "strategy_name": "Depot Refill",
                "reasoning_focus": (
                    f"Dashboard shows Capacity is depleted ({remaining_cap:.1f} remaining). "
                    "Visual logic dictates an immediate return to Depot."
                ),
                "conflict_handling": "Operational feasibility overrides customer proximity."
            }

        # =========================================================
        # 🎭 Playbook C: Capacity Rejection (容量规避) - 引用需求量
        # =========================================================
        elif temptation_id is not None and not gt_stats['is_nearest']:
            temp_str = self._get_node_fmt(temptation_id, facts)
            temp_data = self._get_candidate_data(temptation_id, facts)
            
            # 获取诱惑点的具体需求 (如果是 -1 或获取失败，给个默认描述)
            temp_demand = temp_data.cvrp_factors.demand if temp_data else "Unknown"
            
            # 构建强有力的拒绝理由
            narrative = {
                "strategy_name": "Capacity Awareness",
                "reasoning_focus": (
                    f"We select {gt_str} as it fits within the remaining load ({remaining_cap:.1f}) "
                    f"and aligns with the route."
                ),
                "conflict_handling": (
                    f"REJECT GREEDY: Visually {temp_str} is closer. However, **cite the Dashboard data**: "
                    f"its Demand ({temp_demand}) exceeds Remaining Capacity ({remaining_cap:.1f}). "
                    f"Thus, it is infeasible."
                )
            }

        # =========================================================
        # 🎭 Playbook A: Angular Sweep (扇区扫描) - 引用对齐
        # =========================================================
        elif abs(cvrp.angle_from_current) < 35.0:
            narrative = {
                "strategy_name": "Angular Sweep",
                "reasoning_focus": (
                    f"Selecting {gt_str} maintains the counter-clockwise sweep direction. "
                    "This clears the current angular sector sequentially."
                ),
                "conflict_handling": "Prioritize rotational order to prevent route crossing."
            }

        # =========================================================
        # 🎭 Playbook D: Sector Seeding (新扇区)
        # =========================================================
        elif abs(cvrp.angle_from_current) >= 35.0:
             narrative = {
                "strategy_name": "Sector Seeding",
                "reasoning_focus": (
                    f"We jump to {gt_str} to initiate coverage of a new angular sector "
                    "far from the previous cluster."
                ),
                "conflict_handling": "Strategic repositioning overrides local density."
            }

        return narrative

    def inject_logic(self, 
                     verified_desc: str, 
                     text_dashboard: str,  # <--- [NEW] 传入文本观察
                     spatial_facts: Dict[str, Any], 
                     problem_context: str = "CVRP") -> str:
        """
        Generates the 'Thought' component.
        Now enforces Multimodal Cross-Verification (Visuals + Dashboard Data).
        """
        gt_id = spatial_facts['gt_id']
        gt_str = self._get_node_fmt(gt_id, spatial_facts)
        
        # 1. 获取剧本
        narrative = self._select_strategic_narrative(spatial_facts)
        
        # 2. 构建 Prompt
        system_instruction = (
            "You are an expert Logistics Planner solving a CVRP instance. "
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
            f"1. **The Anchor**: Start by citing the Visual Observation (e.g., 'Visually, {gt_str} aligns with...').\n"
            f"2. **The Verification**: Explicitly **cite specific numbers** from the Dashboard Data to support the visual claim "
            f"(e.g., 'Data confirms Demand 3.0 fits in Remaining 5.0' or 'Distance 0.25 is minimal').\n"
            f"3. **The Strategy**: Conclude with the Strategy **'{narrative['strategy_name']}'**.\n"
            f"4. **Rejection Logic**: If rejecting a closer node (Conflict), you MUST cite the Demand/Capacity numbers that cause the rejection.\n"
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
from typing import Dict, Any, Optional

class RefinementModule:
    """
    Dedicated CVRP Editor.
    Final Assembly Line: Ensures consistent ID formatting and strictly enforces logistics terminology.
    """
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def _get_target_info(self, facts: Dict[str, Any]) -> tuple[str, str]:
        """
        根据 Facts 自动计算标准化的 ID 格式和决策框内容。
        Returns: (display_fmt, boxed_content)
        Example: ("Option A [Node 3]", "A")
        """
        gt_id = facts['gt_id']
        
        # 1. 如果是 Depot (Node 0)
        if gt_id == 0:
            return "The Depot [Node 0]", "0" # 或者 "Depot" 看你的数据集约定
            
        # 2. 如果是普通节点
        cand = next((c for c in facts['candidates'] if c.id == gt_id), None)
        label = cand.label if cand else "?" # e.g. "A"
        
        display_fmt = f"Option {label} [Node {gt_id}]"
        boxed_content = label # CVRP 通常输出选项 Label (A, B, C...)
        
        return display_fmt, boxed_content

    def assemble_cot(self, 
                     verified_desc: str, 
                     logic_reasoning: str, 
                     facts: Dict[str, Any]) -> str:
        """
        Assembles the final CVRP Chain-of-Thought.
        
        Args:
            verified_desc: Output from CVRPPerceptionModule
            logic_reasoning: Output from CVRPLogicInjectionModule
            facts: The master data dict containing IDs and Labels
        """
        
        # 1. 准备标准格式 (Python 逻辑，绝对正确)
        target_fmt, boxed_val = self._get_target_info(facts)
        decision_str = f"\\boxed{{{boxed_val}}}"
        
        # 2. 构造 Prompt：定义 "Logistics Editor" 角色
        system_instruction = (
            "You are an expert Logistics Editor refining training data for a CVRP solver. "
            "Your task: Polish the Observation and Thought into a strict, concise format. "
            "Constraint: Maximize information density. Eliminate redundancy."
        )

        user_prompt = (
            f"### Raw Input Data\n"
            f"1. [Raw Observation]: {verified_desc}\n"
            f"2. [Raw Thought]: {logic_reasoning}\n"
            f"3. [Target Identity]: {target_fmt}\n\n"
            
            f"### Refinement Task\n"
            f"Rewrite the inputs into two distinct parts separated by '|||'.\n\n"
            
            f"**Part 1: <Observation>** (Visual Facts)\n"
            f"- Summarize the visual location: Quadrant, Direction, and Alignment.\n"
            f"- **MANDATORY**: Use the exact format **'{target_fmt}'** to refer to the target.\n"
            f"- Remove fluff like 'I can see'. Max 100 words.\n\n"
            
            f"**Part 2: <Thought>** (Strategic Logic)\n"
            f"- Explain the logistics strategy (e.g., 'Angular Sweep', 'Capacity Awareness', 'Depot Refill').\n"
            f"- **CRITICAL**: Do NOT repeat the location facts. Explain the *consequence* (e.g., 'to clear the sector', 'to avoid overload').\n"
            f"- Max 100 words.\n\n"
            
            f"### Output Example\n"
            f"Option A [Node 12] aligns with the current angular sector. ||| Adhering to the Angular Sweep strategy, we select it to preserve rotational order and avoid route crossing.\n\n"
            
            f"### Your Output:"
        )

        try:
            # 3. LLM Generation
            response = self.llm.generate(
                system_prompt=system_instruction,
                text=user_prompt,
                max_tokens=2048,
                temperature=0.5 
            )
            
            # 4. 鲁棒解析
            parts = re.split(r'\s*\|\|\|\s*', response.strip())
            
            if len(parts) >= 2:
                obs_clean = parts[0].strip()
                thought_clean = parts[1].strip()
            else:
                # Fallback: 如果 LLM 抽风没加分隔符
                print(f"Refinement Warning: Separator missing. Using raw inputs.")
                obs_clean = verified_desc.strip()
                thought_clean = logic_reasoning.strip()

            # 5. 安全清洗 (去掉可能的 Tag 前缀)
            obs_clean = re.sub(r'^(Observation|Part 1):?', '', obs_clean, flags=re.IGNORECASE).strip()
            thought_clean = re.sub(r'^(Thought|Part 2):?', '', thought_clean, flags=re.IGNORECASE).strip()

            # 6. ID 格式终极校验 (Safety Net)
            # 如果 LLM 还是把 Option A [Node 3] 写成了 "Node 3"，这里强制替换回来
            # 这是一个简单的文本替换，确保 100% 一致性
            if target_fmt not in obs_clean:
                # 尝试把 "Option A" 或 "Node 3" 替换为全称
                # 这里只做简单的 fallback，防止替换错误
                pass 

            final_cot = (
                f"<Observation> {obs_clean} </Observation>\n"
                f"<Thought> {thought_clean} </Thought>\n"
                f"<Decision> {decision_str} </Decision>"
            )
            return final_cot

        except Exception as e:
            print(f"Refinement Failed: {e}")
            # Fallback
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
    demand: List[int],
    vehicle_capacity: int,
    vlm_agent: VLMAgent, 
    llm_agent: LLMAgent,
    debug_image_dir: str = None
) -> List[Dict[str, Any]]:
    """
    Processes a single trajectory (sequence of steps) to generate CoT data.
    
    Args:
        trajectory_steps: List of step dicts: {'obs': str, 'trajectory': str, 'image': str, 'candidates': list, ...}
        node_coords: Dictionary of node coordinates {id: [x, y]}
        demand: List of demand for each node
        vehicle_capacity: Capacity of each vehicle
        vlm_agent: Agent for Perception Module
        llm_agent: Agent for Logic Injection Module
        debug_image_dir: Directory containing images (if needed for VLM)
        
    Returns:
        List of processed step dicts with 'cot' field added.
    """
    # Initialize Modules
    geo_engine = GeometryEngine(coords=node_coords, demands=demand)
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
            current_load=step_data['current_load'],
            vehicle_capacity=vehicle_capacity,
        )

        prev_node = current_idx
        current_idx = action_id
        
        # 4. Prepare Image (Base64)
        image_b64 = step_data.get('image', None)
        # 5. Module 2: Perception Loop
        verified_desc = perception_module.run_perception_loop(image_b64, spatial_facts) # TODO: 这里的verified——desc是结构体
        if verified_desc is None:
            print(f"Step {step_idx} failed perception loop.")
            continue
        
        # 6. Module 3: Logic Injection
        logic_reasoning = logic_module.inject_logic(verified_desc, obs_text, spatial_facts)
        
        # 7. Module 4: Refinement
        final_cot = refinement_module.assemble_cot(verified_desc, logic_reasoning, spatial_facts)
        
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
    log_file = "generation_process_cvrp.log"
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
        demand = traj_data.get("demand", [])
        # 在最开始添加0
        demand.insert(0, 0)
        vehicle_capacity = traj_data.get("capacity", [])
        
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
                "current_load":traj_data['load_list'][i],
                "trajectory_idx": traj_idx # Add context
            }
            trajectory_steps.append(step_dict)

        # Run Pipeline
        processed_steps = process_single_trajectory(
            trajectory_steps,
            node_coords,
            demand,
            vehicle_capacity,
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
    input_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/cvrp_agent_output.json"
    output_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/cvrp_cot_dataset.json"
    
    # Run
    main(input_path, output_path)
