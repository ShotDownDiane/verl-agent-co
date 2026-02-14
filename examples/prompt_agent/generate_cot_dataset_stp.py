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
    """宏观态势因子：用于描述 '局势如何' (STP Global State)"""
    distribution_type: str           # "Clustered", "Uniform", "Mixed"
    connected_terminals_count: int   # 当前已连接到主组件的终端数量
    total_terminals: int             # 总终端数量
    progress_ratio: float            # 任务进度 (已连接终端 / 总终端)
    disjoint_sets_count: int         # 当前剩余的连通分量数量

    def to_json(self):
        return dataclasses.asdict(self)

@dataclass
class STPFactors:
    """STP 微观因子：用于支撑 '连通性优先' 策略"""
    is_merge: bool                   # 是否连接了两个不同的连通分量 (关键布尔值)
    edge_cost: float                 # 建设成本 (欧几里得距离)
    component_size_gain: int         # 连接后，所在组件的总节点数增加量 (若 merge=False 则为 0)
    is_bridge_to_isolated: bool      # 是否连接了一个孤立的终端点
    source_comp: int                 # 起点所属组件ID
    target_comp: int                 # 终点所属组件ID

    def to_json(self):
        return dataclasses.asdict(self)

@dataclass
class CandidateMeta:
    """单个候选边的完整几何元数据"""
    id: Any # Tuple[int, int]
    label: str                       # e.g., "A", "B"
    coords: Tuple[Tuple[float, float], Tuple[float, float]] # Edge Endpoints ((x1,y1), (x2,y2))
    midpoint: Tuple[float, float]    # Edge Midpoint
    dist: float                      # 距离 (Edge Cost)
    orientation: str                 # 边的走向 (e.g., "Vertical", "Horizontal", "Diagonal")
    stp_factors: Optional[STPFactors] = None # STP Specific

    def to_json(self):
        return dataclasses.asdict(self)

# ==========================================
# Core Engine
# ==========================================

class UnionFind:
    """Helper class for tracking connected components."""
    def __init__(self, n):
        self.parent = list(range(n))
        self.size = [1] * n
        self.count = n # Number of disjoint sets

    def find(self, i):
        if i >= len(self.parent):
            raise ValueError(f"Index {i} out of bounds for UnionFind of size {len(self.parent)}")
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Union by size
            if self.size[root_i] < self.size[root_j]:
                root_i, root_j = root_j, root_i
            self.parent[root_j] = root_i
            self.size[root_i] += self.size[root_j]
            self.count -= 1
            return True
        return False

    def get_size(self, i):
        return self.size[self.find(i)]

class TaskContextManager:
    """
    管理不同运筹任务的'世界观'。
    """
    PROFILES = {
        "STP": {
            "role_def": "You are an infrastructure network engineer solving the Steiner Tree Problem (STP).",
            "objective": "Connect all required terminal nodes with minimum total edge cost.",
            "visual_focus": "Focus on bridging gaps between disjoint clusters and connecting isolated terminals.",
            "strategy_keywords": ["Connectivity First", "Critical Bridge", "Cost-Efficiency", "Loop Rejection"]
        }
    }

    @classmethod
    def get_system_prompt(cls, task_type: str) -> str:
        profile = cls.PROFILES.get(task_type, cls.PROFILES["STP"])
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
                 edge_list: List[Tuple[int, int]],
                 edge_weights: Dict[Tuple[int, int], float],
                 terminals: List[int], # STP Specific
                 depot_idx: int = 0, 
                 global_bounds: Tuple[float, float, float, float] = None):
        """
        Args:
            coords: Shape (N, 2) array of all node coordinates.
            edge_list: List of edges (u, v) where u, v are node indices.
            edge_weights: Dict mapping (u, v) to edge cost.
            terminals: List of terminal node indices that MUST be connected.
            depot_idx: Index of the start/depot node (often one of the terminals).
            global_bounds: Optional (min_x, min_y, max_x, max_y) for static normalization.
        """
        self.coords = coords
        self.coords = np.array(list(coords.values()))
        self.edge_list = np.array(edge_list)[0]
        self.edge_weights_2d = np.array(edge_weights)[0] # shape (N,N)
        self.edge_weights_1d = self.edge_weights_2d[self.edge_list[:, 0], self.edge_list[:, 1]] # shape (E,)
        self.terminals = terminals[0]
        self.depot_idx = depot_idx
        self.total_nodes = self.coords.shape[0]
        
        # Initialize Union-Find structure
        # NOTE: This UF should ideally persist across steps to track connectivity.
        # However, the `analyze_step` is stateless. We need to reconstruct state or pass it in.
        # For this implementation, we will pass `edges_taken` to `analyze_step`.
        
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
    
    def get_orientation(self, start: np.ndarray, end: np.ndarray) -> str:
        """Determines if edge is Vertical, Horizontal, or Diagonal."""
        dx = abs(end[0] - start[0])
        dy = abs(end[1] - start[1])
        
        if dx == 0: return "Vertical"
        if dy == 0: return "Horizontal"
        
        slope = dy / dx
        if slope > 2.0: return "Vertical"
        elif slope < 0.5: return "Horizontal"
        else: return "Diagonal"

    # --- Main Analysis Pipelines ---

    def analyze_general(self, edges_taken: List[Tuple[int, int]]) -> GeneralFactors:
        """生成通用宏观因子"""
        # Reconstruct UF to check connectivity
        uf = UnionFind(self.total_nodes)
        for u, v in edges_taken:
            uf.union(u, v)
            
        # Count connected terminals
        # Assuming Terminal 0 (or first terminal) is the 'root' of the main component we care about?
        # Or just check max connected terminals in one component.
        
        max_connected = 0
        if not self.terminals:
            connected_count = 0
        else:
            # Group terminals by component
            term_comps = {}
            for t in self.terminals:
                root = uf.find(t)
                term_comps[root] = term_comps.get(root, 0) + 1
            max_connected = max(term_comps.values()) if term_comps else 0
            
        total_terminals = len(self.terminals) if self.terminals else 1
        
        return GeneralFactors(
            distribution_type=self.global_distribution,
            connected_terminals_count=max_connected,
            total_terminals=total_terminals,
            progress_ratio=max_connected / total_terminals,
            disjoint_sets_count=uf.count
        )

    def analyze_stp_micro(self, 
                          candidates: List[Any],
                          edges_taken: List[Tuple[int, int]] 
                          ) -> Dict[Any, STPFactors]:
        """
        生成 STP 微观因子 (Union-Find Analysis)
        """
        # 1. Rebuild Union-Find State
        uf = UnionFind(self.total_nodes)
        for u, v in edges_taken:
            uf.union(u, v)
            
        results = {}
        for item in candidates:
            # Handle Edge Index (int) or Edge Tuple
            if isinstance(item, (int, np.integer)):
                idx = int(item)
                if idx < 0 or idx >= len(self.edge_list):
                    continue
                u, v = self.edge_list[idx]
                cid = idx
            elif isinstance(item, (list, tuple)):
                u, v = item
                cid = tuple(item)
            else:
                 continue
                
            comp_u = uf.find(u)
            comp_v = uf.find(v)
            
            # A. Edge Cost
            p1 = self._get_coords(u)
            p2 = self._get_coords(v)
            cost = np.linalg.norm(p1 - p2)
            
            # B. Is Merge?
            is_merge = (comp_u != comp_v)
            
            # C. Component Size Gain
            size_gain = uf.get_size(v) if is_merge else 0 # Simplified
            
            # D. Is Bridge to Isolated Terminal
            def is_isolated_terminal_comp(node_idx):
                c_root = uf.find(node_idx)
                c_size = uf.get_size(node_idx)
                # Check if this component contains any terminals
                has_terminal = any(uf.find(t) == c_root for t in self.terminals)
                return has_terminal and c_size < 3 
            
            is_bridge_to_isolated = False
            if is_merge:
                if is_isolated_terminal_comp(u) or is_isolated_terminal_comp(v):
                    is_bridge_to_isolated = True

            results[cid] = STPFactors(
                is_merge=is_merge,
                edge_cost=float(cost),
                component_size_gain=size_gain,
                is_bridge_to_isolated=is_bridge_to_isolated,
                source_comp=comp_u,
                target_comp=comp_v
            )
            
        return results, uf.count 

    # --- Master Analysis Interface ---

    def analyze_step(self, 
                     candidates: List[Any], 
                     gt_edge: Any, 
                     edges_taken: List[Tuple[int, int]] = []) -> Dict[str, Any]:
        """
        Main entry point for generating all geometric truths for a step.
        """
        # 1. Macro Analysis
        general_facts = self.analyze_general(edges_taken)
        
        # 2. Micro Analysis (STP)
        # Filter valid candidates (tuples or ints)
        valid_cands = []
        for c in candidates:
            if isinstance(c, (list, tuple)):
                valid_cands.append(tuple(c))
            elif isinstance(c, (int, np.integer)):
                valid_cands.append(int(c))
        
        stp_facts_map, disjoint_count = self.analyze_stp_micro(valid_cands, edges_taken)
        
        # 3. Assemble Candidates Metadata
        candidates_meta = []
        min_cost = float('inf')
        
        for i, cid in enumerate(valid_cands):
            if isinstance(cid, (int, np.integer)):
                u, v = self.edge_list[cid]
            else:
                u, v = cid
            
            p1 = self._get_coords(u)
            p2 = self._get_coords(v)
            midpoint = (p1 + p2) / 2
            dist = np.linalg.norm(p1 - p2)
            
            if dist < min_cost: min_cost = dist
            
            label = idx2option.get(i, f"UNK_{i}")

            candidates_meta.append(CandidateMeta(
                id=cid,
                label=label,
                coords=(tuple(p1), tuple(p2)),
                midpoint=tuple(midpoint),
                dist=float(dist),
                orientation=self.get_orientation(p1, p2),
                stp_factors=stp_facts_map.get(cid)
            ))

        # 4. GT Specifics
        if isinstance(gt_edge, (int, np.integer)):
            gt_id = int(gt_edge)
            gt_u, gt_v = self.edge_list[gt_id]
        else:
            gt_id = gt_edge
            gt_u, gt_v = gt_id
            
        gt_p1 = self._get_coords(gt_u)
        gt_p2 = self._get_coords(gt_v)
        gt_dist = np.linalg.norm(gt_p1 - gt_p2)
        
        is_cheapest = abs(gt_dist - min_cost) < 1e-6
        
        # 5. Temptation Generation
        temptation_id = None
        gt_factors = stp_facts_map.get(gt_edge)
        
        sorted_by_cost = sorted(candidates_meta, key=lambda x: x.dist)
        for cand in sorted_by_cost:
            # Handle equality check based on type
            is_same = False
            if isinstance(cand.id, (int, np.integer)) and isinstance(gt_id, (int, np.integer)):
                is_same = (cand.id == gt_id)
            elif isinstance(cand.id, (list, tuple)) and isinstance(gt_id, (list, tuple)):
                is_same = (set(cand.id) == set(gt_id))
            else:
                 # Mixed types or other types: assume not same unless values match directly
                 is_same = (cand.id == gt_id)
                 
            if is_same: continue
            
            # Temptation Logic: Cheaper Loop vs Expensive Merge
            if gt_factors and gt_factors.is_merge:
                if cand.stp_factors and not cand.stp_factors.is_merge:
                    temptation_id = cand.id
                    break 
            
            if temptation_id is None:
                temptation_id = cand.id
        return {
            "general": general_facts,
            "candidates": candidates_meta,
            "gt_id": gt_edge,
            "gt_stats": {
                "dist": gt_dist,
                "orientation": self.get_orientation(gt_p1, gt_p2),
                "is_cheapest": is_cheapest,
                "stp_factors": stp_facts_map.get(gt_edge)
            },
            "temptation_id": temptation_id,
            "global_state": {
                "disjoint_sets": disjoint_count,
                "total_cost": sum([np.linalg.norm(self._get_coords(u)-self._get_coords(v)) for u,v in edges_taken])
            }
        }
# ==========================================
# Module 2: Perception Loop
# ==========================================
from typing import Dict, Any, Tuple, Optional, List
import re

class PerceptionModule:
    def __init__(self, vlm_agent):
        self.vlm = vlm_agent
        
        # --- 1. 词表扩展 (STP Specifics) ---
        self.orientation_map = {
            'Vertical': ['vertical', 'up-down', 'north-south', 'standing'],
            'Horizontal': ['horizontal', 'left-right', 'east-west', 'flat'],
            'Diagonal': ['diagonal', 'slanted', 'angled', 'tilted']
        }
        
        # 拓扑特征关键词 (Topology Focus)
        self.topology_keywords = [
            'bridge', 'gap', 'disjoint', 'cluster', 'group', 'backbone', 
            'spanning', 'connect', 'merge', 'link', 'reach'
        ]
        self.isolation_keywords = ['isolated', 'alone', 'terminal', 'dead-end', 'remote']
        self.loop_keywords = ['loop', 'cycle', 'redundant', 'internal', 'already connected']

    def _check_keywords(self, text: str, keywords: List[str]) -> bool:
        text_lower = text.lower()
        return any(k in text_lower for k in keywords)

    def construct_grounding_prompt(self, facts: Dict[str, Any]) -> str:
        """
        STP 专用感知引导：关注连通性、桥梁和组件状态。
        """
        gt_id = facts['gt_id']
        gt_stats = facts['gt_stats']
        stp_factors = gt_stats['stp_factors'] # STPFactors dataclass
        general = facts['general'] # dataclass
        temptation_id = facts.get('temptation_id')

        # Helper to format ID
        def get_fmt(cid):
            cand = next((c for c in facts['candidates'] if c.id == cid), None)
            lbl = cand.label if cand else "?"
            return f"Option {lbl} [Edge {cid}]"

        gt_str = get_fmt(gt_id)

        # 1. 基础定位
        prompt = (
            f"Please generate a precise visual observation for the candidate edge **{gt_str}**.\n"
            f"Context: We are building a network. Current status: {general.distribution_type} distribution.\n"
            f"Ground Truths to Describe:\n"
            f"1. **Orientation**: {gt_str} is oriented **{gt_stats['orientation']}**.\n"
        )

        # 2. 拓扑特征引导 (STP Specifics)
        if stp_factors.is_merge:
            prompt += f"2. **Topology**: This edge acts as a **Bridge**. It crosses a **Gap** to connect two previously **Disjoint Clusters** (Component {stp_factors.source_comp} <-> {stp_factors.target_comp}).\n"
        else:
            prompt += f"2. **Topology**: This edge is **Internal** to the current cluster (Potential Loop/Redundant).\n"

        if stp_factors.is_bridge_to_isolated:
            prompt += f"3. **Connectivity**: It specifically targets an **Isolated Terminal**, connecting it to the main backbone.\n"
        
        # 3. 诱惑点对比 (Trap Rejection)
        if not gt_stats['is_cheapest'] and temptation_id is not None:
            temp_str = get_fmt(temptation_id)
            prompt += (
                f"4. **Comparison**: Explicitly mention that **{temp_str}** is visually shorter/cheaper, "
                f"but explain that {gt_str} is critical because it bridges a gap (Merges Components), whereas {temp_str} might just be a local loop.\n"
            )
        
        prompt += "\nOutput a concise 'Infrastructure Engineer' style observation paragraph confirming these visual topology facts."
        return prompt

    def verify_description(self, description: str, facts: Dict[str, Any]) -> Tuple[bool, str]:
        """
        STP 验证逻辑：ID -> 方位 -> 拓扑功能 (Merge/Bridge) -> 陷阱排除
        """
        desc_lower = description.lower()
        gt_id = str(facts['gt_id'])
        gt_stats = facts['gt_stats']
        stp_factors = gt_stats['stp_factors']
        temptation_id = facts.get('temptation_id')

        # --- 1. Identity Check ---
        if not re.search(rf"\b(candidate|node|option|edge)\s*{re.escape(gt_id)}\b", desc_lower):
             return False, f"Failed to explicitly mention target 'Option/Edge {gt_id}'."

        # --- 2. Orientation Check ---
        gt_orient = gt_stats['orientation']
        expected_kws = self.orientation_map.get(gt_orient, [])
        if not self._check_keywords(desc_lower, expected_kws):
            return False, f"Failed to identify orientation '{gt_orient}'. Expected keywords: {expected_kws}"

        # --- 3. Topology Feature Check (STP) ---
        # A. Merge / Bridge Check
        if stp_factors.is_merge:
            if not self._check_keywords(desc_lower, self.topology_keywords):
                return False, f"Missed topological feature: This edge is a 'Bridge/Merge', but description didn't use terms like {self.topology_keywords}."

        # B. Isolation Check
        if stp_factors.is_bridge_to_isolated:
            if not self._check_keywords(desc_lower, self.isolation_keywords):
                return False, f"Missed connectivity feature: Target connects an 'Isolated' node, but description missed it."

        # --- 4. Temptation/Comparison Check ---
        if temptation_id is not None and not gt_stats['is_cheapest']:
            temp_id_str = str(temptation_id)
            if not re.search(rf"\b{re.escape(temp_id_str)}\b", desc_lower):
                return False, f"Failed to compare with the cheaper temptation 'Edge {temptation_id}'. Evaluation requires 'Loop Rejection' logic."

        return True, "Verified"

    def construct_reflexion_prompt(self, gt_id: int, previous_response: str, missing_reason: str, label: str = "?") -> str:
        """
        STP 反思提示词
        """
        gt_str = f"Option {label} [Edge {gt_id}]"
        return (
            f"### Review of your previous output:\n"
            f"**Your Draft**: \"{previous_response}\"\n"
            f"**Critique**: {missing_reason}\n\n"
            
            f"### New Task:\n"
            f"Please **REWRITE** the observation for {gt_str} as an Infrastructure Engineer.\n"
            f"1. Correctly describe the location.\n"
            f"2. **MANDATORY**: Fix the missing topological fact (Bridge/Gap/Isolation) mentioned in the critique.\n"
            f"3. Maintain a professional tone."
        )

    def run_perception_loop(self, image_b64: str, spatial_facts: Dict[str, Any], max_retries: int = 3) -> Optional[str]:
        gt_id = spatial_facts['gt_id']
        
        # Find Label for Reflexion
        gt_cand = next((c for c in spatial_facts['candidates'] if c.id == gt_id), None)
        gt_label = gt_cand.label if gt_cand else "?"

        # 1. 第一次尝试
        current_prompt = self.construct_grounding_prompt(spatial_facts)
        task_system_prompt = TaskContextManager.get_system_prompt("STP")
        system_instruction = (
            f"{task_system_prompt}\n\n"
            "Your specific sub-task is: Ground the topological truths (Merges, Bridges) into a visual description."
        )

        for attempt in range(max_retries):
            print(f"--- Attempt {attempt + 1} ---")
            
            response = self.vlm.generate(
                system_prompt=system_instruction,
                text=current_prompt, 
                image=image_b64,
                max_tokens=2048
            )
            print(f"Prompt:\n{current_prompt}\n")
            print(f"Response:\n{response}\n")
            
            # 2. 验证
            is_valid, reason = self.verify_description(response, spatial_facts)
            
            if is_valid:
                return response # 成功
            
            # 3. 失败：Reflexion
            current_prompt = self.construct_reflexion_prompt(
                gt_id=gt_id,
                previous_response=response,
                missing_reason=reason,
                label=gt_label
            )
            
        return None # 彻底失败

# ==========================================
# Module 3: Logic Injection
# ==========================================
from typing import Dict, Any, Optional

def build_stp_dashboard(facts: Dict[str, Any]) -> str:
    """
    Phase 2: Text Observation Builder (仪表盘构建)
    将图论状态转化为 LLM 可读的文本证据。
    """
    global_state = facts['global_state']
    candidates = facts['candidates']
    
    # 1. Global Status Line
    dashboard = (
        f"Status: {global_state['disjoint_sets']} Disjoint Groups Remaining | "
        f"Total Cost: {global_state['total_cost']:.2f}\n"
    )
    dashboard += "-" * 50 + "\n"
    
    # 2. Candidate Details
    # Sort by ID or Label for consistent display, or by Cost? 
    # Usually Dashboard lists options.
    for cand in candidates:
        factors = cand.stp_factors
        merge_tag = "(MERGE)" if factors.is_merge else "(INTERNAL/LOOP)"
        
        line = (
            f"Option {cand.label} [Edge {cand.id}]: "
            f"Cost {cand.dist:.2f} | "
            f"Connects Comp {factors.source_comp} <-> Comp {factors.target_comp} {merge_tag}"
        )
        dashboard += line + "\n"
        
    return dashboard

class LogicInjectionModule:
    """
    STP Logic Injection (Network Engineer Persona).
    """
    def __init__(self, llm_agent):
        self.llm = llm_agent

    def _get_node_fmt(self, node_id: Any, facts: Dict[str, Any]) -> str:
        cand = next((c for c in facts['candidates'] if c.id == node_id), None)
        lbl = cand.label if cand else "?"
        return f"Option {lbl} [Edge {node_id}]"

    def _select_strategic_narrative(self, facts: Dict[str, Any]) -> Dict[str, str]:
        """
        [导演中心]：STP 专用剧本分发器
        剧本 A: 战略合并 (Critical Merge)
        剧本 B: 贪心延伸 (Greedy Extension)
        剧本 C: 拒绝环路 (Loop Rejection)
        """
        gt_id = facts['gt_id']
        gt_stats = facts['gt_stats']
        stp_factors = gt_stats['stp_factors'] # STPFactors
        temptation_id = facts.get('temptation_id')
        
        gt_str = self._get_node_fmt(gt_id, facts)
        gt_cost = f"{stp_factors.edge_cost:.2f}"
        
        # Default Narrative
        narrative = {
            "strategy_name": "Cost-Efficiency",
            "reasoning_focus": f"Selecting {gt_str} offers the best balance of cost and connectivity.",
            "conflict_handling": "None"
        }

        # 🎭 Playbook A: Critical Merge (Strategic Merge)
        # 场景：连接两个不同的组件，且不是单纯的孤立点延伸（或者优先级更高）
        if stp_factors.is_merge and not stp_factors.is_bridge_to_isolated:
            narrative = {
                "strategy_name": "Critical Merge",
                "reasoning_focus": (
                    f"This edge bridges a gap between two major clusters (Comp {stp_factors.source_comp} & {stp_factors.target_comp}). "
                    f"Reducing the number of disjoint sets is the primary objective."
                ),
                "conflict_handling": "Prioritize merging components over cheaper internal edges."
            }
            
            # Conflict with Cheap Loop
            if temptation_id is not None and not gt_stats['is_cheapest']:
                temp_cand = next((c for c in facts['candidates'] if c.id == temptation_id), None)
                temp_cost = f"{temp_cand.dist:.2f}" if temp_cand else "?"
                narrative["conflict_handling"] = (
                    f"REJECT LOOP: Dashboard shows Option {temp_cand.label} is cheaper ({temp_cost}) but is Internal (Non-Merge). "
                    f"We accept the higher cost ({gt_cost}) of {gt_str} to achieve a Critical Merge."
                )

        # 🎭 Playbook B: Greedy Extension (To Isolated)
        # 场景：连接孤立点
        elif stp_factors.is_bridge_to_isolated:
            narrative = {
                "strategy_name": "Greedy Extension",
                "reasoning_focus": (
                    f"Visually points to an isolated terminal. Dashboard confirms it connects to a singleton component. "
                    f"Extending the network to cover this node is necessary."
                ),
                "conflict_handling": "Standard greedy extension for isolated nodes."
            }

        # 🎭 Playbook C: Loop Rejection (Focus on why we picked GT over a Loop)
        # This is implicitly handled in Conflict Handling of A/B, but if GT itself is just a "Valid Move" vs "Invalid Move":
        # If we are here, it means GT is likely a Merge (since we filter for STP).
        
        return narrative

    def inject_logic(self, 
                     verified_desc: str, 
                     text_dashboard: str, 
                     spatial_facts: Dict[str, Any], 
                     problem_context: str = "STP") -> str:
        """
        Generates the <Thought> component using Dashboard Data.
        """
        gt_id = spatial_facts['gt_id']
        gt_str = self._get_node_fmt(gt_id, spatial_facts)
        
        narrative = self._select_strategic_narrative(spatial_facts)
        
        system_instruction = (
            "You are an Infrastructure Network Engineer. "
            "Synthesize the **Visual Topology** and the **Dashboard Metrics** into a strategic decision. "
            "Keywords: Connectivity, Merge, Cost-Efficiency, Disjoint Sets."
        )

        user_prompt = (
            f"**Task**: Generate the <Thought> rationale for choosing **{gt_str}**.\n\n"
            
            f"**Input 1: Visual Topology**\n"
            f"\"{verified_desc}\"\n\n"
            
            f"**Input 2: Dashboard Data**\n"
            f"```text\n{text_dashboard}\n```\n\n"
            
            f"**Input 3: Strategic Directive**\n"
            f"- **Strategy**: {narrative['strategy_name']}\n"
            f"- **Core Logic**: {narrative['reasoning_focus']}\n"
            f"- **Conflict/Trade-off**: {narrative['conflict_handling']}\n\n"
            
            f"**Requirements**:\n"
            f"1. **Visual Anchor**: Start with 'Visually, ...' citing the Bridge/Gap.\n"
            f"2. **Data Verification**: Explicitly cite the **Cost** and **Component IDs** from Dashboard.\n"
            f"3. **Strategic Conclusion**: Conclude with '{narrative['strategy_name']}'.\n"
            f"4. **Loop Rejection**: If relevant, explain why the cheaper option was rejected (Internal Loop).\n"
            f"5. Max 200 words. Professional Engineering Tone."
        )
        
        try:
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
        Assembles the final CoT with 'Infrastructure Network Engineer' persona.
        """
        decision_str = f"\\boxed{{{trajectory_action}}}"
        
        system_instruction = (
            "You are an expert Infrastructure Network Engineer. "
            "Refine the raw inputs into a professional 'Observation-Thought' pair. "
            "Tone: Technical, Precise, Objective. "
            "Terminology: Use 'Linkage', 'Partition', 'Connectivity', 'Cost-Efficiency'."
        )

        user_prompt = (
            f"### Raw Input Data\n"
            f"1. [Raw Observation]: {verified_desc}\n"
            f"2. [Raw Thought]: {logic_reasoning}\n\n"
            
            f"### Refinement Task\n"
            f"Rewrite into two distinct parts separated by '|||'.\n\n"
            
            f"**Part 1: <Observation>**\n"
            f"- Summarize topological facts: Bridge, Gap, Disjoint Clusters.\n"
            f"- Remove 'I can see'.\n\n"
            
            f"**Part 2: <Thought>**\n"
            f"- Explain the strategic decision (Critical Merge, Loop Rejection).\n"
            f"- **Strictly Preserve Numbers**: Do not summarize costs or IDs. Keep them exact.\n"
            f"- Focus on maximizing connectivity and minimizing cost.\n\n"
            
            f"### Output Format:\n"
            f"Option A acts as a bridge crossing the central gap. ||| Dashboard confirms a cost of 0.12 merging Comp 1 and 2. We execute this Critical Merge to reduce disjoint sets.\n\n"
            
            f"### Your Output:"
        )

        try:
            response = self.llm.generate(
                system_prompt=system_instruction,
                text=user_prompt,
                max_tokens=2048,
                temperature=0.5
            )
            
            parts = re.split(r'\s*\|\|\|\s*', response.strip())
            
            if len(parts) >= 2:
                obs_clean = parts[0].strip()
                thought_clean = parts[1].strip()
            else:
                obs_clean = verified_desc.strip()
                thought_clean = logic_reasoning.strip()

            obs_clean = re.sub(r'^(Observation|Part 1):?', '', obs_clean, flags=re.IGNORECASE).strip()
            thought_clean = re.sub(r'^(Thought|Part 2):?', '', thought_clean, flags=re.IGNORECASE).strip()

            final_cot = (
                f"<Observation> {obs_clean} </Observation>\n"
                f"<Thought> {thought_clean} </Thought>\n"
                f"<Decision> {decision_str} </Decision>"
            )
            return final_cot

        except Exception as e:
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
    terminals: List[int],
    edge_list: List[List[int]],
    edge_weights: List[float],
    vlm_agent: VLMAgent, 
    llm_agent: LLMAgent,
    debug_image_dir: str = None
) -> List[Dict[str, Any]]:
    """
    Processes a single trajectory (sequence of steps) to generate CoT data.
    """
    # Initialize Modules
    # Assume all nodes are terminals if not specified (Standard Steiner Tree / MST)
    terminals = terminals if terminals else list(node_coords.keys())
    
    geo_engine = GeometryEngine(coords=node_coords, terminals=terminals, edge_list=edge_list, edge_weights=edge_weights)
    perception_module = PerceptionModule(vlm_agent)
    logic_module = LogicInjectionModule(llm_agent)
    refinement_module = RefinementModule(llm_agent)
    
    processed_steps = []
    
    # Track history
    global NODE_COORDS
    NODE_COORDS = {int(k): tuple(v) for k, v in node_coords.items()}
    
    # STP: Track edges taken to maintain Union-Find state
    edges_taken = [] 
    
    for step_idx, step_data in enumerate(trajectory_steps):
        obs_text = step_data.get('obs', '')
        action_raw = str(step_data['trajectory'])
        
        candidates_list = step_data.get('candidates', [])
        
        action_idx = option2idx.get(action_raw.replace("\\boxed{", "").replace("}", ""), 0)
        
        if action_idx < len(candidates_list):
            action_id = candidates_list[action_idx]
        else:
            # Fallback
            if candidates_list and isinstance(candidates_list[0], (list, tuple)):
                 action_id = candidates_list[0]
            else:
                 # Should ideally be an edge tuple
                 action_id = (0, 0)
        
        # 3. Module 1: Geometry Analysis
        spatial_facts = geo_engine.analyze_step(
            candidates=candidates_list,
            gt_edge=action_id,
            edges_taken=edges_taken
        )

        # Update History (Edge Taken)
        if isinstance(action_id, (list, tuple)):
            edges_taken.append(tuple(action_id))
        elif isinstance(action_id, (int, np.integer)):
             u, v = geo_engine.edge_list[action_id]
             edges_taken.append((u, v))
        else:
            # If somehow int, try to infer? No, STP strictly edges.
            pass
        
        # 4. Prepare Image (Base64)
        image_b64 = step_data.get('image', None)
        # 5. Module 2: Perception Loop
        verified_desc = perception_module.run_perception_loop(image_b64, spatial_facts)
        if verified_desc is None:
            print(f"Step {step_idx} failed perception loop.")
            continue
        
        # 6. Generate Text Dashboard (Phase 2)
        text_dashboard = build_stp_dashboard(spatial_facts)

        # 7. Module 3: Logic Injection
        logic_reasoning = logic_module.inject_logic(verified_desc, text_dashboard, spatial_facts)
        
        # 8. Module 4: Refinement
        final_cot = refinement_module.assemble_cot(verified_desc, logic_reasoning, action_raw)
        
        new_step = step_data.copy()
        new_step['cot'] = final_cot
        processed_steps.append(new_step)
        
    return processed_steps


def main(input_file: str, output_file: str, loc_file: str = None, debug_img_dir: str = None):
    # Setup Logging to file
    log_file = "generation_process_stp.log"
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
        terminals = traj_data.get("terminals", [])
        edge_list = traj_data.get("edge_list", [])
        edge_weights = traj_data.get("edge_weights", [])
        
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
            terminals,
            edge_list,
            edge_weights,
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
    
    with open(output_file, 'w') as f:
        json.dump(all_processed_data, f, indent=2, cls=EnhancedJSONEncoder)
    print(f"Saved processed data to {output_file}")

if __name__ == "__main__":
    # Default paths
    input_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/stp_agent_output.json"
    output_path = "/root/autodl-tmp/verl-agent-co/examples/prompt_agent/stp_cot_dataset.json"
    
    # Run
    main(input_path, output_path)
