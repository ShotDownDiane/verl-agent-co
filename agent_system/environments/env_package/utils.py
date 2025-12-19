import numpy as np
from pathlib import Path



def get_root_dir():
    cur = Path(__file__).resolve()
    for parent in cur.parents:
        if parent.name == "verl-agent-urban":
            return str(parent)
    raise RuntimeError("verl-agent-urban not found in path")


class WeightManager:
    def __init__(self, initial_weights, eta=10.0, epsilon=1e-8):
        """
        动态权重管理器初始化。
        
        参数：
        - initial_weights: 可迭代对象，包含 M 个初始权重 w_m(0)，并将归一化到和为 1。
        - eta: 温度参数 η，用于控制 exp(-Δ/η) 的“锐度”。
        - epsilon: 稳定项，防止除 0。
        """
        self.M = len(initial_weights)
        w = np.array(initial_weights, dtype=float)
        self.w = w / w.sum()  # 归一化
        # 历史最优 performance p_best，初始设为 -inf，方便第一次直接更新
        self.p_best = np.full(self.M, -np.inf, dtype=float)
        self.eta = eta
        self.epsilon = epsilon

    def update(self, performance):
        """
        每次训练后，用当前各目标的 performance 向量 p(n) 更新内部状态并重新计算权重。
        
        参数：
        - performance: 可迭代对象，长度 M，对应每个目标 m 的 p_m(n)。
        
        返回：
        - 新的权重向量 w(n)，长度 M。
        """
        p = np.array(performance, dtype=float)
        # 检查是否严格帕累托改善：p >= p_best 且至少有一个 >
        better_or_equal = (p >= self.p_best).all()
        strictly_better = (p > self.p_best).any()
        if better_or_equal and strictly_better:
            # 更新历史 Pareto 最优
            self.p_best = p.copy()
        
        # 计算 degradation Δ_m(n) = p_m(n) - p_m^best
        delta = p - self.p_best  # 若刚更新，则 delta 中会有 0
        
        # 计算每个目标的新权重分子：w_m(n-1) + exp(-Δ_m/η)
        add_terms = self.w + (np.exp(-delta / self.eta)-1)
        denom = add_terms.sum() + self.epsilon
        
        # 归一化得到 w_m(n)
        self.w = add_terms / denom
        
        return self.w

    def get_weights(self):
        """
        返回当前的权重向量 w(n)。
        """
        return self.w.copy()
