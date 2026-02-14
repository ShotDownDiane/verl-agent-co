import numpy as np
import random
from typing import List, Dict, Any

class ConfigSampler:
    """
    基于目标决策步数 (Step Size) 生成环境配置。
    确保不同类型的环境在同一个 Batch 中具有相似的序列长度。
    """
    # From Kool et al. 2019, Hottung et al. 2022, Kim et al. 2023
    CAPACITIES = {
        10: 20.0,
        15: 25.0,
        20: 30.0,
        30: 33.0,
        40: 37.0,
        50: 40.0,
        60: 43.0,
        75: 45.0,
        100: 50.0,
        125: 55.0,
        150: 60.0,
        200: 70.0,
        500: 100.0,
        1000: 150.0,
    }
    def __init__(self, 
                 worker_schema: List[Dict[str, Any]], 
                 step_levels: List[int] = [30, 50, 80, 100]):
        """
        Args:
            worker_schema: 定义了 Worker 顺序和类型的列表 (长度为 10)
            step_levels: 可选的步数难度级别
        """
        self.schema = worker_schema
        self.step_levels = step_levels


    def _get_params_by_steps(self, env_name: str, target_steps: int) -> Dict[str, Any]:
        """根据目标步数反推环境具体参数"""
        cfg = {}
        
        if env_name == 'tsp':
            # TSP: 步数 = 节点数
            cfg['num_loc'] = target_steps
            cfg['loc_distribution'] = 'uniform'
            
        elif env_name == 'cvrp':
            # CVRP: 步数 ≈ 1.2 * 节点数 (考虑回库)
            # 所以 节点数 ≈ 步数 * 0.8
            cfg['num_loc'] = int(target_steps * 0.7)
            cfg['loc_distribution'] = 'uniform'
            # 保持默认 capacity 或根据规模动态调整
            
        elif env_name == 'flp':
            # FLP: 严格对齐 "500选30" 的直觉
            # 比例因子 ~16.6，我们取 16
            cfg['to_chosen'] = target_steps
            
            # 计算总点数
            raw_num_loc = target_steps * 16
            
            # 【重要】设置上限，防止 Transformer OOM
            cfg['num_loc'] = raw_num_loc
            cfg['loc_distribution'] = 'uniform'
            if cfg['num_loc'] > 1000:
                cfg['num_loc'] = 1000
                scale = cfg['num_loc'] / raw_num_loc
                cfg['to_chosen'] = int(cfg['to_chosen'] * scale)                
            
        elif env_name == 'mclp':
            # MCLP: 步数 = 选址数量
            # MCLP: 规模比 FLP 小，更精致
            # 设定：选 K 个，候选 4K，需求 8K
            # Example (Step=30): 选30，候选120，需求240 -> 总点数360 (看起来比较舒服)
            cfg['num_facilities_to_select'] = target_steps
            cfg['num_facility'] = target_steps * 4
            cfg['num_demand'] = target_steps * 10
            
            # 同样设置一个软上限
            if cfg['num_demand'] > 800:
                scale = 800 / cfg['num_demand']
                cfg['num_facilities_to_select'] = int(cfg['num_facilities_to_select'] * scale)
                cfg['num_facility'] = int(cfg['num_facility'] * scale)
                cfg['num_demand'] = 800
            
            cfg['loc_distribution'] = 'uniform'
            
        elif env_name == 'stp':
            # STP: 假设 步数 K 对应解中大约包含 K 个节点
            # 通常总图大小是解大小的 2-3 倍
            cfg['num_node'] = target_steps * 4
            # 终端节点通常占 20% - 25%
            cfg['num_terminal'] = int(cfg['num_node'] * 0.1)
            cfg['loc_distribution'] = 'uniform'
            
        return cfg

    def __call__(self) -> List[Dict[str, Any]]:
        # 1. 在本次 Reset 中，随机选择一个统一的难度级别 (Level)
        #    这样整个 Batch 的计算量是均衡的
        current_step_target = random.choice(self.step_levels)
        
        configs = []
        for item in self.schema:
            env_name = item['env_name']
            
            # 2. 根据步数生成参数
            dynamic_params = self._get_params_by_steps(env_name, current_step_target)
            
            # 3. 构造完整配置
            cfg = {'env_name': env_name}
            cfg.update(dynamic_params)
            
            configs.append(cfg)
            
        return configs

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 定义 Schema (这对应了你原来的 mixed_configs 列表结构，指明了每个 Worker 负责什么)
    # 这里的参数只是为了占位或者定义范围
    worker_schema = [
        {"env_name": "tsp"}, {"env_name": "tsp"},
        {"env_name": "cvrp"}, {"env_name": "cvrp"},
        {"env_name": "flp"}, {"env_name": "flp"},
        {"env_name": "mclp"}, {"env_name": "mclp"},
        {"env_name": "stp"}, {"env_name": "stp"},
    ]
    difficulty_levels = [30, 40, 50, 60]

    # 2. 实例化 Sampler
    sampler = ConfigSampler(worker_schema, step_levels=difficulty_levels)

    # 3. 构建环境
    new_configs = sampler()
    print(new_configs)

    new_configs = sampler()
    print(new_configs)

    new_configs = sampler()
    print(new_configs)