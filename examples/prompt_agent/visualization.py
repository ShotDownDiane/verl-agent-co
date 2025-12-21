"""
可视化模块：从环境的原始 TensorDict 生成可视化图片
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
import base64
import io
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def visualize_tsp_from_td(td, idx: int, save_path: Optional[str] = None) -> str:
    """
    从 TensorDict 中提取 TSP 坐标，生成可视化图片
    
    Args:
        td: TensorDict containing 'locs' key
        idx: Index of the instance in the batch
        save_path: Optional path to save the image file
    
    Returns:
        base64 encoded image string
    """
    try:
        locs = td["locs"][idx].cpu().numpy()  # (num_nodes, 2)
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot cities as scatter points
        ax.scatter(locs[:, 0], locs[:, 1], s=100, c='blue', marker='o', 
                  edgecolors='black', linewidths=2, zorder=3)
        
        # Label each city
        for i, (x, y) in enumerate(locs):
            ax.annotate(f'{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                       fontsize=10, fontweight='bold')
        
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title(f'TSP Problem Visualization (Instance {idx})\n{len(locs)} Cities', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        
        # Optionally save to file
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(base64.b64decode(img_base64))
            logger.info(f"Saved TSP visualization to {save_path}")
        
        return img_base64
        
    except Exception as e:
        logger.error(f"Error generating TSP visualization: {e}")
        raise


def visualize_cvrp_from_td(td, idx: int, save_path: Optional[str] = None) -> str:
    """
    从 TensorDict 中提取 CVRP 数据，生成可视化图片
    
    Args:
        td: TensorDict containing 'locs' and 'demand' keys
        idx: Index of the instance in the batch
        save_path: Optional path to save the image file
    
    Returns:
        base64 encoded image string
    """
    try:
        locs = td["locs"][idx].cpu().numpy()
        demands = td["demand"][idx].cpu().numpy()
        capacity = float(td.get("capacity", td.get("vehicle_capacity"))[idx].item())
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Depot is typically node 0
        depot = locs[0]
        customers = locs[1:]
        
        # Plot depot
        ax.scatter([depot[0]], [depot[1]], s=200, c='red', marker='s', 
                  edgecolors='black', linewidths=3, zorder=3, label='Depot')
        ax.annotate('Depot', (depot[0], depot[1]), xytext=(5, 5), 
                   textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Plot customers with demand labels
        ax.scatter(customers[:, 0], customers[:, 1], s=100, c='blue', marker='o',
                  edgecolors='black', linewidths=2, zorder=3, label='Customers')
        
        for i, (x, y) in enumerate(customers, start=1):
            ax.annotate(f'{i}\n(d={demands[i]})', (x, y), xytext=(5, 5),
                       textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title(f'CVRP Problem Visualization (Instance {idx})\n'
                    f'{len(customers)} Customers, Capacity: {capacity}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        
        # Optionally save to file
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(base64.b64decode(img_base64))
            logger.info(f"Saved CVRP visualization to {save_path}")
        
        return img_base64
        
    except Exception as e:
        logger.error(f"Error generating CVRP visualization: {e}")
        raise


def visualize_op_from_td(td, idx: int, save_path: Optional[str] = None) -> str:
    """
    从 TensorDict 中提取 OP (Orienteering Problem) 数据，生成可视化图片
    
    Args:
        td: TensorDict containing 'locs' and 'prize' keys
        idx: Index of the instance in the batch
        save_path: Optional path to save the image file
    
    Returns:
        base64 encoded image string
    """
    try:
        locs = td["locs"][idx].cpu().numpy()
        prizes = td["prize"][idx].cpu().numpy()
        max_length = float(td.get("max_length", td.get("max_route_length"))[idx].item())
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Start node is typically node 0
        start_node = locs[0]
        other_nodes = locs[1:]
        
        # Plot start node
        ax.scatter([start_node[0]], [start_node[1]], s=200, c='green', marker='s',
                  edgecolors='black', linewidths=3, zorder=3, label='Start')
        ax.annotate('Start', (start_node[0], start_node[1]), xytext=(5, 5),
                   textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Plot other nodes with prize labels
        # Size nodes by prize value
        sizes = 50 + prizes[1:] * 20  # Scale prize to node size
        ax.scatter(other_nodes[:, 0], other_nodes[:, 1], s=sizes, c='blue', marker='o',
                  edgecolors='black', linewidths=2, zorder=3, label='Nodes', alpha=0.7)
        
        for i, (x, y) in enumerate(other_nodes, start=1):
            ax.annotate(f'{i}\n(p={prizes[i]})', (x, y), xytext=(5, 5),
                       textcoords='offset points', fontsize=9)
        
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title(f'OP Problem Visualization (Instance {idx})\n'
                    f'{len(other_nodes)} Nodes, Max Length: {max_length}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal', adjustable='box')
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to base64
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        
        # Optionally save to file
        if save_path:
            with open(save_path, 'wb') as f:
                f.write(base64.b64decode(img_base64))
            logger.info(f"Saved OP visualization to {save_path}")
        
        return img_base64
        
    except Exception as e:
        logger.error(f"Error generating OP visualization: {e}")
        raise


def visualize_from_instance_data(instance_data: Dict[str, Any], idx: int = 0, 
                                save_path: Optional[str] = None) -> str:
    """
    从实例数据字典生成可视化图片
    
    Args:
        instance_data: Dictionary containing 'locs', 'env_name', and optionally 'demand' or 'prize'
        idx: Index (for logging purposes)
        save_path: Optional path to save the image file
    
    Returns:
        base64 encoded image string
    """
    env_name = instance_data.get('env_name', 'tsp').lower()
    
    # Create a mock TensorDict-like structure
    class MockTD:
        def __init__(self, data):
            self.data = data
        
        def __getitem__(self, key):
            import torch
            if key == 'locs':
                return torch.tensor(self.data['locs']).unsqueeze(0)
            elif key == 'demand':
                return torch.tensor(self.data['demand']).unsqueeze(0)
            elif key == 'prize':
                return torch.tensor(self.data['prize']).unsqueeze(0)
            elif key == 'capacity':
                return torch.tensor([self.data.get('capacity', 20)])
            elif key == 'max_length':
                return torch.tensor([self.data.get('max_length', 100.0)])
            else:
                return None
        
        def get(self, key, default=None):
            return self.__getitem__(key) if key in ['locs', 'demand', 'prize', 'capacity', 'max_length'] else default
    
    mock_td = MockTD(instance_data)
    
    if env_name == 'cvrp':
        return visualize_cvrp_from_td(mock_td, 0, save_path)
    elif env_name == 'op':
        return visualize_op_from_td(mock_td, 0, save_path)
    else:  # Default to TSP
        return visualize_tsp_from_td(mock_td, 0, save_path)

