import torch
import numpy as np
from tensordict import TensorDict
from .env import TDTSPEnv

class TDTSPMatNetWrapper(TDTSPEnv):
    """
    Wrapper for TDTSP to work with MatNet.
    - Uses a global travel time matrix (registered buffer) to save memory.
    - Adds 'cost_matrix' to observations for MatNet.
    """
    name = "tdtsp_matnet"
    
    def __init__(self, matrix, duration, instances_list=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register matrix as buffer [N, N, M]
        if isinstance(matrix, np.ndarray):
            matrix = torch.from_numpy(matrix)
        self.register_buffer("global_matrix", matrix.long())
        self.duration = duration
        
        if instances_list is not None:
            self.generator = TDTSPFixedGenerator(instances_list)

    def _get_travel_time(self, td, prev_node_idx, current_node_idx, is_first_step=None, current_time=None):
        if current_time is None:
            current_time = td["current_time"]
            
        # 1. Get Physical Location Indices
        # td["locs_idx"] maps instance node idx -> matrix location idx
        locs_idx = td["locs_idx"]
        
        u_physical = locs_idx.gather(1, prev_node_idx.view(-1, 1)).squeeze(1)
        v_physical = locs_idx.gather(1, current_node_idx.view(-1, 1)).squeeze(1)
        
        # 2. Calculate Time Step s
        s = (current_time // self.duration).long()
        max_s = self.global_matrix.shape[-1] - 1
        s = s.clamp(max=max_s)
        
        # 3. Lookup in global matrix
        # matrix: [N_matrix, N_matrix, T_steps]
        travel_time = self.global_matrix[u_physical, v_physical, s]
            
        # Mask first step
        if is_first_step is not None:
            travel_time[is_first_step] = 0.0
            
        return travel_time.float()

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        # Recalculate reward (negative makespan)
        batch_size = actions.shape[0]
        device = actions.device
        
        if "start_time" in td.keys():
            current_time = td["start_time"]
        else:
            current_time = torch.zeros(batch_size, device=device)
            
        current_node_idx = actions[:, 0]
        
        for i in range(actions.shape[1] - 1):
            next_node_idx = actions[:, i+1]
            travel_time = self._get_travel_time(td, current_node_idx, next_node_idx, current_time=current_time)
            current_time = current_time + travel_time
            current_node_idx = next_node_idx
            
        # Return to start
        travel_time = self._get_travel_time(td, current_node_idx, actions[:, 0], current_time=current_time)
        current_time = current_time + travel_time
        
        return -current_time

    def _reset(self, td=None, batch_size=None):
        # td contains locs_idx and start_time from DataLoader/Generator
        device = td.device
        
        locs_idx = td["locs_idx"] # [batch, N]
        batch_sz, n_nodes = locs_idx.shape
        
        # Initialize dynamic state
        current_node = torch.zeros((batch_sz,), dtype=torch.int64, device=device)
        
        if "start_time" in td.keys():
            current_time = td["start_time"]
        else:
            current_time = torch.zeros((batch_sz,), dtype=torch.float32, device=device)
            
        i = torch.zeros((batch_sz,), dtype=torch.int64, device=device)
        available = torch.ones((batch_sz, n_nodes), dtype=torch.bool, device=device)
        
        # Calculate cost_matrix for MatNet
        # Extract submatrix for each instance: [batch, N, N, M]
        # Then mean over M -> [batch, N, N]
        
        rows = locs_idx.unsqueeze(2).expand(batch_sz, n_nodes, n_nodes)
        cols = locs_idx.unsqueeze(1).expand(batch_sz, n_nodes, n_nodes)
        
        # Advanced indexing to get submatrix
        submatrix = self.global_matrix[rows, cols] # [B, N, N, M]
        cost_matrix = submatrix.float().mean(dim=-1) # [B, N, N]
        
        return TensorDict(
            {
                "locs_idx": locs_idx,
                "cost_matrix": cost_matrix, # For MatNet
                "first_node": current_node,
                "current_node": current_node,
                "current_time": current_time,
                "i": i,
                "action_mask": available,
                "locs": torch.zeros((batch_sz, n_nodes, 2), device=device) # Dummy
            },
            batch_size=batch_size,
        )

class TDTSPFixedGenerator:
    """Generator that samples from a fixed list of instances."""
    def __init__(self, instances):
        self.instances = instances
        
    def __call__(self, batch_size):
        if isinstance(batch_size, (list, tuple)):
            batch_size = batch_size[0]
            
        indices = np.random.choice(len(self.instances), size=batch_size)
        batch = [self.instances[i] for i in indices]
        
        locs_idx = torch.stack([torch.from_numpy(b["locs_idx"]).long() for b in batch])
        start_time = torch.tensor([b["start_time"] for b in batch], dtype=torch.float32)
        
        return TensorDict({
            "locs_idx": locs_idx,
            "start_time": start_time,
            "locs": torch.zeros((batch_size, locs_idx.shape[1], 2), dtype=torch.float32)
        }, batch_size=batch_size)
