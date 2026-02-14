import torch
import torch.nn as nn
from rl4co.models.nn.env_embeddings.context import EnvContext
from rl4co.models.nn.env_embeddings.init import TSPInitEmbedding

class TDTSPInitEmbedding(nn.Module):
    """
    Initial embedding for TDTSP.
    Supports both speed-parameter based and matrix-based environments.
    
    Embeds:
        - locs: x, y coordinates [Batch, N, 2]
        - matrix-based features (if travel_time_matrix is present):
            - row_feats: learned projection of travel times from each node [Batch, N, d]
            - col_feats: learned projection of travel times to each node [Batch, N, d]
            - time_windows: [Batch, N, 2] (optional)
        - speed-parameter based features (if travel_time_matrix is not present):
            - base_speed, amplitude, period, phase [Batch, 4]
    """
    def __init__(self, embed_dim, linear_bias=True, num_locs=21, num_time_steps=37, matrix_feat_dim=1):
        super(TDTSPInitEmbedding, self).__init__()
        # We use a fixed input dim of 6 for both cases to maintain consistency
        # Case 1 (Matrix): 2 (locs) + 2 (tw) + matrix_feat_dim (row) + matrix_feat_dim (col) = 6 (if dim=1)
        # Case 2 (Speed): 2 (locs) + 4 (speed params) = 6
        self.matrix_feat_dim = matrix_feat_dim
        self.num_locs = num_locs
        self.num_time_steps = num_time_steps
        
        # Matrix feature projections
        # For each node, we project its travel times to/from all other nodes across all time steps
        self.row_proj = nn.Linear(num_locs * num_time_steps, matrix_feat_dim, bias=linear_bias)
        self.col_proj = nn.Linear(num_locs * num_time_steps, matrix_feat_dim, bias=linear_bias)
        
        input_dim = 2 + 2 + 2 * matrix_feat_dim
        # Note: we assume speed mode also fits into this or we handle it separately
        # For now, we keep it at 6 for backward compatibility
        self.init_embed = nn.Linear(max(6, input_dim), embed_dim, linear_bias)

    def forward(self, td):
        locs = td["locs"] # [Batch, N, 2]
        batch_size, num_locs, _ = locs.shape

        if "travel_time_matrix" in td.keys():
            # Matrix-based environment (TDTSPMatrixEnv or TDTSPTWEnv)
            matrix = td["travel_time_matrix"] # [Batch, N, N, T]
            
            # Normalize matrix for numerical stability (e.g., divide by 3600 to convert to hours)
            # This helps nn.Linear layers handle the input better
            matrix = matrix / 3600.0
            
            # 1. Row features (from node i to all others)
            # Flatten N and T dimensions for each node i
            
            row_feats = self.row_proj(matrix.reshape(batch_size, num_locs, -1)) # [Batch, N, d]
            
            # 2. Col features (to node i from all others)
            # Transpose to get [Batch, N_target, N_source, T], then reshape to see [Batch, N_target, N_source * T]
            col_feats = self.col_proj(matrix.transpose(1, 2).reshape(batch_size, num_locs, -1)) # [Batch, N, d]
            
            # 3. Time Windows (if available)
            if "time_windows" in td.keys():
                # Normalize time windows too
                tw = td["time_windows"] / 3600.0 # [Batch, N, 2]
            else:
                # Placeholder for time windows if not present
                tw = torch.zeros((batch_size, num_locs, 2), device=locs.device)
            
            # Concatenate: [Batch, N, 2 + 2 + d + d]
            feats = torch.cat([locs, tw, row_feats, col_feats], dim=-1)
            
        else:
            # Speed-parameter based environment (Original TDTSPEnv)
            # Gather speed params [Batch, 1]
            # These are already relatively small (base_speed ~ 10, amp ~ 5, etc.)
            base_speed = td["base_speed"]
            amp = td["speed_amplitude"]
            period = td["period"] / 3600.0 # Normalize period
            phase = td["phase"]
            
            # [Batch, 4]
            speed_params = torch.cat([base_speed, amp, period, phase], dim=-1)
            
            # Expand to [Batch, N, 4]
            speed_params_expanded = speed_params.unsqueeze(1).expand(-1, num_locs, -1)
            
            # Concatenate: [Batch, N, 2 + 4] = [Batch, N, 6]
            feats = torch.cat([locs, speed_params_expanded], dim=-1)
        
        return self.init_embed(feats)

class TDTSPContext(EnvContext):
    """
    Context embedding for TDTSP.
    Projects:
        - current node embedding
        - first node embedding (standard TSP context)
        - current time (TDTSP specific)
    """
    def __init__(self, embed_dim):
        # Standard TSP context has dim = 2 * embed_dim (first + current node)
        # We add 1 for current_time
        super(TDTSPContext, self).__init__(embed_dim, step_context_dim=2 * embed_dim + 1)
        self.W_placeholder = nn.Parameter(torch.Tensor(2 * self.embed_dim + 1).uniform_(-1, 1))

    def forward(self, embeddings, td):
        batch_size = embeddings.size(0)
        
        # Handle multi-start decoding where td is [Batch, Num_Starts] but embeddings is [Batch, N, D]
        is_multistart = len(td.batch_size) > 1
        if is_multistart:
            num_starts = td.batch_size[1]
            # Expand embeddings: [B, N, D] -> [B*S, N, D]
            embeddings = embeddings.repeat_interleave(num_starts, dim=0)
            
            # Flatten relevant td fields
            first_node = td["first_node"].reshape(-1)
            current_node = td["current_node"].reshape(-1)
            current_time = td["current_time"].reshape(-1)
            
            batch_size = batch_size * num_starts
        else:
            first_node = td["first_node"]
            current_node = td["current_node"]
            current_time = td["current_time"]
            
        # 1. Get First Node and Current Node Embeddings (Standard TSP Logic)
        step = td["i"]
        if step.dim() > 0:
            step = step.flatten()[0]

        if step < 1:  # First step (i=0)
            # Use placeholder context
            context_embedding = self.W_placeholder[None, :].expand(
                batch_size, self.W_placeholder.size(-1)
            )
        else:
            # Gather embeddings
            # [Batch, 2, embed_dim]
            # Note: embeddings is [Batch, N, embed_dim]
            from rl4co.utils.ops import gather_by_index
            
            # Stack indices [Batch, 2]
            indices = torch.stack([first_node, current_node], -1)
            
            # Gather: [Batch, 2, embed_dim]
            node_context = gather_by_index(embeddings, indices)
            
            # Flatten: [Batch, 2 * embed_dim]
            node_context = node_context.view(batch_size, -1)
            
            # 2. Get Current Time (Normalize to hours for stability)
            current_time = current_time / 3600.0 # [Batch] or [Batch, 1]
            if current_time.dim() == 1:
                current_time = current_time.unsqueeze(-1)
            
            # Concatenate: [Batch, 2*embed_dim + 1]
            context_embedding = torch.cat([node_context, current_time], dim=-1)
            
        out = self.project_context(context_embedding)
        
        if is_multistart:
            out = out.view(td.batch_size[0], td.batch_size[1], -1)
            
        return out
