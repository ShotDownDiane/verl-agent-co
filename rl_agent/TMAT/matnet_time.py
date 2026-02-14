
import math
from typing import Optional
import torch
import torch.nn as nn
from tensordict import TensorDict

from rl4co.models.zoo.matnet.encoder import MatNetEncoder, MatNetCrossMHA
from rl4co.models.zoo.matnet.decoder import MatNetDecoder
from rl4co.models.zoo.matnet.policy import MatNetPolicy
from rl4co.models.nn.ops import PositionalEncoding, TransformerFFN
from rl4co.models.nn.env_embeddings import env_init_embedding
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class SinusoidalTimeEncoding(nn.Module):
    def __init__(self, embed_dim: int, max_len: int = 10000):
        super().__init__()
        self.embed_dim = embed_dim
        # Precompute constants
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        self.register_buffer('div_term', div_term)

    def forward(self, time):
        """
        Args:
            time: [Batch, N] or [Batch, 1] or [Batch] float tensor of current times
        Returns:
            [Batch, N, embed_dim]
        """
        if time.dim() == 1:
            time = time.unsqueeze(1)
            
        # time: [B, N]
        # div_term: [embed_dim/2]
        
        # [B, N, embed_dim/2]
        scaled_time = time.unsqueeze(-1) * self.div_term
        
        pe = torch.zeros(*time.shape, self.embed_dim, device=time.device)
        pe[..., 0::2] = torch.sin(scaled_time)
        pe[..., 1::2] = torch.cos(scaled_time)
        return pe

class MatNetTimeInitEmbedding(nn.Module):
    """
    Preparing the initial row and column embeddings for MatNetTime.
    Handles 4D cost matrix [B, N, N, T] and Node Features [B, N, D].
    """
    def __init__(self, embed_dim: int, mode: str = "RandomOneHot") -> None:
        super().__init__()
        self.embed_dim = embed_dim
        assert mode in {"RandomOneHot", "Random"}, "mode must be one of ['RandomOneHot', 'Random']"
        self.mode = mode
        
        # Projections for Node Features
        self.loc_proj = nn.Linear(2, embed_dim)
        self.tw_encoder = SinusoidalTimeEncoding(embed_dim)

    def forward(self, td: TensorDict):
        # Expected shape: [B, N, N, T]
        dmat = td["travel_time_matrix"] 
        
        # Normalize dmat to [0, 1] range for numerical stability
        if dmat.numel() > 0:
            dmat_max = dmat.max()
            if dmat_max > 0:
                dmat = dmat / dmat_max

        # Handle 3D case (static) just in case
        if dmat.dim() == 3:
            b, r, c = dmat.shape
        elif dmat.dim() == 4:
            b, r, c, t = dmat.shape
        else:
            raise ValueError(f"Unexpected dmat shape: {dmat.shape}")

        row_emb = torch.zeros(b, r, self.embed_dim, device=dmat.device)

        if self.mode == "RandomOneHot":
            col_emb = torch.zeros(b, c, self.embed_dim, device=dmat.device)
            rand = torch.rand(b, c)
            rand_idx = rand.argsort(dim=1)
            b_idx = torch.arange(b)[:, None].expand(b, c)
            n_idx = torch.arange(c)[None, :].expand(b, c)
            col_emb[b_idx, n_idx, rand_idx] = 1.0
        elif self.mode == "Random":
            col_emb = torch.rand(b, c, self.embed_dim, device=dmat.device)
        else:
            raise NotImplementedError

        # --- Add Node Features (Locs + TW) ---
        if "locs" in td and "time_windows" in td:
            locs = td["locs"] # [B, N, 2]
            tws = td["time_windows"] # [B, N, 2]
            
            # Project Locs
            loc_emb = self.loc_proj(locs)
            
            # Encode TWs
            tw_start_emb = self.tw_encoder(tws[..., 0])
            tw_end_emb = self.tw_encoder(tws[..., 1])
            
            node_feat = loc_emb + tw_start_emb + tw_end_emb
            
            # Add to Row and Col Embeddings
            # row_emb: [B, N, D]
            # col_emb: [B, N, D]
            row_emb = row_emb + node_feat
            col_emb = col_emb + node_feat

        return row_emb, col_emb, dmat

class MatNetTimeMHA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, bias: bool = False):
        super().__init__()
        self.row_encoding_block = MatNetCrossMHA(embed_dim, num_heads, bias)
        self.col_encoding_block = MatNetCrossMHA(embed_dim, num_heads, bias)

    def forward(self, row_emb, col_emb, dmat, attn_mask=None):
        updated_row_emb = self.row_encoding_block(
            row_emb, col_emb, dmat=dmat, cross_attn_mask=attn_mask
        )
        attn_mask_t = attn_mask.transpose(-2, -1) if attn_mask is not None else None
        
        # Check if dmat is 4D (Time-dependent) [B, M, N, S]
        if dmat.dim() == 4:
            # We want to transpose M and N, which are dims 1 and 2
            # transpose(-3, -2) swaps M and N
            dmat_t = dmat.transpose(-3, -2)
        else:
            # Standard 3D [B, M, N]
            dmat_t = dmat.transpose(-2, -1)
            
        updated_col_emb = self.col_encoding_block(
            col_emb,
            row_emb,
            dmat=dmat_t,
            cross_attn_mask=attn_mask_t,
        )
        return updated_row_emb, updated_col_emb

class MatNetTimeLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = False,
        feedforward_hidden: int = 512,
        normalization: Optional[str] = "instance",
    ):
        super().__init__()
        self.MHA = MatNetTimeMHA(embed_dim, num_heads, bias)
        self.F_a = TransformerFFN(embed_dim, feedforward_hidden, normalization)
        self.F_b = TransformerFFN(embed_dim, feedforward_hidden, normalization)

    def forward(self, row_emb, col_emb, dmat, attn_mask=None):
        row_emb_out, col_emb_out = self.MHA(row_emb, col_emb, dmat, attn_mask)
        row_emb_out = self.F_a(row_emb_out, row_emb)
        col_emb_out = self.F_b(col_emb_out, col_emb)
        return row_emb_out, col_emb_out

class MatNetTimeEncoder(MatNetEncoder):
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 16,
        num_layers: int = 3,
        normalization: str = "batch",
        feedforward_hidden: int = 512,
        init_embedding: nn.Module = None,
        init_embedding_kwargs: dict = {},
        bias: bool = False,
        mask_non_neighbors: bool = False,
    ):
        # We call super init but we will overwrite self.layers
        super().__init__(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            normalization=normalization,
            feedforward_hidden=feedforward_hidden,
            init_embedding=init_embedding,
            init_embedding_kwargs=init_embedding_kwargs,
            bias=bias,
            mask_non_neighbors=mask_non_neighbors
        )

        # Overwrite layers with MatNetTimeLayer
        self.layers = nn.ModuleList(
            [
                MatNetTimeLayer(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    bias=bias,
                    feedforward_hidden=feedforward_hidden,
                    normalization=normalization,
                )
                for _ in range(num_layers)
            ]
        )
    
    def forward(self, td, attn_mask: torch.Tensor = None):
        row_emb, col_emb, dmat = self.init_embedding(td)
        
        # Debug prints
        # print(f"dmat stats: min={dmat.min()}, max={dmat.max()}, mean={dmat.mean()}, shape={dmat.shape}")
        if torch.isnan(dmat).any():
             raise ValueError("dmat has NaNs in MatNetTimeEncoder!")
        
        if self.mask_non_neighbors and attn_mask is None:
            # Check dmat dim for 4D case
            if dmat.dim() == 4:
                 # Collapse time dim for mask. Assuming dmat > 0 implies neighbor.
                 attn_mask = dmat.sum(dim=-1).ne(0)
            else:
                 attn_mask = dmat.ne(0)

        if torch.isnan(row_emb).any() or torch.isnan(col_emb).any():
             raise ValueError("NaNs in initial embeddings!")

        for layer in self.layers:
            row_emb, col_emb = layer(row_emb, col_emb, dmat, attn_mask)
            if torch.isnan(row_emb).any() or torch.isnan(col_emb).any():
                raise ValueError("NaNs after MatNetTimeLayer!")

        embedding = (row_emb, col_emb)
        init_embedding = None
        return embedding, init_embedding

class MatNetTimeDecoder(MatNetDecoder):
    def __init__(self, env_name, embed_dim, num_heads, use_graph_context=True, **kwargs):
        super().__init__(
            env_name=env_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_graph_context=use_graph_context,
            **kwargs
        )
        self.time_encoding = SinusoidalTimeEncoding(embed_dim)
        
    def _compute_q(self, cached, td: TensorDict):
        # Get standard query (Context + Graph Context)
        # glimpse_q: [B, 1, embed_dim]
        glimpse_q = super()._compute_q(cached, td)
        
        # Add Time Encoding
        # td["current_time"] is usually [B] or [B, 1]
        current_time = td["current_time"]
        
        # We might need to scale time if it's in seconds (large values)
        # But standard Sinusoidal handles large values reasonably well if the frequency matches.
        # Alternatively, map to matrix step index if available.
        # For now, we pass raw time.
        
        time_emb = self.time_encoding(current_time)
        
        # Add to query
        glimpse_q = glimpse_q + time_emb
        
        return glimpse_q

class MatNetTimePolicy(MatNetPolicy):
    def __init__(
        self,
        env_name: str = "tdtsp",
        embed_dim: int = 256,
        num_encoder_layers: int = 5,
        num_heads: int = 16,
        normalization: str = "instance",
        init_embedding_kwargs: dict = {"mode": "RandomOneHot"},
        use_graph_context: bool = False,
        bias: bool = False,
        num_matrix_steps: int = 37, # Default for T
        **kwargs,
    ):
        # 1. Custom Encoder with Time-Aware Init
        encoder = MatNetTimeEncoder(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_encoder_layers,
            normalization=normalization,
            init_embedding_kwargs=init_embedding_kwargs, # We will override init_embedding instance
            bias=bias,
        )
        # Override init_embedding with our custom one
        encoder.init_embedding = MatNetTimeInitEmbedding(
            embed_dim=embed_dim, 
            **init_embedding_kwargs
        )
        
        # 2. Custom Decoder with Time Encoding
        decoder = MatNetTimeDecoder(
            env_name=env_name,
            embed_dim=embed_dim,
            num_heads=num_heads,
            use_graph_context=use_graph_context,
        )

        # Initialize Parent
        # We bypass MatNetPolicy.__init__ logic slightly by passing our instances
        # But MatNetPolicy.__init__ reconstructs them. 
        # So we better inherit AutoregressivePolicy directly or super() carefully.
        # MatNetPolicy constructor creates encoder/decoder internally.
        # So we should call super(MatNetPolicy, self).__init__ ... 
        # i.e. call AutoregressivePolicy's init directly.
        
        super(MatNetPolicy, self).__init__(
            env_name=env_name,
            encoder=encoder,
            decoder=decoder,
            embed_dim=embed_dim,
            num_encoder_layers=num_encoder_layers,
            num_heads=num_heads,
            normalization=normalization,
            **kwargs,
        )
        
        # Important: Set num_scores in MixedScoresSDPA layers
        # The MatNetEncoder creates MatNetLayers, which create MatNetMHA, which create MixedScoresSDPA.
        # MixedScoresSDPA has num_scores=1 by default.
        # We need to update it to num_matrix_steps.
        self._update_num_scores(encoder, num_matrix_steps)

    def _update_num_scores(self, module, num_scores):
        for name, child in module.named_children():
            if "MixedScoresSDPA" in child.__class__.__name__:
                child.num_scores = num_scores
                # Also need to resize the mix_W1 parameter if it was initialized with num_scores=1
                # mix_W1 shape: (num_heads, num_scores + 1, mixer_hidden_dim)
                if child.mix_W1.shape[1] != num_scores + 1:
                    # Re-initialize parameters for new num_scores
                    # Note: This resets weights, so do this before loading state dict
                    child.mix_W1 = nn.Parameter(torch.empty(
                        child.num_heads, num_scores + 1, child.mix_W1.shape[2], device=child.mix_W1.device
                    ))
                    child.mix_b1 = nn.Parameter(torch.empty(
                         child.num_heads, child.mix_b1.shape[1], device=child.mix_b1.device
                    )) # b1 doesn't depend on num_scores? 
                    # mix_b1: (num_heads, mixer_hidden_dim) - Correct.
                    
                    # Initialize W1
                    mix1_init = (1 / 2) ** (1 / 2)
                    nn.init.uniform_(child.mix_W1, -mix1_init, mix1_init)
                    # b1 is fine
                    
            else:
                self._update_num_scores(child, num_scores)
