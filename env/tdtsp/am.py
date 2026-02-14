from rl4co.models.zoo.am import AttentionModel
from rl4co.models.zoo.am.policy import AttentionModelPolicy
from .embeddings import TDTSPInitEmbedding, TDTSPContext

class TDTSPPolicy(AttentionModelPolicy):
    """
    Attention Model Policy for TDTSP.
    Uses TDTSPInitEmbedding and TDTSPContext.
    """
    def __init__(self, embed_dim=128, **kwargs):
        # Enforce TDTSP embeddings
        # We instantiate them here because they depend on embed_dim
        if "init_embedding" not in kwargs:
            kwargs["init_embedding"] = TDTSPInitEmbedding(embed_dim)
        
        if "context_embedding" not in kwargs:
            kwargs["context_embedding"] = TDTSPContext(embed_dim)
            
        super().__init__(embed_dim=embed_dim, env_name="tdtsp", **kwargs)

class TDTSPModel(AttentionModel):
    """
    Attention Model for TDTSP.
    Wraps TDTSPPolicy.
    """
    def __init__(self, env, policy=None, baseline="rollout", policy_kwargs={}, **kwargs):
        if policy is None:
            # If policy is not provided, we create TDTSPPolicy
            policy = TDTSPPolicy(env_name=env.name, **policy_kwargs)
            
        super().__init__(env, policy, baseline, **kwargs)
