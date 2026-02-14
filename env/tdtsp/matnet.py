from rl4co.models.zoo.matnet import MatNet
from rl4co.models.zoo.matnet.policy import MatNetPolicy
from .env_matnet import TDTSPMatNetWrapper

class TDTSPMatNetPolicy(MatNetPolicy):
    """
    MatNet Policy for TDTSP.
    Wrapper around MatNetPolicy forcing env_name='atsp' as MatNet treats TDTSP as asymmetric TSP
    via the matrix formulation.
    """
    def __init__(self, env_name="atsp", **kwargs):
        # MatNet works on distance matrices (ATSP)
        super().__init__(env_name=env_name, **kwargs)

class TDTSPMatNetModel(MatNet):
    """
    MatNet Model for TDTSP.
    Wraps TDTSPMatNetPolicy.
    """
    def __init__(self, env, policy=None, baseline="rollout", policy_kwargs={}, **kwargs):
        if policy is None:
            # Force env_name to atsp if not specified, though policy defaults it
            policy = TDTSPMatNetPolicy(**policy_kwargs)
            
        super().__init__(env, policy, baseline, **kwargs)
