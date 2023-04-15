TIME_WINDOW_SIZE: int = 1024
"""
The number of time step to be grouped together into one token, for dimensionality reduction.
"""

EMBEDDED_FEATURE_SIZE: int = 256
"""
The number of element in the feature vector after embedding an input feature.
"""
MAX_SEQUENCE_LENGTH: int = 8192
"""
The maximum number of token (grouped time step) the model can take.
"""

ATTENTION_HEAD_COUNT: int = 4
"""
The number of head used in multi-head attention model, by splitting the embedded feature.
"""

FEED_FORWARD_LATENT_SIZE: int = 2048
"""
The hidden layer dimension of the feed forward MLP.
"""

class DropoutSetting:
    """
    Dropout probability.
    """
    POSITION_EMBEDDING: float = 0.15
    FULL_EMBEDDING: float = 0.18

    MULTIHEAD_ATTENTION: float = 0.0

    RESIDUAL: float = 0.11
    FEED_FORWARD: float = 0.09