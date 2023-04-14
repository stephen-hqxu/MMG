"""
Hyperparameter for the different part of the model.
"""

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

class DropoutSetting:
    # position embedding
    POSITION_DROPOUT: float = 0.15

    # full embedding
    FULL_DROPOUT: float = 0.18