"""
Hyperparameter for the different part of the model.
"""

from dataclasses import dataclass

TIME_WINDOW_SIZE: int = 10240
"""
The number of time step per run.
"""
TIME_WINDOW_ADVANCEMENT: int = 512
"""
The number of time step to move forward for every time window.
"""
FEATURE_SIZE: int = 256
"""
The number of the element in the feature vector after embedding.
"""

@dataclass
class EmbeddingSetting:
    # time step embedding #
    DICTIONARY_VELOCITY: int = 128
    DICTIONARY_CONTROL: int = 128

    # position embedding #
    POSITION_DROPOUT: float = 0.15

    # full embedding #
    FULL_DROPOUT: float = 0.18