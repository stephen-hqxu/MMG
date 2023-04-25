from Data.MidiPianoRoll import MidiPianoRoll

import os
from typing import List, Tuple

PROJECT_ROOT: str = os.getcwd()
TIME_WINDOW_ALLOCATION_INCREMENT: int = 100

class SpecialTokenSetting:
    SOS: int = MidiPianoRoll.NOTE_MAX_LEVEL
    EOS: int = SOS + 1
    PAD: int = EOS + 1

class EmbeddingSetting:
    NOTE_ORIGINAL_FEATURE_SIZE: int = MidiPianoRoll.NOTE_MAX_LEVEL + 3
    NOTE_EMBEDDING_FEATURE_SIZE: int = 128
    TIME_EMBEDDING_LAYER_KERNEL: List[int] = [32, 32]

    TIME_WINDOW_SIZE: int = 1024
    EMBEDDED_FEATURE_SIZE: int = 256
    MAX_SEQUENCE_LENGTH: int = 8192

class TransformerSetting:
    ATTENTION_HEAD_COUNT: int = 4
    FEED_FORWARD_LATENT_SIZE: int = 2048
    CODER_LAYER_COUNT: int = 6
    CAUSAL_ATTENTION_MASK: bool = False

class DiscriminatorSetting:
    TIME_KERNEL_SIZE: List[int] = [8, 8, 8, 16]
    TIME_LAYER_FEATURE: List[int] = [256, 384, 512]

    SEQUENCE_HIDDEN: int = 256
    SEQUENCE_LAYER: int = 2

    LEAKY_SLOPE: float = 0.17

class DropoutSetting:
    POSITION_EMBEDDING: float = 0.15
    FULL_EMBEDDING: float = 0.18

    CODER: float = 0.12

    DISCRIMINATOR_SEQUENCE: float = 0.03

class DatasetSetting:
    DATA_SHUFFLE_SEED: int = 6666
    DATA_SPLIT: List[float] = [0.7, 0.2, 0.1]

    ASAP_PATH: str = "/home/stephen/shared-drives/V\\:/year4/cs407/dataset/asap-dataset-2021-09-16"

    MIDI_CACHE_PATH: str = PROJECT_ROOT + "/cache"
    MODEL_OUTPUT_PATH: str = PROJECT_ROOT + "/output"
    TRAIN_STATS_LOG_PATH: str = PROJECT_ROOT + "/train-stats"

class TrainingSetting:
    EPOCH: int = 500
    BATCH_SIZE: int = 4

    LR_GENERATOR: float = 5e-4
    LR_DISCRIMINATOR: float = 2e-4
    BETA_GENERATOR: Tuple[float, float] = (0.5, 0.98)
    BETA_DISCRIMINATOR: Tuple[float, float] = (0.5, 0.995)

    LOG_FREQUENCY: int = 10