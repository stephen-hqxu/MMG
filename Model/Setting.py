from Data.MidiPianoRoll import MidiPianoRoll

import os
from typing import List, Tuple

PROJECT_ROOT: str = os.getcwd()
"""
We assume the application always starts at the project root.
"""

TIME_WINDOW_ALLOCATION_INCREMENT: int = 100
"""
Specifies, when need to preallocate memory for time steps, grow the array by multiple of time window size.
Set to a lager number reduces the number of allocation, but may increase memory consumption.
"""

class SpecialTokenSetting:
    SOS: int = MidiPianoRoll.NOTE_MAX_LEVEL
    """
    Start Of Sequence
    """
    EOS: int = SOS + 1
    """
    End Of Sequence
    """
    PAD: int = EOS + 1
    """
    Padding
    """

class EmbeddingSetting:
    NOTE_ORIGINAL_FEATURE_SIZE: int = MidiPianoRoll.NOTE_MAX_LEVEL + 3
    """
    The feature size before embedding note.
    128 velocity/control value + all special tokens.
    """
    NOTE_EMBEDDING_FEATURE_SIZE: int = 128
    """
    The feature size after embedding note.
    """
    TIME_EMBEDDING_LAYER_KERNEL: List[int] = [32, 32]
    """
    The kernel size of each layer to embed time; the product must equal to the time window size.
    """

    TIME_WINDOW_SIZE: int = 1024
    """
    The number of time step to be grouped together into one feature, for dimensionality reduction.
    """
    EMBEDDED_FEATURE_SIZE: int = 256
    """
    The number of element in the feature vector after embedding an input feature, including both note and time.
    """
    MAX_SEQUENCE_LENGTH: int = 8192
    """
    The maximum number of time window (grouped time step) the model can take.
    """

class TransformerSetting:
    ATTENTION_HEAD_COUNT: int = 4
    """
    The number of head used in multi-head attention model, by splitting the embedded feature.
    """
    FEED_FORWARD_LATENT_SIZE: int = 2048
    """
    The hidden layer dimension of the feed forward MLP.
    """
    CODER_LAYER_COUNT: int = 6
    """
    The number of hidden layers for encoder and decoder.
    """

class DiscriminatorSetting:
    TIME_KERNEL_SIZE: List[int] = [8, 8, 8, 16]
    """
    The kernel size of each layer. Must be at least 2 sizes specified.

    Stride is half of the kernel, padding is half of the stride;
    except the last layer, which will have full stride and no padding.

    The input will be an integer multiple of the time window size, and output should be one.
    Consult PyTorch documentation to calculate the size of output of each layer.
    """
    TIME_LAYER_FEATURE: List[int] = [256, 384, 512]
    """
    The number of convolutional layer to extract time information.
    Must have one less member than the number of element.
    The input and output layers always have feature size of one.
    """

    SEQUENCE_HIDDEN: int = 256
    """
    The hidden layer dimension for the sequence extraction network.
    """
    SEQUENCE_LAYER: int = 2
    """
    The number of layer of the sequence network.
    """

    LEAKY_SLOPE: float = 0.17

class DropoutSetting:
    POSITION_EMBEDDING: float = 0.15
    FULL_EMBEDDING: float = 0.18

    CODER: float = 0.12

    DISCRIMINATOR_SEQUENCE: float = 0.03

class DatasetSetting:
    """
    Please change the path variables so that it matches your environment.
    As a convention to avoid confusion, please alway use absolute path.
    """
    DATA_SHUFFLE_SEED: int = 6666
    """
    Seed used for randomly permuting the dataset.
    """
    DATA_SPLIT: List[float] = [0.7, 0.2, 0.1]
    """
    Proportion of [train, validation, test]; must sum up to 1.0.
    """

    ASAP_PATH: str = "/home/stephen/shared-drives/V\\:/year4/cs407/dataset/asap-dataset-2021-09-16"

    MIDI_CACHE_PATH: str = PROJECT_ROOT + "/cache"
    """
    Intermediate MIDI file cache output directory.
    """
    MODEL_OUTPUT_PATH: str = PROJECT_ROOT + "/output"
    """
    Hold binary of the trained model.
    """
    TRAIN_STATS_LOG_PATH: str = PROJECT_ROOT + "/train-stats"
    """
    Store training stats such as training accuracy and loss.
    """

class TrainingSetting:
    EPOCH: int = 500
    """
    The number of epoch.
    """
    BATCH_SIZE: int = 4
    """
    The size of batch.
    """

    LEARNING_RATE: float = 2e-4
    """
    The learning rate for the optimiser.
    """
    BETA: Tuple[float, float] = (0.5, 0.98)
    """
    The beta parameter for the optimiser.
    """

    LOG_FREQUENCY: int = 10
    """
    Specify logging frequency in term of number of iteration elapsed.
    """