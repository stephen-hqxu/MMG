from typing import List

class EmbeddingSetting:
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