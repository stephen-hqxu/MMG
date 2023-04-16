class EmbeddingSetting:
    TIME_WINDOW_SIZE: int = 1024
    """
    The number of time step to be grouped together into one feature, for dimensionality reduction.
    """
    EMBEDDED_FEATURE_SIZE: int = 256
    """
    The number of element in the feature vector after embedding an input feature.
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
    TIME_FEATURE_START_EXPONENT: int = 7
    """
    The size of the first feature layer will be 2^x, and grows exponentially.
    """
    TIME_KERNEL: int = 8
    """
    Stride is half of the kernel, padding is half of the stride.
    """
    TIME_LAYER: int = 3
    """
    The number of convolutional layer to extract time information.
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