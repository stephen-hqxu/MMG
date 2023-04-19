from Data.MidiPianoRoll import MidiPianoRoll

from Model.Component.Embedding import FullEmbedding, TimeStepExpansion
from Model.Component.Coder import CoderMask, Encoder, Decoder

from torch import Tensor
from torch.nn import Module, Flatten, Unflatten

class Transformer(Module):
    """
    The complete transformer.
    """

    def __init__(this):
        super().__init__()
        # embedding
        this.EncoderEmbedding: FullEmbedding = FullEmbedding()
        this.DecoderEmbedding: FullEmbedding = FullEmbedding()

        # coder
        this.EncoderBlock: Encoder = Encoder()
        this.DecoderBlock: Decoder = Decoder()

        # un-embedding
        this.Output: TimeStepExpansion = TimeStepExpansion()

        # data manipulation
        this.FeatureFlatten: Flatten = Flatten(1, 2)
        this.FeatureUnflatten: Unflatten = Unflatten(1, (-1, MidiPianoRoll.DIMENSION_PER_TIME_STEP))

    def toFeatureSequence(this, x: Tensor) -> Tensor:
        """
        @brief Flatten the feature axes (time window, note) to a linear feature sequence.

        @param x (batch, embedded feature, note, time window)
        @return (batch, sequence, embedded feature)
        """
        # change to (batch, time window, note, embedded feature)
        x = x.swapaxes(1, 3)
        # it's very important that when we reshape it back to 4D, the order of the axis remains the same as this
        return this.FeatureFlatten(x)
    
    def toFeatureMatrix(this, x: Tensor) -> Tensor:
        """
        @brief Split the linear feature sequence back to 2D feature matrix.

        @param x Same size as output of `toFeatureSequence(x)`.
        @return Same size as input of `toFeatureSequence(x)`.
        """
        # we knot the number of note (velocity and controller) is the same, so can recover time
        # sequence length is definitely divisible by the number of note
        x = this.FeatureUnflatten(x)
        return x.swapaxes(1, 3)
    
    def encode(this, source: Tensor, mask: CoderMask) -> Tensor:
        """
        Input: (batch, time step, note) in integer.
        Output: (batch, sequence, feature)
        """
        source = this.toFeatureSequence(this.EncoderEmbedding(source))
        return this.EncoderBlock(source, mask)
    
    def decode(this, target: Tensor, enc_output: Tensor, mask: CoderMask) -> Tensor:
        """
        Input: same shape as encoder input.
        Output: same as shape and size as target input.
        """
        target = this.toFeatureSequence(this.DecoderEmbedding(target))
        target = this.DecoderBlock(target, enc_output, mask)
        return this.Output(this.toFeatureMatrix(target))
    
    def forward(this, source: Tensor, target: Tensor, mask: CoderMask) -> Tensor:
        """
        Regarding shape, see encode and decode function.
        This function is used for training only.
        """
        enc: Tensor = this.encode(source, mask)
        return this.decode(target, enc, mask)