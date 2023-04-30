from Data.MidiPianoRoll import MidiPianoRoll
from Model.Setting import EmbeddingSetting, DropoutSetting

import torch
from torch import Tensor
from torch.nn import Module, Dropout, Embedding, Flatten, Sequential, Linear, Conv2d, ConvTranspose2d, Hardswish, Softmax, Mish, Hardtanh

import numpy as np

from typing import List, Tuple, TypeVar

L = TypeVar("L")
"""
@tparam L The layer type. Must be a convolutional layer type.
"""
def createTimeStepSequence(layer_t: L, encode: bool) -> Sequential:
    """
    @brief Create a sequence of layers to encode or decode time step.

    @param layer_t L
    @param encode True to calculate feature for encode, false for decode.
    @return A sequence of created time step layer sequence model.
    """
    remaining_feature: int = EmbeddingSetting.EMBEDDED_FEATURE_SIZE - EmbeddingSetting.NOTE_EMBEDDING_FEATURE_SIZE
    feature_increment: int = remaining_feature // len(EmbeddingSetting.TIME_EMBEDDING_LAYER_KERNEL)
    layer_feature: np.ndarray = np.arange(EmbeddingSetting.NOTE_EMBEDDING_FEATURE_SIZE, EmbeddingSetting.EMBEDDED_FEATURE_SIZE + 1, feature_increment)
    
    # reverse the order if we are doing decoding
    if not encode:
        layer_feature = np.flip(layer_feature)

    sequence: Sequential = Sequential()
    # basically do a folding on the layer feature, so the input size of the next layer is the output size of previous
    for i in range(layer_feature.size - 1):
        kernel: Tuple[int, int] = (1, EmbeddingSetting.TIME_EMBEDDING_LAYER_KERNEL[i])
        # the recent paper shows Swish function outperforms ReLU; H-Swish is cheaper to compute
        sequence.extend([layer_t(layer_feature[i], layer_feature[i + 1], kernel, kernel), Hardswish(True)])
    return sequence

class TimeStepEmbedding(Module):
    """
    @brief Time step dimensionality reduction and feature embedding.
    """

    def __init__(this):
        super().__init__()
        # split the input feature into different embedding vectors based on their categories
        def embedFeature() -> Embedding:
            # frequency of appearance of each note is not the same, it is definite some notes are used less than others, so scale gradient
            return Embedding(EmbeddingSetting.NOTE_ORIGINAL_FEATURE_SIZE, EmbeddingSetting.NOTE_EMBEDDING_FEATURE_SIZE, scale_grad_by_freq = True)
        this.VelocityEmbedding: Embedding = embedFeature()
        this.ControlEmbedding: Embedding = embedFeature()

        # dimensionality reduction of time
        this.TimeReduction: Sequential = createTimeStepSequence(Conv2d, True)

    def forward(this, x: Tensor) -> Tensor:
        """
        Input: (batch, time step, note). The input must be an integer tensor.
        Output: (batch, embedded feature, note, time window)
        """
        vel_emb: Tensor = this.VelocityEmbedding(x[:, :, MidiPianoRoll.sliceVelocity()])
        ctrl_emb: Tensor = this.ControlEmbedding(x[:, :, MidiPianoRoll.sliceControl()])
        # concatenate the embeddings in the original order of each feature to form a matrix
        # unlike the token-based model that connects all embedded vectors into one big vector
        # in the end, we will have a 2D embedded feature matrix
        x = torch.cat((vel_emb, ctrl_emb), dim = 2) # (batch, time step, note, feature)

        # now perform dimensionality reduction on time domain
        x = x.swapaxes(1, 3) # (batch, feature, note, time step)
        return this.TimeReduction(x)
    
class PositionEmbedding(Module):
    """
    @brief Note position embedding based on time window.
    """

    def __init__(this):
        super().__init__()
        this.Zeroing: Dropout = Dropout(p = DropoutSetting.POSITION_EMBEDDING)

        # generate position encoder, based on the position encoder from the original paper
        # TODO: it's possible to train a positional embedder if this implementation doesn't work well
        position: Tensor = torch.arange(EmbeddingSetting.MAX_SEQUENCE_LENGTH)
        normaliser: Tensor = torch.exp(torch.arange(0, EmbeddingSetting.EMBEDDED_FEATURE_SIZE, 2) * (-4 / EmbeddingSetting.EMBEDDED_FEATURE_SIZE)).unsqueeze(1)

        # batch and note axis will be broadcasted
        pe: Tensor = torch.zeros((1, EmbeddingSetting.EMBEDDED_FEATURE_SIZE, 1, EmbeddingSetting.MAX_SEQUENCE_LENGTH), dtype = torch.float32)
        pe[0, 0::2, 0, :] = torch.sin(position * normaliser)
        pe[0, 1::2, 0, :] = torch.cos(position * normaliser)

        this.register_buffer("PositionEncoder", pe)

    def forward(this, x: Tensor) -> Tensor:
        """
        Input: shape of output of time step embedding
        Output: same shape as this input
        """
        # we only need to embed temporal position
        return this.Zeroing(x + this.PositionEncoder[:, :, :, :x.size(3)])
    
class FullEmbedding(Module):
    """
    All embedding models put together.
    """

    def __init__(this):
        super().__init__()
        this.Zeroing: Dropout = Dropout(p = DropoutSetting.FULL_EMBEDDING)

        this.TimeStep: TimeStepEmbedding = TimeStepEmbedding()
        this.Position: PositionEmbedding = PositionEmbedding()

    def forward(this, x: Tensor) -> Tensor:
        """
        Input: shape of input of time step embedding.
        Output: shape of output of position embedding.
        """
        time_step_emb: Tensor = this.TimeStep(x)
        pos_emb: Tensor = this.Position(time_step_emb)
        return this.Zeroing(time_step_emb + pos_emb)
    
class TimeStepExpansion(Module):
    """
    Essentially an inverse of time step embedding to convert from embedded feature and time window back to time steps.
    """

    def __init__(this):
        super().__init__()
        # basically do everything as in time step embedding, but in reversed order
        this.TimeExpansion: Sequential = createTimeStepSequence(ConvTranspose2d, False)

        noteSummaryFeature: List[int] = [EmbeddingSetting.NOTE_ORIGINAL_FEATURE_SIZE] + EmbeddingSetting.NOTE_FEATURE_SUMMARY_LAYER_FEATURE + [1]
        summaryLayerCount: int = len(noteSummaryFeature) - 1
        # The output from the transformer is linear, so a single layer for each suffices.
        def lineariseFeatureEmbedding() -> Sequential:
            # add some layers to learn to summarise the probability of each note feature to a single note feature
            noteSummaryLayer: List[Linear] = list()
            for i in range(summaryLayerCount):
                noteSummaryLayer.append(Linear(noteSummaryFeature[i], noteSummaryFeature[i + 1]))
                # For hidden layers: The Mish function is just like ReLU, but it is self-regularised.
                # For output layer: There is a significance to use hard Tanh rather than Tanh.
                # We can see that Tanh is non-linear and has double-opened range (-1.0, 1.0),
                # this means there is a chance some feature values are never reached, or output with bias.
                noteSummaryLayer.append(Mish(True) if i < summaryLayerCount - 1 else Hardtanh(inplace = True))

            return Sequential(
                Linear(EmbeddingSetting.NOTE_EMBEDDING_FEATURE_SIZE, EmbeddingSetting.NOTE_ORIGINAL_FEATURE_SIZE),
                # get a probability of each original feature level, this is the velocity of control changes value at each matrix cell
                Softmax(3),
                *noteSummaryLayer,
                # remove feature axis of size one, to shape (batch, time step, note)
                Flatten(2, 3)
            )
        this.VelocityProjection: Sequential = lineariseFeatureEmbedding()
        this.ControlProjection: Sequential = lineariseFeatureEmbedding()

    def forward(this, x: Tensor) -> Tensor:
        """
        Input: shape of output of full embedding.
        Output: shape of input of time step embedding. The output is activated to signed normalised range.
        """
        # undo dimensionality reduction of time
        x = this.TimeExpansion(x)
        x = x.swapaxes(1, 3) # (batch, time step, note, feature)

        # undo note feature embedding
        vel: Tensor = this.VelocityProjection(x[:, :, MidiPianoRoll.sliceVelocity(), :])
        ctrl: Tensor = this.ControlProjection(x[:, :, MidiPianoRoll.sliceControl(), :])
        return torch.cat((vel, ctrl), dim = 2)