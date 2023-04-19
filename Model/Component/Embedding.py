from Data.MidiPianoRoll import MidiPianoRoll
from Model.Setting import EmbeddingSetting, DropoutSetting

import torch
from torch import Tensor
from torch.nn import Module, Dropout, Embedding, Sequential, Linear, Conv2d, ConvTranspose2d, Hardswish, Tanh

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
        # frequency of appearance of each note is not the same, it is definite some notes are used less than others, so scale gradient
        embedding_param = { "num_embeddings" : EmbeddingSetting.NOTE_ORIGINAL_FEATURE_SIZE,
            "embedding_dim" : EmbeddingSetting.NOTE_EMBEDDING_FEATURE_SIZE, "scale_grad_by_freq" : True }
        this.VelocityEmbedding: Embedding = Embedding(**embedding_param)
        this.ControlEmbedding: Embedding = Embedding(**embedding_param)

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

        pe: Tensor = torch.zeros((EmbeddingSetting.EMBEDDED_FEATURE_SIZE, 1, EmbeddingSetting.MAX_SEQUENCE_LENGTH), dtype = torch.float32)
        pe[0::2, 0, :] = torch.sin(position * normaliser)
        pe[1::2, 0, :] = torch.cos(position * normaliser)

        this.register_buffer("PositionEncoder", pe)

    def forward(this, x: Tensor) -> Tensor:
        """
        Input: shape of output of time step embedding
        Output: same shape as this input
        """
        # we only need to embed temporal position
        for bat in range(x.size(0)):
            x[bat] += this.PositionEncoder[:, :, :x.size(3)]
        return this.Zeroing(x)
    
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

        # The output from the transformer is linear, so a single layer for each suffices.
        expansion_param = { "in_features" : EmbeddingSetting.NOTE_EMBEDDING_FEATURE_SIZE, "out_features": 1 }
        this.VelocityProjection: Linear = Linear(**expansion_param)
        this.ControlProjection: Linear = Linear(**expansion_param)

        this.Activate: Tanh = Tanh()

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
        # remove feature dimension
        x = torch.cat((vel, ctrl), dim = 2)[:, :, :, 0] # (batch, time step, note)

        return this.Activate(x)