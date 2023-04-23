from Data.MidiPianoRoll import MidiPianoRoll

from Model.Setting import DiscriminatorSetting, DropoutSetting

from torch import Tensor
from torch.nn import Module, Flatten, Unflatten, Sequential, Linear, Conv2d, Conv1d, LSTM, LeakyReLU, BatchNorm1d, Sigmoid

from typing import List

class Discriminator(Module):
    """
    Evaluate how *human* the output music sounds like.
    """

    def __init__(this):
        super().__init__()
        # calculate number of feature for each layer, both input and output has feature dimension of one
        feature_in: List[int] = [1] + DiscriminatorSetting.TIME_LAYER_FEATURE
        feature_out: List[int] = DiscriminatorSetting.TIME_LAYER_FEATURE + [1]
        def getKernel(layer: int) -> int:
            return DiscriminatorSetting.TIME_KERNEL_SIZE[layer]
        def getStride(layer: int) -> int:
            return getKernel(layer) // 2
        def getPadding(layer: int) -> int:
            return getStride(layer) // 2
        def activate() -> Module:
            return LeakyReLU(DiscriminatorSetting.LEAKY_SLOPE, True)

        # just a funny name: piano roll to feature sequence
        # this discriminator is a modified version of DCGAN
        roll2seq: List[Module] = list()
        # the input layer summarises both time and note, this will reduce the matrix to 1D
        roll2seq.extend([
            Conv2d(feature_in[0], feature_out[0],
                (MidiPianoRoll.DIMENSION_PER_TIME_STEP, getKernel(0)),
                (MidiPianoRoll.DIMENSION_PER_TIME_STEP, getStride(0)),
                (0, getPadding(0)), bias = False),
            Flatten(2, 3), # remove dimension 1 note axis, now (batch, feature, time)
            activate()
        ])

        layerCount: int = len(DiscriminatorSetting.TIME_KERNEL_SIZE)
        # for hidden layers
        for i in range(1, layerCount - 1):
            roll2seq.extend([
                Conv1d(feature_in[i], feature_out[i], getKernel(i), getStride(i), getPadding(i), bias = False),
                BatchNorm1d(feature_out[i]),
                activate()
            ])

        # for output layer, kernels are non-overlapping and with no padding, and summarise all feature to 1
        lastKernel: int = getKernel(-1)
        roll2seq.extend([
            Conv1d(feature_in[-1], feature_out[-1], lastKernel, lastKernel, bias = False),
            activate(),
            Flatten(1, 2), # remove feature, now (batch, time)
            Unflatten(1, (-1, 1)) # create feature axis for LSTM, now (batch, time, feature)
        ])

        # this helps us to summarise all time windows into one output
        seq2score: LSTM = LSTM(1, DiscriminatorSetting.SEQUENCE_HIDDEN, DiscriminatorSetting.SEQUENCE_LAYER,
            batch_first = True, dropout = DropoutSetting.DISCRIMINATOR_SEQUENCE)

        this.Summarise: Sequential = Sequential(*roll2seq, seq2score)
        this.SummaryProjection: Sequential = Sequential(
            Linear(DiscriminatorSetting.SEQUENCE_HIDDEN, 1),
            Sigmoid()
        )

    def forward(this, x: Tensor) -> Tensor:
        """
        Input: (batch, time step, note)
        Output: (batch); 1.0 is real, 0.0 is fake.
        """
        x = x.swapaxes(1, 2)[:, None, :, :] # (batch, feature, note, time step)

        # for the final LSTM output, we only care about the last score
        summary: Tensor = this.Summarise(x)[0][:, -1, :]
        return this.SummaryProjection(summary)[:, 0] # remove axis of size one