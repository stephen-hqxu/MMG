from Data.MidiPianoRoll import MidiPianoRoll

from Model import Setting
from Model.Setting import EmbeddingSetting, DiscriminatorSetting, DropoutSetting

from torch import Tensor
from torch.nn import Module, Flatten, Unflatten, Sequential, Linear, Conv1d, LSTM, LeakyReLU, BatchNorm1d, Sigmoid

import numpy as np

from typing import List

class Discriminator(Module):
    """
    Evaluate how *human* the output music sounds like.
    """

    def __init__(this):
        super().__init__()
        # just a funny name: piano roll to feature sequence
        # this discriminator is a modified version of DCGAN
        roll2seq: List[Module] = list()

        # --------------------------------------------- summarise note ------------------------------------------ #
        roll2seq.extend([Linear(MidiPianoRoll.DIMENSION_PER_TIME_STEP, 1), LeakyReLU(DiscriminatorSetting.LEAKY_SLOPE, True)])
        # then we can convert this to a 1D time domain with an additional channel axis for latent vector
        roll2seq.extend([Flatten(1, 2), Unflatten(1, (1, -1))]) # ... -> (batch, sequence) -> (batch, feature, sequence)

        # ---------------------------------- summarise time steps in a time window ------------------------------ #
        # remember: input and output both have latent feature of 1
        exponent: np.ndarray = DiscriminatorSetting.TIME_FEATURE_START_EXPONENT + np.arange(DiscriminatorSetting.TIME_LAYER - 1)
        feature_count: np.ndarray = np.left_shift(1, exponent)
        
        feature_in: np.ndarray = feature_count
        feature_out: np.ndarray = feature_count
        # calculate power-of-2; make correction for input and output layer
        feature_in = np.insert(feature_in, 0, 1)
        feature_out = np.hstack((feature_out, 1))
        # filter properties
        kernel: int = DiscriminatorSetting.TIME_KERNEL
        stride: int = kernel // 2
        padding: int = stride // 2

        # use a linear layer at the end to summarise up remaining sequence in a time window
        remaining_sequence: int = EmbeddingSetting.TIME_WINDOW_SIZE // np.power(stride, DiscriminatorSetting.TIME_LAYER, dtype = np.uint32)
        need_final_layer: bool = remaining_sequence > 1
        
        for idx, (f_in, f_out) in enumerate(zip(feature_in, feature_out)):
            roll2seq.append(Conv1d(f_in, f_out, kernel, stride, padding))

            if idx == DiscriminatorSetting.TIME_LAYER - 1 and not need_final_layer:
                # no norm or activation and the end if there is no more layer
                continue
            if idx > 0:
                # no norm at the beginning
                roll2seq.append(BatchNorm1d(f_out))
            roll2seq.append(LeakyReLU(DiscriminatorSetting.LEAKY_SLOPE, True))
        
        if need_final_layer:
            roll2seq.append(Conv1d(1, 1, remaining_sequence, remaining_sequence))
        # output activation
        roll2seq.append(Sigmoid())
        
        # ---------------------------------------- summarise all time window ------------------------------------ #
        # move feature axis to the end, feature has dimension of one
        roll2seq.extend([Flatten(1, 2), Unflatten(1, (-1, 1))]) # (batch, sequence)

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
        # for the final LSTM output, we only care about the last score
        summary: Tensor = this.Summarise(x)[0][:, -1, :]
        return this.SummaryProjection(summary)