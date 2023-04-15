from Model import Setting
from Model.Setting import DropoutSetting

import torch
from torch import Tensor
from torch.nn import Module, Dropout, LayerNorm, Linear, LeakyReLU

class Residual(Module):
    """
    Layer normalisation and regularisation to prevent parameters from changing too much.
    """

    def __init__(this, ascendant: Module):
        """
        @param ascendant The parent layer of this residual layer.
        It's important to ensure the output of this layer has consistent shape with its input.
        """
        super().__init__()
        this.Ascendant: Module = ascendant

        this.Normaliser: LayerNorm = LayerNorm(Setting.EMBEDDED_FEATURE_SIZE)
        this.Zeroing: Dropout = Dropout(p = DropoutSetting.RESIDUAL)

    def forward(this, x: Tensor, *args, **kwargs) -> Tensor:
        """
        The first tensor will be used for summed up with the output from the ascendant layer.
        The rest of arguments will be passed to the ascendant layer.

        Input: shape doesn't matter, provided embedded feature is the last axis.
            This input will be used to sum with the result from the ascendant layer.
        Output: depends on the output of the ascendant layer.
        """
        return this.Normaliser(x + this.Zeroing(this.Ascendant(*args, **kwargs)))
    
class FeedForward(Module):
    """
    The MLP in the encoder/decoder.
    """

    def __init__(this):
        super().__init__()
        this.Input: Linear = Linear(Setting.EMBEDDED_FEATURE_SIZE, Setting.FEED_FORWARD_LATENT_SIZE)
        this.Output: Linear = Linear(Setting.FEED_FORWARD_LATENT_SIZE, Setting.EMBEDDED_FEATURE_SIZE)

        this.Activation: LeakyReLU = LeakyReLU(0.05)
        this.Zeroing: Dropout = Dropout(p = DropoutSetting.FEED_FORWARD)

    def forward(this, x: Tensor) -> Tensor:
        """
        Input: (batch, sequence, embedded feature)
        Output: same
        """
        x = this.Input(x)
        x = this.Activation(x)
        x = this.Zeroing(x)
        return this.Output(x)