from Model import Setting
from Model.Setting import DropoutSetting

import torch
from torch import Tensor
from torch.nn import Module, Dropout, LayerNorm

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

    def forward(this, x: Tensor, *args) -> Tensor:
        """
        Input: shape doesn't matter, provided embedded feature is the last axis.
            This input will be used to sum with the result from the ascendant layer.
        Output: depends on the output of the ascendant layer.
        """
        return this.Normaliser(x + this.Zeroing(this.Ascendant(*args)))