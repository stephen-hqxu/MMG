from Data.MidiTensor import MidiTensor
from Model import Setting
from Model.Setting import DropoutSetting

import torch
from torch import Tensor
from torch.nn import Module, Dropout, Conv2d

class TimeStepEmbedding(Module):
    """
    @brief Time step dimensionality reduction and feature embedding.
    """

    def __init__(this):
        super().__init__()
        # split the input feature into different embedding vectors based on their categories
        # combine dimensionality reduction of time and feature embedding together
        # move the time window by full size each time
        embedding_param = (1, Setting.EMBEDDED_FEATURE_SIZE, (1, Setting.TIME_WINDOW_SIZE), (1, Setting.TIME_WINDOW_SIZE))
        this.VelocityEmbedding: Conv2d = Conv2d(*embedding_param)
        this.ControlEmbedding: Conv2d = Conv2d(*embedding_param)

    def forward(this, x: Tensor) -> Tensor:
        """
        Input: (batch, time, note).
        CNN: (batch, channel, note, time), where `x` and `y` axes are swapped.
        Output: (batch, embedded feature, note, time token)
        """
        # add a channel axis
        x = x[:, None, :, :]
        # swap x and y
        x = x.swapaxes(2, 3)

        vel_emb: Tensor = this.VelocityEmbedding(x[:, :, MidiTensor.sliceVelocity(), :])
        ctrl_emb: Tensor = this.ControlEmbedding(x[:, :, MidiTensor.sliceControl(), :])
        # concatenate the embeddings in the original order of each feature to form a matrix
        # unlike the token-based model that connects all embedded vectors into one big vector
        # in the end, we will have a 2D embedded feature matrix
        return torch.cat((vel_emb, ctrl_emb), dim = 2)
    
class PositionEmbedding(Module):
    """
    @brief Note position embedding based on time.
    """

    def __init__(this):
        super().__init__()
        this.Zeroing: Dropout = Dropout(p = DropoutSetting.POSITION_EMBEDDING)

        # generate position encoder, based on the position encoder from the original paper
        # TODO: it's possible to train a positional embedder if this implementation doesn't work well
        position: Tensor = torch.arange(Setting.MAX_SEQUENCE_LENGTH)
        normaliser: Tensor = torch.exp(torch.arange(0, Setting.EMBEDDED_FEATURE_SIZE, 2) * (-4 / Setting.EMBEDDED_FEATURE_SIZE)).unsqueeze(1)

        pe: Tensor = torch.zeros((Setting.EMBEDDED_FEATURE_SIZE, 1, Setting.MAX_SEQUENCE_LENGTH), dtype = torch.float32)
        pe[0::2, 0, :] = torch.sin(position * normaliser)
        pe[1::2, 0, :] = torch.cos(position * normaliser)

        this.register_buffer("PositionEncoder", pe)

    def forward(this, x: Tensor) -> Tensor:
        """
        Input: (batch, embedded feature, note, time token)
        Output: same
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
        Input: shape of time step embedding.
        Output: shape of position embedding.
        """
        time_step_emb: Tensor = this.TimeStep(x)
        pos_emb: Tensor = this.Position(time_step_emb)
        return this.Zeroing(time_step_emb + pos_emb)