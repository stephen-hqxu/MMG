from Data.MidiPianoRoll import MidiPianoRoll
from Model import Setting
from Model.Setting import DropoutSetting

import torch
from torch import Tensor
from torch.nn import Module, Dropout, Conv2d, ConvTranspose2d

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
        Input: (batch, time step, note).
        CNN: (batch, channel, note, time step), where `x` and `y` axes are swapped.
        Output: (batch, embedded feature, note, time window)
        """
        # add a channel axis
        x = x[:, None, :, :]
        # swap x and y
        x = x.swapaxes(2, 3)

        vel_emb: Tensor = this.VelocityEmbedding(x[:, :, MidiPianoRoll.sliceVelocity(), :])
        ctrl_emb: Tensor = this.ControlEmbedding(x[:, :, MidiPianoRoll.sliceControl(), :])
        # concatenate the embeddings in the original order of each feature to form a matrix
        # unlike the token-based model that connects all embedded vectors into one big vector
        # in the end, we will have a 2D embedded feature matrix
        return torch.cat((vel_emb, ctrl_emb), dim = 2)
    
class PositionEmbedding(Module):
    """
    @brief Note position embedding based on time window.
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
        expansion_param = (Setting.EMBEDDED_FEATURE_SIZE, 1, (1, Setting.TIME_WINDOW_SIZE), (1, Setting.TIME_WINDOW_SIZE))
        this.VelocityExpansion: ConvTranspose2d = ConvTranspose2d(*expansion_param)
        this.ControlExpansion: ConvTranspose2d = ConvTranspose2d(*expansion_param)

    def forward(this, x: Tensor) -> Tensor:
        """
        Input: shape of output of full embedding.
        Output: shape of input of time step embedding.
        """
        # separate different type of note and un-project them from embedded features
        vel: Tensor = this.VelocityExpansion(x[:, :, MidiPianoRoll.sliceVelocity(), :])
        ctrl: Tensor = this.ControlExpansion(x[:, :, MidiPianoRoll.sliceControl(), :])
        
        # merge notes back to one axis
        x = torch.cat((vel, ctrl), dim = 2)
        # remove channel axis of size 1
        x = x[:, 0, :, :]
        # swap `x` and `y` back
        return x.swapaxes(1, 2)