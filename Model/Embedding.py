from Data.MidiTensor import MidiTensor
from Model import Setting
from Model.Setting import EmbeddingSetting

import torch
from torch import Tensor
from torch.nn import Module, Dropout, Embedding

class TimeStepEmbedding(Module):
    """
    @brief Embed MIDI information in a single time step.
    """

    def __init__(this):
        super().__init__()
        # split the input feature into different embedding vectors based on their categories
        this.VelocityEmbedding: Embedding = Embedding(EmbeddingSetting.DICTIONARY_VELOCITY, Setting.FEATURE_SIZE,
            max_norm = 1.0, norm_type = torch.inf)
        this.ControlEmbedding: Embedding = Embedding(EmbeddingSetting.DICTIONARY_CONTROL, Setting.FEATURE_SIZE,
            max_norm = 127.0, norm_type = torch.inf)

    def forward(this, x: Tensor) -> Tensor:
        vel_emb: Tensor = this.VelocityEmbedding(x[:, :, MidiTensor.sliceVelocity()])
        ctrl_emb: Tensor = this.ControlEmbedding(x[:, :, MidiTensor.sliceControl()])
        # concatenate the embeddings in the original order of each feature to form a matrix
        # unlike the token-based model that connects all embedded vectors into one big vector
        # in the end, we will have a 2D embedded feature matrix
        return torch.cat((vel_emb, ctrl_emb), dim = 2)
    
class PositionEmbedding(Module):
    """
    @brief Embed note position information.
    """

    def __init__(this):
        super().__init__()
        this.Zeroing: Dropout = Dropout(p = EmbeddingSetting.POSITION_DROPOUT)

        # generate position encoder, based on the position encoder from the original paper
        position: Tensor = torch.arange(Setting.TIME_WINDOW_SIZE).unsqueeze(1)
        normaliser: Tensor = torch.exp(torch.arange(0, Setting.FEATURE_SIZE, 2) * (-4 / Setting.FEATURE_SIZE))

        pe: Tensor = torch.zeros((Setting.TIME_WINDOW_SIZE, 1, Setting.FEATURE_SIZE), dtype = torch.float32)
        pe[:, 0, 0::2] = torch.sin(position * normaliser)
        pe[:, 0, 1::2] = torch.cos(position * normaliser)

        this.register_buffer("PositionEncoder", pe)

    def forward(this, x: Tensor) -> Tensor:
        # the input matrix has 4 dimensions: sequence, batch, input feature, embedded feature
        # the dimension of our model is in 2D
        for bat in range(x.size(1)):
            x[:, bat] += this.PositionEncoder[:x.size(0)]
        return this.Zeroing(x)
    
class FullEmbedding(Module):
    """
    All embedding models put together.
    """

    def __init__(this):
        super().__init__()
        this.Zeroing: Dropout = Dropout(p = EmbeddingSetting.FULL_DROPOUT)

        this.TimeStep: TimeStepEmbedding = TimeStepEmbedding()
        this.Position: PositionEmbedding = PositionEmbedding()

    def forward(this, x: Tensor) -> Tensor:
        time_step_emb: Tensor = this.TimeStep(x)
        pos_emb: Tensor = this.Position(time_step_emb)
        return this.Zeroing(time_step_emb + pos_emb)