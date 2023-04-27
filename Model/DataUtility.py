from Data.MidiPianoRoll import MidiPianoRoll
from Model.Setting import EmbeddingSetting

import torch
from torch import Tensor
import numpy as np

from typing import Union

LengthData = Union[int, np.ndarray]

def calcTimeWindowLength(time_step: LengthData) -> LengthData:
    """
    @brief Calculate the number of time window based on given number of time step.

    @param time_step The number of time step.
    @return The length of time window.
    """
    # minus one on time step because it is a size, we are calculating as an index
    return (time_step - 1) // EmbeddingSetting.TIME_WINDOW_SIZE + 1

def calcTimeStepLength(time_window: LengthData) -> LengthData:
    """
    @brief Inverse of calcTimeWindowLength(); time step is a round-up multiple of the original time step.

    @see calcTimeWindowLength()
    """
    return time_window * EmbeddingSetting.TIME_WINDOW_SIZE

def calcSequenceLength(time_window: LengthData) -> LengthData:
    """
    @brief Calculate the sequence length based on the number of time window given.

    @param time_window The number of time window.
    @return The length of sequence.
    """
    return time_window * MidiPianoRoll.DIMENSION_PER_TIME_STEP

def makeNoPeekMask(extent: int) -> Tensor:
    """
    @brief Create mask that prevents paying attention to the sequence beyond the current input.

    @param extent The length of of extent of the square mask matrix.
    @return The boolean mask, with size (extent, extent).
    This mask can be sliced from top-left origin to any size square matrix for different input sequence length.
    """
    mask: Tensor = torch.ones((extent, extent), dtype = torch.bool)
    # create triangular matrix
    return torch.triu(mask, out = mask, diagonal = 1)

def makePadMask(time_window: np.ndarray) -> Tensor:
    """
    @brief Create mask that ignores attention at certain position.
    This is used for padding due to use of mini-batch, such that padding are always at the end of each sequence.

    @param time_window Specifies an array of time window size for each batch.
    @return A matrix of padding mask, with size (batch, L) where `L` is the longest sequence in the batch.
    This matrix can be sliced based on current sequence length.
    """
    batchSize: int = time_window.size
    # things now get a bit tricky, we need to fill padding index for full padded time window
    # if any time step does not make up a full time window, leave it as zero as we filled up initially
    sequence_pad_start: np.ndarray = calcSequenceLength(time_window)
    max_sequence: int = np.max(sequence_pad_start)

    mask: Tensor = torch.zeros((batchSize, max_sequence), dtype = torch.bool)
    for b in range(batchSize):
        mask[b, sequence_pad_start[b]:] = True
    return mask