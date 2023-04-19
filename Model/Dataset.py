from Model.Component.Coder import CoderMask

from Model.Setting import SpecialTokenSetting, EmbeddingSetting, DatasetSetting
from Data.MidiPianoRoll import MidiPianoRoll

from pretty_midi import PrettyMIDI

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd

from typing import Tuple, List, Union

class ASAPDataset(Dataset):
    """
    @see https://github.com/fosfrancesco/asap-dataset
    """

    def __init__(this):
        """
        @brief Initialise the ASAP dataset.
        """
        this.ASAPRoot: str = DatasetSetting.ASAP_PATH + '/'
        """
        Root to the ASAP dataset; with trailing slash.
        """
        # TODO: can also include *Maestro* performance MIDI if needed
        this.ASAPMeta: pd.DataFrame = pd.read_csv(this.ASAPRoot + "metadata.csv", usecols = ["midi_score", "midi_performance"])
        """
        The metadata of ASAP dataset.
        """

    def __len__(this) -> int:
        return len(this.ASAPMeta.index)
    
    def __getitem__(this, idx: int) -> Tuple[Tensor, Tensor]:
        # read MIDI from file into piano roll
        midi_filename: pd.DataFrame = this.ASAPMeta.iloc[[idx]]
        midi_data: List[Tensor] = list()
        for _, filename in midi_filename.items():
            midi: PrettyMIDI = PrettyMIDI(this.ASAPRoot + filename)
            piano_roll: MidiPianoRoll = MidiPianoRoll.fromMidi(midi)
            midi_data.append(piano_roll.viewTensor()) # reference counter will detach the reference automatically
        return tuple(midi_data)
    
class BatchCollation:
    """
    @brief Post-process data returned from dataset so they can be used in a batch.
    """
    LengthData = Union[int, np.ndarray]

    def __init__(this):
        pass

    @staticmethod
    def calcTimeWindowLength(time_step: LengthData) -> LengthData:
        """
        @brief Calculate the number of time window based on given number of time step.

        @param time_step The number of time step.
        @return The length of time window.
        """
        # minus one on time step because it is a size, we are calculating as an index
        return (time_step - 1) // EmbeddingSetting.TIME_WINDOW_SIZE + 1
    
    @staticmethod
    def calcTimeStepLength(time_window: LengthData) -> LengthData:
        """
        @brief Inverse of calcTimeWindowLength(); time step is a round-up multiple of the original time step.

        @see calcTimeWindowLength()
        """
        return time_window * EmbeddingSetting.TIME_WINDOW_SIZE
    
    @staticmethod
    def calcSequenceLength(time_window: LengthData) -> LengthData:
        """
        @brief Calculate the sequence length based on the number of time window given.

        @param time_window The number of time window.
        @return The length of sequence.
        """
        return time_window * MidiPianoRoll.DIMENSION_PER_TIME_STEP
    
    @staticmethod
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
    
    @staticmethod
    def makePadMask(time_window: np.ndarray) -> Tensor:
        """
        @brief Create mask that ignores attention at certain position.
        This is used for padding due to use of mini-batch, such that padding are always at the end of each sequence.

        @param time_window Specifies an array of time window size for each batch.
        @return A matrix of padding mask, with size (batch, L) where `L` is the longest sequence in the batch.
        This matrix can be sliced based on current sequence length.
        """
        batch_size: int = time_window.size
        # things now get a bit tricky, we need to fill padding index for full padded time window
        # if any time step does not make up a full time window, leave it as zero as we filled up initially
        sequence_pad_start: np.ndarray = BatchCollation.calcSequenceLength(time_window)
        max_sequence: int = np.max(sequence_pad_start)

        mask: Tensor = torch.zeros((batch_size, max_sequence), dtype = torch.bool)
        for b in range(batch_size):
            mask[b, sequence_pad_start[b]:] = True
        return mask

    def __call__(this, batch: List[Tuple[Tensor, Tensor]]) -> Tuple[Tensor, Tensor, CoderMask]:
        """
        @brief Collate a number of samples into a batch.

        @param batch Array of data sample of robotic and performance MIDI from dataset.
        Each sample should have shape of (time step, note).
        @return Batched sample, with coder mask computed.
        """
        batchSize: int = len(batch)
        # first, we should pad the data sample of different time lengths
        data: Tuple[List[Tensor], List[Tensor]] = (list(), list())
        # also record the length of time window of each data sample
        timeWindow: np.ndarray = np.zeros((batchSize, 2), dtype = np.uint32)
        for idx, b in enumerate(batch):
            sample, label = b

            data[0].append(sample)
            data[1].append(label)
            timeWindow[idx] = BatchCollation.calcTimeWindowLength(np.array([sample.size(0), label.size(0)], dtype = np.uint32))
        # we pad zero by default, to indicate no note information
        data: List[Tensor] = [pad_sequence(d, True, 0) for d in data] # (batch, time step, note)
        timeWindow = timeWindow.transpose()

        # data padding and fill token
        for d_i in range(len(data)): # for example and label
            # round up the size so it's a multiple of window size
            old_size: int = data[d_i].size(1)
            # create one more time window for the end token
            new_size: int = BatchCollation.calcTimeStepLength(BatchCollation.calcTimeWindowLength(old_size) + 1)

            start_pad: Tensor = torch.zeros((batchSize, EmbeddingSetting.TIME_WINDOW_SIZE, MidiPianoRoll.DIMENSION_PER_TIME_STEP), dtype = torch.uint8)
            end_pad: Tensor = torch.zeros((batchSize, new_size - old_size, MidiPianoRoll.DIMENSION_PER_TIME_STEP), dtype = torch.uint8)
            # insert padding
            data[d_i] = torch.concatenate((start_pad, data[d_i], end_pad), dim = 1)

            # fill special tokens
            data[d_i][:, :EmbeddingSetting.TIME_WINDOW_SIZE, :] = SpecialTokenSetting.SOS
            # for end and pad token, we only fill in if there is a full time window
            for b in range(batchSize):
                end_step: int = BatchCollation.calcTimeStepLength(timeWindow[d_i][b] + 1)
                pad_step: int = end_step + EmbeddingSetting.TIME_WINDOW_SIZE

                data[d_i][b, end_step:pad_step, :] = SpecialTokenSetting.EOS
                data[d_i][b, pad_step:, :] = SpecialTokenSetting.PAD
        # include start and end token to the total time window count
        timeWindow += 2

        # generate mask
        targetSequence: int = BatchCollation.calcSequenceLength(BatchCollation.calcTimeWindowLength(data[1].size(1)))
        mask: CoderMask = CoderMask(
            SourcePadding = BatchCollation.makePadMask(timeWindow[0]),
            TargetPadding = BatchCollation.makePadMask(timeWindow[1]),
            TargetAttention = BatchCollation.makeNoPeekMask(targetSequence)
        )
        return (*data, mask)