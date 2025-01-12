from Model.Component.Coder import CoderMask

from Model.Setting import SpecialTokenSetting, EmbeddingSetting, TransformerSetting, DatasetSetting, TrainingSetting
from Model import DataUtility
from Data.MidiPianoRoll import MidiPianoRoll

from pretty_midi import PrettyMIDI

import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

import numpy as np
import pandas as pd

from typing import Tuple, List, Dict

class RandomDataset(Dataset):
    """
    @brief For debug purposes only.
    """

    def __init__(this, random_seed: int, max_time_step: int, size: int):
        this.SampleGenerator: torch.Generator = torch.Generator().manual_seed(random_seed)
        """
        Generate random dataset.
        """
        this.SampleState: Dict[int, Tensor] = dict()
        """
        A hash table of generator states for each index, to be restored.
        """

        this.MaxTimeStep: int = max_time_step
        """
        The maximum possible number of time step to be generated.
        """
        this.Size: int = size
        """
        The number of entry in the dataset.
        """

    def __len__(this) -> int:
        return this.Size
    
    def __getitem__(this, idx: int) -> Tuple[Tensor, Tensor]:
        if idx in this.SampleState:
            # restore generator state
            this.SampleGenerator.set_state(this.SampleState[idx])
        else:
            # store the current state so it can be restored later
            this.SampleState[idx] = this.SampleGenerator.get_state()

        timeStep: Tensor = torch.randint(1, this.MaxTimeStep + 1, (2, ), generator = this.SampleGenerator)
        # do not include special tokens, so range is [0, 128)
        return tuple((torch.randint(0, MidiPianoRoll.NOTE_MAX_LEVEL,
            (ts, MidiPianoRoll.DIMENSION_PER_TIME_STEP), generator = this.SampleGenerator, dtype = torch.uint8) for ts in timeStep))

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
        midi_filename: pd.DataFrame = this.ASAPMeta.iloc[idx]
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

    START_PAD: Tensor = torch.zeros((
        TrainingSetting.BATCH_SIZE,
        EmbeddingSetting.TIME_WINDOW_SIZE,
        MidiPianoRoll.DIMENSION_PER_TIME_STEP
    ), dtype = torch.int32)
    """
    Padding to be placed at the beginning of each batched samples for the SOS token.
    """
    END_PAD: Tensor = torch.zeros((
        TrainingSetting.BATCH_SIZE,
        2 * EmbeddingSetting.TIME_WINDOW_SIZE,
        MidiPianoRoll.DIMENSION_PER_TIME_STEP
    ), dtype = torch.int32)
    """
    Padding for rounding up the sample to full time window size, and for the end EOS token.
    """
    
    def __init__(this):
        pass

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
            timeWindow[idx] = DataUtility.calcTimeWindowLength(np.array([sample.size(0), label.size(0)], dtype = np.uint32))
        # we pad zero by default, to indicate no note information
        # convert to IntTensor because embedding only works with that
        data: List[Tensor] = [pad_sequence(d, True, 0).int() for d in data] # (batch, time step, note)
        timeWindow = timeWindow.transpose()

        # data padding and fill token
        for d_i in range(len(data)): # for sample and label
            # round up the size so it's a multiple of window size
            old_size: int = data[d_i].size(1)
            # create one more time window for the end token
            new_size: int = DataUtility.calcTimeStepLength(DataUtility.calcTimeWindowLength(old_size) + 1)
            end_pad_size: int = new_size - old_size
            assert(end_pad_size <= BatchCollation.END_PAD.size(1))

            start_pad: Tensor = BatchCollation.START_PAD[:batchSize, :, :]
            end_pad: Tensor = BatchCollation.END_PAD[:batchSize, :end_pad_size, :]
            # insert padding
            data[d_i] = torch.concatenate((start_pad, data[d_i], end_pad), dim = 1)

            # fill special tokens
            data[d_i][:, :EmbeddingSetting.TIME_WINDOW_SIZE, :] = SpecialTokenSetting.SOS
            # for end and pad token, we only fill in if there is a full time window
            for b in range(batchSize):
                end_step: int = DataUtility.calcTimeStepLength(timeWindow[d_i][b] + 1)
                pad_step: int = end_step + EmbeddingSetting.TIME_WINDOW_SIZE

                # pur EOS first, then fill in the rest with PAD
                data[d_i][b, end_step:pad_step, :] = SpecialTokenSetting.EOS
                data[d_i][b, pad_step:, :] = SpecialTokenSetting.PAD
        # include start and end token to the total time window count
        timeWindow += 2

        # generate mask
        targetSequence: int = DataUtility.calcSequenceLength(DataUtility.calcTimeWindowLength(data[1].size(1)))
        # do not generate explicit masks if causal attention is intended to be used
        mask: CoderMask = CoderMask(
            SourcePadding = DataUtility.makePadMask(timeWindow[0]),
            TargetPadding = DataUtility.makePadMask(timeWindow[1]),
            TargetAttention = DataUtility.makeNoPeekMask(targetSequence)
        ) if not TransformerSetting.CAUSAL_ATTENTION_MASK else CoderMask()
        return (*data, mask)
    
def loadData(dataset: Dataset) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    @brief Load a dataset, and splits the dataset into training, validation and testing partition.

    @param dataset The dataset to be loaded.
    @return Data loaders for different purposes.
    """
    # split the dataset randomly
    generator: torch.Generator = torch.Generator().manual_seed(DatasetSetting.DATA_SHUFFLE_SEED)
    split_data = random_split(dataset, DatasetSetting.DATA_SPLIT, generator)

    def makeLoader(data: Dataset) -> DataLoader:
        loaderGen: torch.Generator = torch.Generator().set_state(generator.get_state())
        return DataLoader(data, TrainingSetting.BATCH_SIZE, shuffle = True, collate_fn = BatchCollation(), generator = loaderGen)
    return tuple((makeLoader(d) for d in split_data))