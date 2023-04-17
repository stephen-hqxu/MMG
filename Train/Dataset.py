from Train.Setting import DatasetSetting
from Data.MidiPianoRoll import MidiPianoRoll

from pretty_midi import PrettyMIDI

from torch import Tensor
from torch.utils.data import Dataset

import pandas as pd

from typing import Tuple, List

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