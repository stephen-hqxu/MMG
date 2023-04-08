import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from extract import getMapping
import pretty_midi

class MMGDataset(Dataset):
    def __init__(self, data_cfg):
        super().__init__()
        self.dataset_path = data_cfg.get("dataset_path")
        self.metadata_path = data_cfg.get("metadata_path")

        self.data = self._build_dataset()

    def _build_dataset(self):
        """
        Input, all robotic bars + idxs of most similar bars to tgt
        Target, single tgt bar
        """
        if os.path.exists(self.dataset_save_path):
            return pd.read_csv(self.dataset_save_path)
        else:
            print("Building dataset")
            data = pd.read_csv(self.metadata_path)
            data = data[['midi_score', 'midi_performance']]
            data.to_csv(self.dataset_save_path, index=False)
            return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        robotic_path, performance_path = self.data.iloc[index]
        mapping = getMapping(pretty_midi.PrettyMIDI(performance_path), pretty_midi.PrettyMIDI(robotic_path))
        robotic_notes = np.delete(mapping[:,0], -1, axis=1)
        performance_notes = np.delete(mapping[:,1], -1, axis=1)
        return torch.tensor(robotic_notes), torch.tensor(performance_notes)
