import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from extract import getMapping
import pretty_midi
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler
import yaml

class MMGDataset(Dataset):
    def __init__(self, data_cfg):
        super().__init__()
        self.dataset_path = data_cfg.get("dataset_path")
        self.metadata_path = data_cfg.get("metadata_path")
        self.dataset_save_path = data_cfg.get("dataset_save_path")

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
        mapping = getMapping(pretty_midi.PrettyMIDI(os.path.join(self.dataset_path, performance_path)), 
                             pretty_midi.PrettyMIDI(os.path.join(self.dataset_path, robotic_path)))
        # don't include pitch
        robotic_notes = np.delete(mapping[:,0], -1, axis=1)
        performance_notes = np.delete(mapping[:,1], -1, axis=1)
        return torch.tensor(robotic_notes), torch.tensor(performance_notes)

def load_config(path) -> dict:
    """
    Loads and parses a YAML configuration file.
    path: path to YAML configuration file
    return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.Loader)
    return cfg

def load_data(data_cfg: dict):

    dataset = MMGDataset(data_cfg)

    # Create splits
    indices = list(range(len(dataset)))
    if data_cfg.get("shuffle"):
        np.random.shuffle(indices)

    train_prop, val_prop, test_prop = data_cfg.get("dataset_split")
    train_split = int(np.floor(train_prop * len(dataset)))
    val_split = train_split + int(np.floor(val_prop * len(dataset)))
    train_indices, val_indices, test_indices = indices[:train_split], indices[train_split:val_split], indices[val_split:]

    batch_size = data_cfg.get("batch_size")

    train_loader = DataLoader(dataset, batch_size=batch_size, sampler=SubsetRandomSampler(train_indices))
    val_loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(val_indices))
    test_loader = DataLoader(dataset, batch_size=batch_size, sampler=SequentialSampler(test_indices))

    return train_loader, val_loader, test_loader