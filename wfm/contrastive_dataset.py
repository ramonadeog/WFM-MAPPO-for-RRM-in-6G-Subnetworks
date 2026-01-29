# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:40:33 2026

@author: ra
"""

# wfmmarl_simulator/wfm/contrastive_dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset

class WFMDataset(Dataset):
    """
    Dataset of raw observations from environment.
    For each raw obs x, we generate two augmentations x1 and x2.
    """

    def __init__(self, raw_obs_list, noise_std=0.01, dropout_p=0.1):
        self.X = np.array(raw_obs_list, dtype=np.float32)
        self.noise_std = noise_std
        self.dropout_p = dropout_p

    def augment(self, x):
        # Gaussian noise
        x_aug = x + self.noise_std * np.random.randn(*x.shape)
        # Random dropout
        mask = np.random.rand(*x.shape) > self.dropout_p
        return x_aug * mask.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        x1 = self.augment(x)
        x2 = self.augment(x)
        return torch.tensor(x1, dtype=torch.float32), torch.tensor(x2, dtype=torch.float32)