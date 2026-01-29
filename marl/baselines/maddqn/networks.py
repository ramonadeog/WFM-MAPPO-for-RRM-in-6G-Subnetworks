# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:06:05 2026

@author: ra
"""

# wfmmarl_simulator/marl/baselines/maddqn/networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class DQNNetwork(nn.Module):
    """
    Simple MLP Q-network: Q(s_i, a_i).
    """

    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)