# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:51:09 2026

@author: ra
"""

# wfmmarl_simulator/marl/mappo/critic.py
import torch
import torch.nn as nn

class CentralizedCritic(nn.Module):
    """
    Centralized value network observing joint latent state.
    """

    def __init__(self, embed_dim: int, num_agents: int):
        super().__init__()
        input_dim = embed_dim * num_agents
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, joint_latent):
        # joint_latent: shape (batch, N*embed_dim)
        return self.net(joint_latent)