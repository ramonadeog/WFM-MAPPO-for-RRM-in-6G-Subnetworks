# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:50:42 2026

@author: ra
"""

# wfmmarl_simulator/marl/mappo/actor.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MAPPOActor(nn.Module):
    """
    Decentralized policy π(a_i | s_i^latent)
    """

    def __init__(self, embed_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, s_latent):
        logits = self.net(s_latent)
        return F.softmax(logits, dim=-1)