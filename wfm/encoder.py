# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:45:58 2026

@author: ra
"""

# wfmmarl_simulator/wfm/encoder.py
import torch
import torch.nn as nn

class WFMEncoder(nn.Module):
    """
    Wireless Foundation Model encoder (state abstraction).
    Raw obs -> latent vector
    """

    def __init__(self, input_dim: int, embed_dim: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        return self.net(x)


class ProjectionHead(nn.Module):
    """
    Projection head used only during contrastive pretraining.
    """
    def __init__(self, embed_dim: int = 16, proj_dim: int = 32):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x):
        return self.proj(x)