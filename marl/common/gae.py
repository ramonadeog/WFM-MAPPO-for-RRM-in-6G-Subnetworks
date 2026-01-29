# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:51:43 2026

@author: ra
"""

# wfmmarl_simulator/marl/common/gae.py
import torch

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    """
    rewards: (T, N)
    values: (T+1, N)
    dones: (T, N)
    returns advantages (T, N), returns (T, N)
    """

    T, N = rewards.shape
    advantages = torch.zeros(T, N)
    gae = torch.zeros(N)

    for t in reversed(range(T)):
        delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae

    returns = advantages + values[:-1]
    return advantages, returns