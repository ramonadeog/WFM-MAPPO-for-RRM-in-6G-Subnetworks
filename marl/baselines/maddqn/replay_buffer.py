# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:05:13 2026

@author: ra
"""

# wfmmarl_simulator/marl/baselines/maddqn/replay_buffer.py
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, num_agents):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, num_agents, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, num_agents), dtype=np.int64)
        self.rewards = np.zeros((capacity, num_agents), dtype=np.float32)
        self.dones = np.zeros((capacity, num_agents), dtype=np.float32)

    def add(self, obs, actions, rewards, next_obs, dones):
        idx = self.ptr
        self.obs[idx] = obs
        self.actions[idx] = actions
        self.rewards[idx] = rewards
        self.next_obs[idx] = next_obs
        self.dones[idx] = dones

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.obs[idxs]),
            torch.tensor(self.actions[idxs]),
            torch.tensor(self.rewards[idxs]),
            torch.tensor(self.next_obs[idxs]),
            torch.tensor(self.dones[idxs])
        )