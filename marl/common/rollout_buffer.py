# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:00:11 2026

@author: ra
"""

# wfmmarl_simulator/marl/common/rollout_buffer.py
import torch

class RolloutBuffer:
    """
    Stores trajectories for MAPPO.
    Dimensions:
      obs_latent: (T, N, embed_dim)
      actions:    (T, N)
      log_probs:  (T, N)
      rewards:    (T, N)
      dones:      (T, N)
      values:     (T+1, N)  ← critic outputs
    """

    def __init__(self, T, num_agents, embed_dim, device):
        self.T = T
        self.N = num_agents
        self.embed_dim = embed_dim
        self.device = device

        self.obs_latent = torch.zeros(T, self.N, embed_dim, device=device)
        self.actions = torch.zeros(T, self.N, dtype=torch.long, device=device)
        self.log_probs = torch.zeros(T, self.N, device=device)
        self.rewards = torch.zeros(T, self.N, device=device)
        self.dones = torch.zeros(T, self.N, device=device)

        self.values = torch.zeros(T + 1, self.N, device=device)

        self.ptr = 0

    def add(self, obs_latent, actions, log_probs, rewards, dones, values):
        """
        Add one timestep for all agents.
        obs_latent: (N, embed_dim)
        actions:    (N,)
        log_probs:  (N,)
        rewards:    (N,)
        dones:      (N,)
        values:     (N,) value estimates V(s)
        """
        t = self.ptr
        self.obs_latent[t] = obs_latent
        self.actions[t] = actions
        self.log_probs[t] = log_probs
        self.rewards[t] = rewards
        self.dones[t] = dones
        self.values[t] = values
        self.ptr += 1

    def finish(self, last_values):
        """
        Store final V(s_T) for advantage computation.
        """
        self.values[self.T] = last_values

    def full(self):
        return self.ptr == self.T