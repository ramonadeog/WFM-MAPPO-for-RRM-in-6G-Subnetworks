# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:03:31 2026

@author: ra
"""

# wfmmarl_simulator/marl/baselines/random_policy.py
import numpy as np

class RandomPolicy:
    """
    Always selects a random channel for each agent.
    """

    def __init__(self, num_agents, num_channels):
        self.N = num_agents
        self.K = num_channels

    def act(self, obs):
        # obs ignored
        return np.random.randint(0, self.K, size=self.N)