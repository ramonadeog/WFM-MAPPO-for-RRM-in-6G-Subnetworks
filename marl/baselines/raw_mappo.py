# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:04:16 2026

@author: ra
"""

# wfmmarl_simulator/marl/baselines/raw_mappo.py
import torch
from marl.mappo.mappo_trainer import MAPPOTrainer
from marl.mappo.actor import MAPPOActor
from marl.mappo.critic import CentralizedCritic

def build_raw_mappo(env, device="cpu"):
    """
    Builds a MAPPO trainer using raw environment observations directly
    without the WFM encoder.
    """
    raw_obs_dim = env.obs_dim
    embed_dim = raw_obs_dim    # raw-state MAPPO uses identical dimensionality

    trainer = MAPPOTrainer(
        env=env,
        actor_class=lambda emb, K: MAPPOActor(embed_dim, env.cfg.K),
        critic_class=lambda emb, N: CentralizedCritic(embed_dim, env.cfg.N),
        encoder=None,                # <--- NO encoder for raw-state baseline
        embed_dim=embed_dim,
        device=device
    )

    return trainer