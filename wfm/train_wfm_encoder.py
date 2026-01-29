# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:48:06 2026

@author: ra
"""

# wfmmarl_simulator/wfm/train_wfm_encoder.py
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np

from environment.subnetwork_env import SubnetworkEnv, EnvConfig
from wfm.encoder import WFMEncoder, ProjectionHead
from wfm.contrastive_dataset import WFMDataset
from wfm.contrastive_loss import nt_xent_loss

def collect_observations(env, num_steps=50000):
    obs_list = []
    env.reset()
    for _ in range(num_steps):
        actions = np.random.randint(0, env.cfg.K, size=env.cfg.N)
        obs, _, _, _ = env.step(actions)
        obs_list.extend(obs)     # store all agents’ obs
    return obs_list


def train_wfm_encoder(
    embed_dim=16,
    proj_dim=32,
    batch_size=256,
    num_epochs=5,
    lr=1e-3,
    save_path="pretrained_wfm_encoder.pt"
):
    # 1. Initialize environment
    cfg = EnvConfig()
    env = SubnetworkEnv(cfg)

    # 2. Collect raw observations
    raw_obs = collect_observations(env, num_steps=40000)
    dataset = WFMDataset(raw_obs)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 3. Build model
    input_dim = env.obs_dim
    encoder = WFMEncoder(input_dim=input_dim, embed_dim=embed_dim)
    proj = ProjectionHead(embed_dim=embed_dim, proj_dim=proj_dim)

    model = nn.Sequential(encoder, proj)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 4. Training loop
    for epoch in range(num_epochs):
        total_loss = 0
        for x1, x2 in loader:
            z1 = model(x1)
            z2 = model(x2)
            loss = nt_xent_loss(z1, z2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg = total_loss / len(loader)
        print(f"[Epoch {epoch+1}/{num_epochs}] Loss: {avg:.4f}")

    # 5. Save encoder (without projection head)
    torch.save(encoder.state_dict(), save_path)
    print(f"Saved pretrained WFM encoder → {save_path}")


if __name__ == "__main__":
    train_wfm_encoder()