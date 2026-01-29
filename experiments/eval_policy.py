# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:10:46 2026

@author: ra
"""

# wfmmarl_simulator/experiments/eval_policy.py
import yaml
import torch
import numpy as np

from environment.subnetwork_env import SubnetworkEnv, EnvConfig
from wfm.encoder import WFMEncoder
from marl.mappo.actor import MAPPOActor

def evaluate_policy(env, actor_list, encoder=None, episodes=100):
    total_sumrate = 0
    total_interf_viol = 0
    total_qos_viol = 0
    total_steps = 0

    for ep in range(episodes):
        obs = env.reset()

        while True:
            # encode if using WFM
            if encoder is not None:
                obs_latent = encoder(torch.tensor(obs, dtype=torch.float32))
            else:
                obs_latent = torch.tensor(obs, dtype=torch.float32)

            # Agents act independently
            actions = []
            for i, actor in enumerate(actor_list):
                #print('a', obs_latent)
                policy = actor(obs_latent[i:i+1])
                a = torch.argmax(policy, dim=1).item()
                actions.append(a)
            actions = np.array(actions)

            obs, r, done, info = env.step(actions)

            total_sumrate += info["rates_bps"].sum()
            total_interf_viol += np.sum(info["interf_W"] > env.cfg.I_th_W)
            total_qos_viol += np.sum(info["rates_bps"] < env.cfg.R_min_bps)
            total_steps += env.cfg.N

            if done:
                break

    return {
        "avg_sumrate_Mbps": total_sumrate / (episodes * 1e6),
        "interference_violation_rate": total_interf_viol / total_steps,
        "qos_violation_rate": total_qos_viol / total_steps
    }

def main():
    cfg = yaml.safe_load(open("experiments/config.yaml"))
    env_cfg = EnvConfig(**cfg["environment"])
    env = SubnetworkEnv(env_cfg)

    device = cfg["device"]

    # Load WFM encoder and actors
    encoder = WFMEncoder(env.obs_dim, cfg["wfm"]["embed_dim"])
    encoder.load_state_dict(torch.load(cfg["wfm"]["pretrained_encoder_path"], map_location=device))
    encoder.eval()

    actors = torch.load("wfm_mappo_actors.pt", map_location=device)
    print('actors', actors)
    print("Evaluating WFM‑MAPPO...")

    results = evaluate_policy(env, actors, encoder=encoder, episodes=50)
    print(results)

if __name__ == "__main__":
    main()