# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:09:00 2026

@author: ra
"""

# wfmmarl_simulator/experiments/train_raw_mappo.py
import yaml
import torch

from environment.subnetwork_env import SubnetworkEnv, EnvConfig
from marl.mappo.actor import MAPPOActor
from marl.mappo.critic import CentralizedCritic
from marl.mappo.mappo_trainer import MAPPOTrainer

def main():
    cfg = yaml.safe_load(open("experiments/config.yaml"))
    env_cfg = EnvConfig(**cfg["environment"])
    env = SubnetworkEnv(env_cfg)

    raw_dim = env.obs_dim
    device = cfg["device"]

    trainer = MAPPOTrainer(
        env=env,
        actor_class=lambda emb, K: MAPPOActor(raw_dim, env.cfg.K),
        critic_class=lambda emb, N: CentralizedCritic(raw_dim, env.cfg.N),
        encoder=None,       # <-- raw-state baseline
        embed_dim=raw_dim,
        device=device
    )

    steps = cfg["experiment"]["training_steps"]
    rollout_len = cfg["experiment"]["rollout_length"]

    for step in range(steps):
        buffer = trainer.collect_rollout(T=rollout_len)
        trainer.update(buffer)

        if step % 20 == 0:
            print(f"[Step {step}] Raw-MAPPO training step complete.")

    torch.save(trainer.actors.state_dict(), "raw_mappo_actors.pt")
    torch.save(trainer.critic.state_dict(), "raw_mappo_critic.pt")
    print("Raw‑MAPPO training complete!")

if __name__ == "__main__":
    main()