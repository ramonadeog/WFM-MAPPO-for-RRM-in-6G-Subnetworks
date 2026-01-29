# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:08:27 2026

@author: ra
"""

# wfmmarl_simulator/experiments/train_wfm_mappo.py
import yaml
import torch

from environment.subnetwork_env import SubnetworkEnv, EnvConfig
from wfm.encoder import WFMEncoder
from marl.mappo.mappo_trainer import MAPPOTrainer
from marl.mappo.actor import MAPPOActor
from marl.mappo.critic import CentralizedCritic

def main():
    # Load config
    cfg = yaml.safe_load(open("experiments/config.yaml"))
    env_cfg = EnvConfig(**cfg["environment"])
    env = SubnetworkEnv(env_cfg)

    device = cfg["device"]

    # Load pretrained WFM encoder
    encoder = WFMEncoder(input_dim=env.obs_dim, embed_dim=cfg["wfm"]["embed_dim"])
    encoder.load_state_dict(torch.load(cfg["wfm"]["pretrained_encoder_path"], map_location=device))
    encoder.eval()

    trainer = MAPPOTrainer(
        env=env,
        actor_class=lambda emb, K: MAPPOActor(cfg["wfm"]["embed_dim"], env.cfg.K),
        critic_class=lambda emb, N: CentralizedCritic(cfg["wfm"]["embed_dim"], env.cfg.N),
        encoder=encoder,
        embed_dim=cfg["wfm"]["embed_dim"],
        device=device
    )

    steps = cfg["experiment"]["training_steps"]
    rollout_len = cfg["experiment"]["rollout_length"]
    ppo_epochs = cfg["experiment"]["ppo_epochs"]
    batch_size = cfg["experiment"]["mappo_batch_size"]

    for step in range(steps):
        buffer = trainer.collect_rollout(T=rollout_len)
        losses = trainer.update(buffer, ppo_epochs=ppo_epochs, batch_size=batch_size)

        if step % 20 == 0:
            print(f"[Step {step}] Actor={losses['actor']:.3f}  Critic={losses['critic']:.3f}")

    torch.save(trainer.actors.state_dict(), "wfm_mappo_actors.pt")
    torch.save(trainer.critic.state_dict(), "wfm_mappo_critic.pt")

    print("WFM‑MAPPO training complete!")

if __name__ == "__main__":
    main()