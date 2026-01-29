# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:10:14 2026

@author: ra
"""

# wfmmarl_simulator/experiments/train_random.py
import yaml
import numpy as np

from environment.subnetwork_env import SubnetworkEnv, EnvConfig
from marl.baselines.random_policy import RandomPolicy

def main():
    cfg = yaml.safe_load(open("experiments/config.yaml"))
    env_cfg = EnvConfig(**cfg["environment"])
    env = SubnetworkEnv(env_cfg)

    random = RandomPolicy(env.cfg.N, env.cfg.K)
    episodes = cfg["experiment"]["training_steps"]

    for ep in range(episodes):
        obs = env.reset()
        done = False
        sum_rate = 0.0

        while not done:
            actions = random.act(obs)
            obs, r, done, info = env.step(actions)
            sum_rate += info["rates_bps"].sum()

        if ep % 20 == 0:
            print(f"[Episode {ep}] Sum-rate={sum_rate/1e6:.2f} Mbps")

if __name__ == "__main__":
    main()