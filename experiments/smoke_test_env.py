# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:37:56 2026

@author: ra
"""

# wfmmarl_simulator/experiments/smoke_test_env.py
import numpy as np
from environment.subnetwork_env import SubnetworkEnv, EnvConfig

def main():
    cfg = EnvConfig(
        N=5, K=10, L_m=50.0, cell_radius_m=2.0,
        fc_hz=3.5e9, noise_dBm=-100.0, p_tx_W=0.1, bandwidth_Hz=10e6,
        pl0_dB=46.0, d0_m=1.0, pathloss_exp=2.2,
        sigma_shadow_dB=6.0, shad_decorr_m=10.0,
        speed_mps=3.0, Ts_s=1e-3,
        include_buffer=True, mean_arrival_bps=2e6,
        seed=2026
    )
    env = SubnetworkEnv(cfg)
    obs = env.reset()
    print("Initial obs shape:", obs.shape, "obs_dim:", env.obs_dim)

    T = 20
    for t in range(T):
        # Random channel selection baseline (for smoke test)
        actions = np.random.randint(0, cfg.K, size=cfg.N)
        obs, reward, done, info = env.step(actions)
        print(obs.shape)
        sumrate_mbps = info["rates_bps"].sum() / 1e6
        print(f"t={t:02d}  actions={actions}  sum-rate={sumrate_mbps:.2f} Mbps")

if __name__ == "__main__":
    main()