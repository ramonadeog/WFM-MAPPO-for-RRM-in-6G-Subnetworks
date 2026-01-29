# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:12:06 2026

@author: ra
"""

# wfmmarl_simulator/plots/plot_convergence.py
import matplotlib.pyplot as plt
import numpy as np
from utils_plot import setup_plot, savefig

def plot_convergence(log_files, labels, output="convergence_curve.png"):
    """
    log_files: list of .npy files with reward_per_step arrays
    labels: list of labels (same length)
    """
    setup_plot("Training Convergence", "Training Steps", "Average Reward")

    for file, label in zip(log_files, labels):
        rewards = np.load(file)      # shape: (steps,)
        smoothed = np.convolve(rewards, np.ones(20)/20, mode="same")
        plt.plot(smoothed, label=label)

    plt.legend()
    savefig(output)

if __name__ == "__main__":
    log_files = [
        "logs/wfm_mappo_rewards.npy",
        "logs/raw_mappo_rewards.npy",
        "logs/maddqn_rewards.npy",
    ]
    labels = ["WFM‑MAPPO", "Raw‑MAPPO", "MAD‑DQN"]
    plot_convergence(log_files, labels)