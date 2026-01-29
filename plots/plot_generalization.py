# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:13:37 2026

@author: ra
"""

# wfmmarl_simulator/plots/plot_generalization.py
import numpy as np
import matplotlib.pyplot as plt
from utils_plot import setup_plot, savefig

def plot_generalization(results, output="generalization.png"):
    """
    results: dict of lists
    {
      "WFM‑MAPPO": [52.3, 50.1, 49.8, ...],
      "Raw‑MAPPO": [...],
      "MAD‑DQN":   [...],
    }
    """

    setup_plot("Generalization Across Unseen Topologies",
               "Test Scenario Index", "Sum‑Rate (Mbps)")

    for name, values in results.items():
        plt.plot(values, label=name, marker='o')

    plt.legend()
    savefig(output)

if __name__ == "__main__":
    example = {
        "WFM‑MAPPO": [52, 51, 49, 50, 52],
        "Raw‑MAPPO": [48, 47, 43, 46, 45],
        "MAD‑DQN":   [46, 44, 40, 41, 42],
    }
    plot_generalization(example)