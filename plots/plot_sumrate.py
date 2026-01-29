# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:13:01 2026

@author: ra
"""

# wfmmarl_simulator/plots/plot_sumrate.py
import numpy as np
import matplotlib.pyplot as plt
from utils_plot import setup_plot, savefig

def plot_sumrate(results_dict, output="sumrate_comparison.png"):
    """
    results_dict example:
    {"WFM‑MAPPO": 52.3, "Raw‑MAPPO": 48.9, "MAD‑DQN": 46.5, "Random": 31.2}
    """

    labels = list(results_dict.keys())
    rates = [results_dict[k] for k in labels]

    setup_plot("Average Sum‑Rate", "Method", "Sum‑Rate (Mbps)")
    x = np.arange(len(labels))

    plt.bar(x, rates)
    plt.xticks(x, labels)
    savefig(output)

if __name__ == "__main__":
    example = {
      "WFM‑MAPPO": 52.3,
      "Raw‑MAPPO": 48.9,
      "MAD‑DQN":   46.5,
      "Random":    31.2
    }
    plot_sumrate(example)