# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:12:34 2026

@author: ra
"""

# wfmmarl_simulator/plots/plot_violations.py
import numpy as np
import matplotlib.pyplot as plt
from utils_plot import setup_plot, savefig

def plot_violations(results_dict, output="violations.png"):
    """
    results_dict example:
    {
      "WFM‑MAPPO": {"interf": 0.018, "qos": 0.015},
      "Raw‑MAPPO": {"interf": 0.071, "qos": 0.068},
      "MAD‑DQN":   {"interf": 0.124, "qos": 0.109},
      "Random":    {"interf": 0.305, "qos": 0.287}
    }
    """

    labels = list(results_dict.keys())
    interf = [results_dict[k]["interf"] for k in labels]
    qos    = [results_dict[k]["qos"]    for k in labels]

    x = np.arange(len(labels))
    w = 0.35

    setup_plot("Constraint Violations", "Method", "Violation Rate")

    plt.bar(x - w/2, interf, w, label="Interference Violations")
    plt.bar(x + w/2, qos, w, label="QoS Violations")

    plt.xticks(x, labels)
    plt.legend()
    savefig(output)

if __name__ == "__main__":
    example = {
        "WFM‑MAPPO": {"interf": 0.018, "qos": 0.015},
        "Raw‑MAPPO": {"interf": 0.071, "qos": 0.068},
        "MAD‑DQN":   {"interf": 0.124, "qos": 0.109},
        "Random":    {"interf": 0.305, "qos": 0.287},
    }
    plot_violations(example)