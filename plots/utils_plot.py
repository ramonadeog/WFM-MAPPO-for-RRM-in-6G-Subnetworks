# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 13:11:36 2026

@author: ra
"""

# wfmmarl_simulator/plots/utils_plot.py
import matplotlib.pyplot as plt
import numpy as np

def setup_plot(title="", xlabel="", ylabel="", grid=True):
    plt.figure(figsize=(7, 5))
    plt.title(title, fontsize=14)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if grid:
        plt.grid(True, linestyle="--", alpha=0.6)

def savefig(path):
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    print(f"Saved figure → {path}")