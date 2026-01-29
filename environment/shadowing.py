# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:37:03 2026

@author: ra
"""

# wfmmarl_simulator/environment/shadowing.py
import numpy as np

class SpatialShadowingField:
    """
    Per-link log-normal shadowing in dB with exponential spatial correlation:
        Corr(Δd) = exp(-Δd / d_c)
    We model slow fading per RX (device) column: links (AP_j -> RX_i) update
    with the distance moved by RX_i at each time step.
    """

    def __init__(self, N: int, sigma_dB: float, decorrelation_d: float, rng: np.random.Generator):
        self.N = N
        self.sigma_dB = sigma_dB
        self.d_c = decorrelation_d
        self.rng = rng

        # Shadowing in dB, per link j->i (AP_j to RX_i), shape (N, N)
        self.S_dB = np.zeros((N, N), dtype=float)

    def reset(self):
        # initialize from N(0, sigma^2) per link
        self.S_dB = self.rng.normal(loc=0.0, scale=self.sigma_dB, size=(self.N, self.N))

    def update_with_receiver_motion(self, delta_d_per_rx: np.ndarray):
        """
        Update each column i using alpha_i = exp(-Δd_i / d_c):
            S_{:,i}(t) = alpha_i * S_{:,i}(t-1) + sqrt(1 - alpha_i^2) * sigma * z
        where z ~ N(0,1) i.i.d. per link in column i.
        """
        for i in range(self.N):
            delta = float(delta_d_per_rx[i])
            alpha = np.exp(-delta / max(self.d_c, 1e-12))
            noise = self.rng.normal(0.0, self.sigma_dB, size=(self.N,))
            self.S_dB[:, i] = alpha * self.S_dB[:, i] + np.sqrt(max(0.0, 1.0 - alpha ** 2)) * noise

    def get_shadowing_linear(self) -> np.ndarray:
        """
        Returns per-link multiplicative shadowing gain in linear power scale:
            G_shad = 10^(S_dB / 10)
        """
        return 10.0 ** (self.S_dB / 10.0)