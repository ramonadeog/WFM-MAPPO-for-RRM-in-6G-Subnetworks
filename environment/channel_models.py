# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:36:40 2026

@author: ra
"""

# wfmmarl_simulator/environment/channel_models.py
import numpy as np
from numpy import pi
from scipy.special import j0

C_LIGHT = 3.0e8

def ar1_jakes_coefficient(speed, fc_hz, Ts) -> float:
    """
    ρ = J0(2π f_D Ts), with f_D = v * fc / c.
    Coerces inputs to float to guard against string inputs.
    """
    try:
        v = float(speed)
        fc = float(fc_hz)
        Ts = float(Ts)
    except Exception as e:
        raise TypeError(
            f"ar1_jakes_coefficient expects numeric speed/fc_hz/Ts; got types "
            f"{type(speed)}, {type(fc_hz)}, {type(Ts)}"
        ) from e

    f_D = (v * fc) / C_LIGHT
    rho = float(j0(2.0 * pi * f_D * Ts))
    # Numerical safety for very large/small arguments
    return max(min(rho, 1.0), -1.0)

def complex_ar1_update(h_prev: np.ndarray, rho: float, rng: np.random.Generator) -> np.ndarray:
    """
    Complex circularly-symmetric AR(1) update:
        h_t = rho * h_{t-1} + sqrt(1 - rho^2) * w_t,  w_t ~ CN(0,1)
    Shapes preserved.
    """
    shape = h_prev.shape
    # CN(0,1) => real and imag ~ N(0, 1/2)
    w = (rng.normal(0.0, 1.0 / np.sqrt(2), size=shape) +
         1j * rng.normal(0.0, 1.0 / np.sqrt(2), size=shape))
    return rho * h_prev + np.sqrt(max(0.0, 1.0 - rho ** 2)) * w

def pathloss_linear_db(distance_m: np.ndarray,
                       pl0_dB: float,
                       d0_m: float,
                       alpha: float) -> np.ndarray:
    """
    Log-distance path loss (in dB):
        PL(d) = PL(d0) + 10*alpha*log10(d/d0)
    Returns PL in dB for each distance.
    """
    d = np.maximum(distance_m, 1e-3)
    return pl0_dB + 10.0 * alpha * np.log10(d / d0_m)

def db_to_linear(db_vals: np.ndarray | float) -> np.ndarray | float:
    return 10.0 ** (np.asarray(db_vals) / 10.0)

def linear_to_db(lin_vals: np.ndarray | float) -> np.ndarray | float:
    lin = np.asarray(lin_vals)
    lin = np.maximum(lin, 1e-30)
    return 10.0 * np.log10(lin)