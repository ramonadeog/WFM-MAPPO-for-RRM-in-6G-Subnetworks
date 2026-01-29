# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:34:17 2026

@author: ra
"""

# wfmmarl_simulator/environment/geometry.py
import numpy as np

def uniform_points_in_square(L: float, N: int, rng: np.random.Generator) -> np.ndarray:
    """
    Sample N points uniformly in [0, L] x [0, L].
    Returns: (N, 2) array of (x, y).
    """
    return rng.uniform(0.0, L, size=(N, 2))

def uniform_point_in_disk(radius: float, rng: np.random.Generator) -> np.ndarray:
    """
    Sample a single point uniformly in a disk of given radius centered at (0,0).
    Uses inverse transform: r = R*sqrt(U), theta = 2*pi*V.
    Returns: (2,) vector (x, y).
    """
    u = rng.random()
    v = rng.random()
    r = radius * np.sqrt(u)
    theta = 2.0 * np.pi * v
    return np.array([r * np.cos(theta), r * np.sin(theta)])

def clamp_to_disk(pos: np.ndarray, radius: float) -> np.ndarray:
    """
    Project position pos onto/inside disk of radius 'radius' centered at (0,0).
    """
    r = np.linalg.norm(pos)
    if r <= radius or r == 0.0:
        return pos
    return pos * (radius / r)