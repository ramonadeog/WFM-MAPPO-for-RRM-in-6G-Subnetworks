# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:36:07 2026

@author: ra
"""

# wfmmarl_simulator/environment/mobility.py
import numpy as np

class DiskMobility2D:
    """
    Constant-speed mobility within a disk centered at the AP.
    - Each device moves with speed v and heading theta.
    - Reflects at the disk boundary with specular reflection.
    """

    def __init__(self, N: int, radius: float, speed: float, Ts: float, rng: np.random.Generator):
        self.N = N
        self.radius = radius
        self.speed = speed
        self.Ts = Ts
        self.rng = rng

        # Relative device positions (w.r.t. each subnetwork AP at origin)
        self.rel_pos = np.zeros((N, 2), dtype=float)
        # Heading angles per device
        self.heading = np.zeros(N, dtype=float)

    def reset(self, init_rel_positions: np.ndarray | None = None):
        if init_rel_positions is not None:
            self.rel_pos = init_rel_positions.copy()
        # random headings in [0, 2π)
        self.heading = self.rng.uniform(0.0, 2.0 * np.pi, size=self.N)

    def step(self):
        """
        Move all devices by v * Ts along current heading.
        Handle disk boundary via specular reflection.
        Returns:
            delta_d: (N,) distance moved per device in this step.
        """
        vdt = self.speed * self.Ts
        delta = np.vstack([vdt * np.cos(self.heading), vdt * np.sin(self.heading)]).T
        new_pos = self.rel_pos + delta
        delta_d = np.full(self.N, vdt)

        # Reflect at boundary if outside disk
        radii = np.linalg.norm(new_pos, axis=1)
        outside = radii > self.radius
        if np.any(outside):
            # For each outside point, reflect heading about the tangent at the boundary
            for i in np.where(outside)[0]:
                # move to boundary point
                p = new_pos[i]
                norm = p / np.linalg.norm(p)
                # Decompose velocity along normal/tangent to boundary
                v = np.array([np.cos(self.heading[i]), np.sin(self.heading[i])]) * vdt
                v_n = np.dot(v, norm) * norm
                v_t = v - v_n
                v_ref = v_t - v_n  # reflect normal component
                # place point on boundary with small epsilon inside
                new_pos[i] = norm * (self.radius - 1e-9)
                # update heading according to reflected velocity
                self.heading[i] = np.arctan2(v_ref[1], v_ref[0])

        self.rel_pos = new_pos
        return delta_d
