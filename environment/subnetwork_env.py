# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:37:27 2026

@author: ra
"""

# wfmmarl_simulator/environment/subnetwork_env.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass

from .geometry import uniform_points_in_square, uniform_point_in_disk
from .mobility import DiskMobility2D
from .channel_models import (
    ar1_jakes_coefficient, complex_ar1_update,
    pathloss_linear_db, db_to_linear
)
from .shadowing import SpatialShadowingField

# ... keep previous imports and the EnvConfig dataclass fields ...

@dataclass
class EnvConfig:
    # Topology
    N: int = 5
    K: int = 10
    L_m: float = 50.0
    cell_radius_m: float = 2.0

    # PHY
    fc_hz: float = 3.5e9
    noise_dBm: float = -100.0
    p_tx_W: float = 0.1
    bandwidth_Hz: float = 10e6

    # Channel models
    pl0_dB: float = 46.0
    d0_m: float = 1.0
    pathloss_exp: float = 2.2
    sigma_shadow_dB: float = 6.0
    shad_decorr_m: float = 10.0

    # Mobility and dynamics
    speed_mps: float = 3.0
    Ts_s: float = 1e-3

    # Constraints
    I_th_W: float = 1e-9
    R_min_bps: float = 1e6

    # Observation options
    include_buffer: bool = True
    mean_arrival_bps: float = 2e6

    # RNG
    seed: int = 42

    def __post_init__(self):
        # Cast numeric-like strings to proper types
        float_fields = [
            "L_m", "cell_radius_m",
            "fc_hz", "noise_dBm", "p_tx_W", "bandwidth_Hz",
            "pl0_dB", "d0_m", "pathloss_exp",
            "sigma_shadow_dB", "shad_decorr_m",
            "speed_mps", "Ts_s",
            "I_th_W", "R_min_bps", "mean_arrival_bps",
        ]
        int_fields = ["N", "K", "seed"]

        for name in float_fields:
            val = getattr(self, name)
            try:
                setattr(self, name, float(val))
            except Exception:
                raise TypeError(f"EnvConfig.{name} must be numeric, got {val!r} ({type(val)})")

        for name in int_fields:
            val = getattr(self, name)
            try:
                setattr(self, name, int(val))
            except Exception:
                raise TypeError(f"EnvConfig.{name} must be int, got {val!r} ({type(val)})")

class SubnetworkEnv:
    """
    Multi-cell interference environment.
    - AP_j at center position C_j.
    - RX_i is UE_i located relative to its AP within radius R, moves within disk.
    - Each step:
        * Update small-scale fading with AR(1) Jakes approximation.
        * Update shadowing with exponential correlation using UE travel distance.
        * Compute per-link large-scale gains (fixed pathloss over episode, update shadowing).
        * Apply channel selection actions (integers in [0, K-1]).
        * Compute interference, SINR, and rates per UE_i.
        * Update optional buffers.

    SINR_i = P_tx * |h_{ii}|^2 * G_{ii}  / ( sum_{j != i, a_j=a_i} P_tx * |h_{ji}|^2 * G_{ji} + N0 )
    Rate_i = B * log2(1 + SINR_i)
    (Shannon approximation)  [1](https://aaudk-my.sharepoint.com/personal/ra_es_aau_dk/Documents/Microsoft%20Copilot%20Chat%20Files/IEEE_IoT_Journal_WFMRL_Subnetworks.pdf)
    """

    def __init__(self, cfg: EnvConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # AP centers (global)
        self.ap_xy = None  # shape (N, 2)
        # UE positions relative to own AP (local, inside disk)
        self.ue_rel_xy = None  # shape (N, 2)

        # Mobility model per UE (relative to AP frame)
        self.mobility = DiskMobility2D(
            N=cfg.N, radius=cfg.cell_radius_m, speed=cfg.speed_mps, Ts=cfg.Ts_s, rng=self.rng
        )

        # Small-scale fading per link j->i (complex), AR(1)
        self.H = np.zeros((cfg.N, cfg.N), dtype=np.complex128)
        self.rho = ar1_jakes_coefficient(cfg.speed_mps, cfg.fc_hz, cfg.Ts_s)

        # Per-link shadowing (log-normal in dB)
        self.shadow_field = SpatialShadowingField(
            N=cfg.N, sigma_dB=cfg.sigma_shadow_dB, decorrelation_d=cfg.shad_decorr_m, rng=self.rng
        )

        # Per-link path loss (dB) (fixed during episode)
        self.PL_dB = np.zeros((cfg.N, cfg.N), dtype=float)

        # Buffers (optional)
        self.buffers_bits = np.zeros(cfg.N, dtype=float)

        # Previous actions (for observations)
        self.prev_actions = np.zeros(cfg.N, dtype=int)

        # Noise in Watts
        self.noise_W = 1e-3 * 10.0 ** (cfg.noise_dBm / 10.0)

        # Step counter
        self.t = 0

    # ---------- Reset & Episode Initialization ----------

    def reset(self):
        self.t = 0
        # Place APs in [0,L]^2
        self.ap_xy = uniform_points_in_square(self.cfg.L_m, self.cfg.N, self.rng)
        # Place each UE uniformly in disk around its AP
        self.ue_rel_xy = np.vstack([uniform_point_in_disk(self.cfg.cell_radius_m, self.rng)
                                    for _ in range(self.cfg.N)])
        self.mobility.reset(init_rel_positions=self.ue_rel_xy.copy())
        self.prev_actions[:] = 0

        # Initialize small-scale fading as CN(0,1)
        self.H = (self.rng.normal(0.0, 1.0/np.sqrt(2), size=(self.cfg.N, self.cfg.N)) +
                  1j * self.rng.normal(0.0, 1.0/np.sqrt(2), size=(self.cfg.N, self.cfg.N)))

        # Initialize shadowing
        self.shadow_field.reset()

        # Compute fixed path loss distances (AP_j to RX_i) using INITIAL geometry
        # Note: path loss is frozen for the entire episode (slow mobility assumption)
        self.PL_dB = self._compute_pathloss_dB_matrix()

        # Reset buffers
        self.buffers_bits[:] = 0.0

        # Build first observation
        obs = self._build_observations(
            sinr_lin=np.zeros(self.cfg.N),
            interf_W=np.zeros(self.cfg.N)
        )
        return obs

    def _compute_pathloss_dB_matrix(self) -> np.ndarray:
        """
        Using INITIAL global geometry (AP centers fixed, UE initial positions),
        compute per-link AP_j -> RX_i distances and pathloss (in dB).
        Pathloss remains constant during the episode.
        """
        # UE global coordinates
        ue_xy_global = self.ap_xy + self.ue_rel_xy
        # Distances from AP_j to UE_i
        diff = ue_xy_global[None, :, :] - self.ap_xy[:, None, :]   # shape (N, N, 2) (AP_j -> UE_i)
        dists = np.linalg.norm(diff, axis=2)                       # (N, N)
        # Path loss in dB
        PL_dB = pathloss_linear_db(
            distance_m=dists, pl0_dB=self.cfg.pl0_dB, d0_m=self.cfg.d0_m, alpha=self.cfg.pathloss_exp
        )
        return PL_dB

    # ---------- Step ----------

    def step(self, actions: np.ndarray):
        """
        actions: int array of shape (N,), each in [0, K-1]
        Returns: obs, reward_dict, done, info
        (Rewards are not computed here; this env returns metrics needed by RL.)
        """
        self.t += 1
        actions = np.asarray(actions, dtype=int)
        assert actions.shape == (self.cfg.N,)
        assert np.all((0 <= actions) & (actions < self.cfg.K))

        # 1) Mobility: move UEs in their disks (relative to AP)
        delta_d = self.mobility.step()
        # keep ue_rel_xy consistent with mobility state
        self.ue_rel_xy = self.mobility.rel_pos.copy()

        # 2) Update shadowing with per-RX distance moved
        self.shadow_field.update_with_receiver_motion(delta_d)

        # 3) Update small-scale fading (AR(1) per link)
        self.H = complex_ar1_update(self.H, self.rho, self.rng)

        # 4) Compute per-link total large-scale gain: G_ls = (PL + Shadowing)
        # pathloss fixed (dB), shadowing time-varying (dB) -> sum in dB, then convert to linear
        G_shad_lin = self.shadow_field.get_shadowing_linear()         # (N, N)
        PL_lin = 1.0 / db_to_linear(self.PL_dB)                       # path *loss* to linear attenuation
        G_lin = PL_lin * G_shad_lin                                   # (N, N): total large-scale linear gain

        # 5) Compute received desired power & interference per UE
        # Channel power |h|^2 ~ Exp(1) via complex CN(0,1)
        H_pow = np.abs(self.H) ** 2                                   # (N, N), entry j->i
        P_tx = self.cfg.p_tx_W
        noise = self.noise_W

        # For UE i, desired from AP i, interference from AP j!=i using same channel
        desired_W = P_tx * H_pow[np.arange(self.cfg.N), np.arange(self.cfg.N)] * G_lin[np.arange(self.cfg.N), np.arange(self.cfg.N)]

        interf_W = np.zeros(self.cfg.N, dtype=float)
        for i in range(self.cfg.N):
            same_ch_tx = np.where((np.arange(self.cfg.N) != i) & (actions == actions[i]))[0]
            if len(same_ch_tx) > 0:
                interf_terms = P_tx * H_pow[same_ch_tx, i] * G_lin[same_ch_tx, i]
                interf_W[i] = np.sum(interf_terms)

        sinr_lin = desired_W / (interf_W + noise)
        # Rate per UE using Shannon: R_i = B * log2(1 + SINR_i)  [1](https://aaudk-my.sharepoint.com/personal/ra_es_aau_dk/Documents/Microsoft%20Copilot%20Chat%20Files/IEEE_IoT_Journal_WFMRL_Subnetworks.pdf)
        rates_bps = self.cfg.bandwidth_Hz * np.log2(1.0 + sinr_lin)

        # 6) Update buffers if enabled
        if self.cfg.include_buffer:
            arrivals = self.rng.exponential(self.cfg.mean_arrival_bps * self.cfg.Ts_s, size=self.cfg.N)
            serviced = rates_bps * self.cfg.Ts_s
            self.buffers_bits = np.maximum(0.0, self.buffers_bits + arrivals - serviced)

        # Build observations
        obs = self._build_observations(sinr_lin=sinr_lin, interf_W=interf_W)
        self.prev_actions = actions.copy()

        # No terminal condition (episodic control decided by outer loop)
        done = False
        info = {
            "rates_bps": rates_bps,
            "sinr_lin": sinr_lin,
            "interf_W": interf_W,
            "desired_W": desired_W,
            "G_lin": G_lin,
            "H_pow": H_pow
        }
        # No default reward here; MARL trainer will compute CMDP/Lagrangian rewards.
        reward = np.zeros(self.cfg.N, dtype=float)
        return obs, reward, done, info

    # ---------- Observations ----------

    def _build_observations(self, sinr_lin: np.ndarray, interf_W: np.ndarray) -> np.ndarray:
        """
        Local observation for each agent i:
            [ SINR_i (dB), Interf_i (dBm), prev_channel_i, buffer_i (bits, optional) ]
        """
        obs_list = []
        for i in range(self.cfg.N):
            sinr_dB = 10.0 * np.log10(max(sinr_lin[i], 1e-30))
            interf_dBm = 10.0 * np.log10(max(interf_W[i], 1e-30)) + 30.0
            core = [sinr_dB, interf_dBm, float(self.prev_actions[i])]
            if self.cfg.include_buffer:
                core.append(self.buffers_bits[i])
            obs_list.append(np.array(core, dtype=float))
        return np.vstack(obs_list)

    # ---------- Utility ----------

    @property
    def obs_dim(self) -> int:
        return 3 + (1 if self.cfg.include_buffer else 0)

    def action_dim(self) -> int:
        return self.cfg.K