# wfmmarl_simulator/marl/baselines/maddqn/maddqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .networks import DQNNetwork
from .replay_buffer import ReplayBuffer


class MAD_DQN:
    """
    Multi-agent independent DQN (IDQN) baseline.
    One Q-network per agent; decentralized training with ε-greedy exploration.
    """

    def __init__(
        self,
        env,
        buffer_capacity=50000,
        batch_size=64,
        gamma=0.99,
        lr=1e-3,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.999,
        tau=0.01,             # soft target update
        device="cpu",
    ):
        self.env = env
        self.N = env.cfg.N
        self.K = env.cfg.K
        self.obs_dim = env.obs_dim

        self.device = torch.device(device)

        # Replay buffer (stores per-agent transitions)
        self.buffer = ReplayBuffer(buffer_capacity, self.obs_dim, self.N)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # Per-agent Q-networks and targets
        self.qnets = nn.ModuleList([
            DQNNetwork(self.obs_dim, self.K).to(self.device) for _ in range(self.N)
        ])
        self.target_qnets = nn.ModuleList([
            DQNNetwork(self.obs_dim, self.K).to(self.device) for _ in range(self.N)
        ])
        for i in range(self.N):
            self.target_qnets[i].load_state_dict(self.qnets[i].state_dict())

        self.optimizers = [
            optim.Adam(self.qnets[i].parameters(), lr=lr) for i in range(self.N)
        ]

        # ε-greedy exploration
        self.epsilon = float(epsilon_start)
        self.epsilon_end = float(epsilon_end)
        self.epsilon_decay = float(epsilon_decay)

        # Loss
        self.criterion = nn.SmoothL1Loss()  # Huber loss

    # -------------------------------------------------------------
    # ε-greedy action selection (decentralized)
    # -------------------------------------------------------------
    def act(self, obs):
        """
        obs: numpy array (N, obs_dim)
        returns: numpy array (N,) discrete actions in [0, K-1]
        """
        actions = np.zeros(self.N, dtype=np.int64)
        for i in range(self.N):
            if np.random.rand() < self.epsilon:
                actions[i] = np.random.randint(0, self.K)
            else:
                with torch.no_grad():
                    o = torch.tensor(obs[i], dtype=torch.float32, device=self.device).unsqueeze(0)  # (1, obs_dim)
                    qvals = self.qnets[i](o)  # (1, K)
                    actions[i] = torch.argmax(qvals, dim=1).item()
        return actions

    # -------------------------------------------------------------
    # One gradient step using a minibatch from replay buffer
    # -------------------------------------------------------------
    def train_step(self):
        if self.buffer.size < self.batch_size:
            return

        # Sample a batch and move to device/dtypes
        obs_b, act_b, rew_b, next_obs_b, done_b = self.buffer.sample(self.batch_size)
        obs_b      = obs_b.to(self.device).float()        # (B, N, obs_dim)
        next_obs_b = next_obs_b.to(self.device).float()   # (B, N, obs_dim)
        act_b      = act_b.to(self.device).long()         # (B, N)
        rew_b      = rew_b.to(self.device).float()        # (B, N)
        done_b     = done_b.to(self.device).float()       # (B, N), 1.0 if terminal, else 0.0

        # Per-agent update
        for i in range(self.N):
            # Current Q(s,a) for agent i
            qvals = self.qnets[i](obs_b[:, i])            # (B, K)
            q_taken = qvals.gather(1, act_b[:, i:i+1])    # (B, 1)

            # Target Q using next state
            with torch.no_grad():
                target_qvals = self.target_qnets[i](next_obs_b[:, i])    # (B, K)
                max_next_q   = torch.max(target_qvals, dim=1, keepdim=True)[0]  # (B, 1)
                target = rew_b[:, i:i+1] + self.gamma * (1.0 - done_b[:, i:i+1]) * max_next_q  # (B, 1)

            # Loss and optimizer step
            loss = self.criterion(q_taken, target)
            self.optimizers[i].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.qnets[i].parameters(), 1.0)
            self.optimizers[i].step()

        # ε decay
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # -------------------------------------------------------------
    # Soft target network update
    # -------------------------------------------------------------
    def update_targets(self):
        with torch.no_grad():
            for i in range(self.N):
                for p, p_targ in zip(self.qnets[i].parameters(), self.target_qnets[i].parameters()):
                    p_targ.data.copy_(self.tau * p.data + (1.0 - self.tau) * p_targ.data)