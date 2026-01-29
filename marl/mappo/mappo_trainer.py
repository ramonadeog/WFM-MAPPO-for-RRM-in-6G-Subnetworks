# wfmmarl_simulator/marl/mappo/mappo_trainer.py

from __future__ import annotations
import torch
import torch.nn as nn
from torch.optim import Adam

from marl.common.gae import compute_gae
from marl.common.rollout_buffer import RolloutBuffer
from marl.mappo.utils import select_log_probs, entropy
from marl.mappo.ppo_loss import ppo_policy_loss


class MAPPOTrainer:
    """
    Multi-Agent PPO with:
      - centralized critic (observes joint latent state)
      - decentralized actors (per-agent policies)
      - optional pretrained WFM encoder for state abstraction
    """

    def __init__(
        self,
        env,
        actor_class,
        critic_class,
        encoder=None,
        embed_dim=16,
        actor_lr=3e-4,
        critic_lr=3e-4,
        gamma=0.99,
        lam=0.95,
        clip_eps=0.2,
        entropy_coef=0.01,
        value_coef=0.5,
        device="cpu"
    ):
        self.env = env
        self.N = env.cfg.N
        self.K = env.cfg.K

        self.encoder = encoder  # pretrained WFM encoder or None (raw-state)
        self.embed_dim = embed_dim
        self.device = torch.device(device)

        # Actors: one per agent (parameter sharing can be added later if desired)
        self.actors = nn.ModuleList([
            actor_class(embed_dim, self.K).to(self.device) for _ in range(self.N)
        ])

        # Centralized critic: input dim = N * embed_dim ; output = scalar V(s)
        self.critic = critic_class(embed_dim, self.N).to(self.device)

        # Optimizers
        self.actor_optim = Adam(self.actors.parameters(), lr=actor_lr)
        self.critic_optim = Adam(self.critic.parameters(), lr=critic_lr)

        # PPO hyperparameters
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef

        # Put into train mode by default
        self.actors.train()
        self.critic.train()

    # -------------------------------------------------------------
    # Raw obs -> latent (WFM encoder if provided)
    # -------------------------------------------------------------
    def encode_obs(self, obs):
        """
        obs: numpy array (N, obs_dim)
        returns: torch tensor (N, embed_dim) on device
        """
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if self.encoder is None:
            # Raw-state MAPPO: the "latent" is just the raw obs
            return obs_t.detach()
        else:
            with torch.no_grad():
                lat = self.encoder(obs_t)
            return lat.detach()

    # -------------------------------------------------------------
    # One PPO rollout of length T (collect data; no learning yet)
    # -------------------------------------------------------------
    def collect_rollout(self, T):
        buffer = RolloutBuffer(
            T=T,
            num_agents=self.N,
            embed_dim=self.embed_dim,
            device=self.device
        )

        obs = self.env.reset()
        for _ in range(T):
            latent = self.encode_obs(obs)  # (N, embed_dim)

            actions, log_probs, values = self._policy_step(latent)

            next_obs, rewards_raw, done, info = self.env.step(actions.detach().cpu().numpy())
            # (Optional) replace rewards_raw with CMDP/Lagrangian-shaped rewards before training

            rewards = torch.tensor(rewards_raw, dtype=torch.float32, device=self.device)
            dones = torch.zeros(self.N, dtype=torch.float32, device=self.device)  # no terminal in env; episodic boundaries set by trainer

            # detach refs before storing
            buffer.add(
                obs_latent=latent,
                actions=actions.detach(),
                log_probs=log_probs.detach(),
                rewards=rewards,
                dones=dones,
                values=values.detach()
            )
            obs = next_obs

        # Bootstrap final V(s_T)
        with torch.no_grad():
            last_latent = self.encode_obs(obs)  # (N, embed)
            joint_last = torch.cat([last_latent[i] for i in range(self.N)], dim=-1).unsqueeze(0)
            V_last = self.critic(joint_last).squeeze(0)  # scalar
            buffer.finish(V_last.detach().repeat(self.N))

        return buffer

    # -------------------------------------------------------------
    # Action sampling for all agents + centralized value (NO grad on critic here)
    # -------------------------------------------------------------
    def _policy_step(self, latent: torch.Tensor):
        """
        latent: (N, embed_dim) for the current time-step
        Returns:
          actions:   (N,) sampled discrete actions
          log_probs: (N,) log π_i(a_i | s_i)
          values:    (N,) replicated centralized V(s) for bookkeeping
        """
        # Centralized critic on joint latent (no grad during data collection)
        joint = torch.cat([latent[i] for i in range(self.N)], dim=-1).unsqueeze(0)  # (1, N*embed)
        with torch.no_grad():
            V_joint = self.critic(joint).squeeze(0)  # scalar
        values = V_joint.repeat(self.N)

        actions = []
        log_probs = []
        for i in range(self.N):
            policy_i = self.actors[i](latent[i:i+1])      # (1, K)
            dist_i = torch.distributions.Categorical(probs=policy_i)
            a_i = dist_i.sample()                         # (1,)
            actions.append(a_i.squeeze(0))
            log_probs.append(dist_i.log_prob(a_i).squeeze(0))

        actions = torch.stack(actions)      # (N,)
        log_probs = torch.stack(log_probs)  # (N,)
        return actions, log_probs, values

    # -------------------------------------------------------------
    # PPO update over the collected rollout
    # -------------------------------------------------------------
    def update(self, buffer, ppo_epochs=5, batch_size=256):
        T = buffer.T
        N = self.N
        E = self.embed_dim

        # Flatten everything
        obs_latent = buffer.obs_latent.reshape(T * N, E)     # (T*N, E)
        actions = buffer.actions.reshape(T * N)              # (T*N,)
        log_probs_old = buffer.log_probs.reshape(T * N)      # (T*N,)
        rewards = buffer.rewards.reshape(T, N)               # (T, N)
        dones = buffer.dones.reshape(T, N)                   # (T, N)
        values = buffer.values.reshape(T + 1, N)             # (T+1, N)

        # GAE and returns (refs only; detach to prevent graph retention)
        advantages, returns = compute_gae(
            rewards=rewards, values=values, dones=dones, gamma=self.gamma, lam=self.lam
        )
        advantages = advantages.reshape(T * N).detach()
        returns = returns.reshape(T * N).detach()

        # Advantage normalization
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Minibatch training
        total = T * N
        idxs = torch.randperm(total, device=self.device)

        last_policy_loss = torch.tensor(0.0, device=self.device)
        last_value_loss = torch.tensor(0.0, device=self.device)

        for _ in range(ppo_epochs):
            for start in range(0, total, batch_size):
                mb_idx = idxs[start:start + batch_size]  # (B,)
                mb_latent = obs_latent[mb_idx]           # (B, E)
                mb_actions = actions[mb_idx]             # (B,)

                # Detach reference terms
                mb_old_logp = log_probs_old[mb_idx].detach()
                mb_adv = advantages[mb_idx].detach()
                mb_rets = returns[mb_idx].detach()

                # Recover (t, i) for each sample (since rollout is flattened as T*N)
                t_idx = (mb_idx // N)                    # (B,)
                agent_idx = (mb_idx % N)                 # (B,)

                # ----- Centralized critic: evaluate ONCE per unique time step (with gradients) -----
                uniq_t, inv = torch.unique(t_idx, return_inverse=True)
                V_per_uniq = []
                for t_val in uniq_t.tolist():
                    joint_t = buffer.obs_latent[t_val].reshape(1, N * E)   # (1, N*E)
                    V_t = self.critic(joint_t).squeeze(0)                  # scalar (grad-enabled)
                    V_per_uniq.append(V_t)
                V_per_uniq = torch.stack(V_per_uniq)                       # (U,)
                V_batch = V_per_uniq[inv]                                  # (B,)

                # ----- Per-agent new log-probs & entropy -----
                new_logp = torch.empty_like(mb_old_logp)
                entropies = []

                for i in range(N):
                    mask = (agent_idx == i)
                    if torch.any(mask):
                        latent_i = mb_latent[mask]                 # (B_i, E)
                        actions_i = mb_actions[mask]               # (B_i,)
                        policy_i = self.actors[i](latent_i)        # (B_i, K)

                        logp_i = select_log_probs(policy_i, actions_i)  # (B_i,)
                        new_logp[mask] = logp_i

                        entropies.append(entropy(policy_i))        # scalar entropy (mean inside helper)

                ent = torch.stack(entropies).mean() if len(entropies) > 0 else torch.tensor(0.0, device=self.device)

                # ----- PPO losses -----
                policy_loss = ppo_policy_loss(new_logp, mb_old_logp, mb_adv, clip_eps=self.clip_eps)
                value_loss = torch.mean((V_batch - mb_rets) ** 2)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * ent

                # ----- Optimize -----
                self.actor_optim.zero_grad(set_to_none=True)
                self.critic_optim.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.actors.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)

                self.actor_optim.step()
                self.critic_optim.step()

                last_policy_loss = policy_loss.detach()
                last_value_loss = value_loss.detach()

        return {"actor": float(last_policy_loss), "critic": float(last_value_loss)}