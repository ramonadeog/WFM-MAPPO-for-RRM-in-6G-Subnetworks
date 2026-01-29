# wfmmarl_simulator/marl/mappo/utils.py
import torch

def to_tensor(x, device=None, dtype=None):
    """
    Safe conversion to torch.Tensor with optional device/dtype.
    """
    t = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    if dtype is not None:
        t = t.to(dtype)
    if device is not None:
        t = t.to(device)
    return t


# -----------------------------------------------------------------------------
# Core helpers used by MAPPO
# -----------------------------------------------------------------------------

def joint_latent_tensor(latent_list):
    """
    Concatenate a list of per-agent latent tensors into a joint latent.

    Args:
        latent_list: list of tensors, each with shape (batch, embed_dim)
                     representing agents [0..N-1].

    Returns:
        joint: (batch, N*embed_dim) tensor created by concatenation
               along the last dimension.
    """
    if not isinstance(latent_list, (list, tuple)) or len(latent_list) == 0:
        raise ValueError("joint_latent_tensor expects a non-empty list/tuple of tensors.")
    # Ensure all have same batch size
    bsz = latent_list[0].shape[0]
    for t in latent_list:
        if t.shape[0] != bsz:
            raise ValueError("All latent tensors must share the same batch size.")
    return torch.cat(latent_list, dim=-1)


def select_log_probs(policy, actions):
    """
    Compute log π(a|s) for a categorical policy.

    Args:
        policy: (batch, K) tensor of action probabilities (each row sums to 1).
        actions: (batch,) tensor of integer actions in [0, K-1].

    Returns:
        log_probs: (batch,) tensor of log-probabilities for the chosen actions.
    """
    if policy.dim() != 2:
        raise ValueError("Policy tensor must be 2D: (batch, K).")
    if actions.dim() != 1:
        raise ValueError("Actions tensor must be 1D: (batch,).")
    if policy.shape[0] != actions.shape[0]:
        raise ValueError("Policy batch size must match actions batch size.")

    dist = torch.distributions.Categorical(probs=policy)
    return dist.log_prob(actions)


def entropy(policy):
    """
    Mean entropy of a categorical policy distribution.

    Args:
        policy: (batch, K) tensor of action probabilities.

    Returns:
        scalar tensor: mean entropy over the batch.
    """
    dist = torch.distributions.Categorical(probs=policy)
    return dist.entropy().mean()


# -----------------------------------------------------------------------------
# Optional convenience utilities (use as needed)
# -----------------------------------------------------------------------------

def build_joint_from_step(latent_step):
    """
    Convenience wrapper when you have a single time-step latent matrix.

    Args:
        latent_step: (N, embed_dim) per-agent latent at a single time step.

    Returns:
        joint: (1, N*embed_dim) for centralized critic.
    """
    if latent_step.dim() != 2:
        raise ValueError("Expected latent_step with shape (N, embed_dim).")
    return torch.cat([latent_step[i] for i in range(latent_step.shape[0])], dim=-1).unsqueeze(0)


def agent_time_index(flat_idx, num_agents):
    """
    Recover (t, i) from flattened indices when rollout is flattened as (T*N, ...).

    Args:
        flat_idx: (batch,) tensor of flattened indices in [0, T*N-1].
        num_agents: int, number of agents N.

    Returns:
        t_idx: (batch,) integer tensor of time indices in [0, T-1].
        agent_idx: (batch,) integer tensor of agent indices in [0, N-1].
    """
    if not isinstance(num_agents, int) or num_agents <= 0:
        raise ValueError("num_agents must be a positive integer.")
    if flat_idx.dim() != 1:
        raise ValueError("flat_idx must be a 1D tensor.")
    t_idx = flat_idx // num_agents
    agent_idx = flat_idx % num_agents
    return t_idx, agent_idx