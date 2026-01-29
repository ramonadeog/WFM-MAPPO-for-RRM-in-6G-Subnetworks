# wfmmarl_simulator/marl/mappo/ppo_loss.py
import torch

def ppo_policy_loss(
    new_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_eps: float = 0.2
) -> torch.Tensor:
    """
    Standard PPO clipped surrogate objective (to be MINIMIZED).
    All tensors are aligned elementwise: shape (batch,)

    L_policy = -E[min( r_t * A_t,
                       clip(r_t, 1-eps, 1+eps) * A_t )]
    where r_t = exp(new_logp - old_logp)
    """
    assert new_log_probs.shape == old_log_probs.shape == advantages.shape
    ratio = torch.exp(new_log_probs - old_log_probs)
    surr1 = ratio * advantages
    surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    return -torch.mean(torch.min(surr1, surr2))