# -*- coding: utf-8 -*-
"""
Created on Wed Jan 28 12:47:38 2026

@author: ra
"""

# wfmmarl_simulator/wfm/contrastive_loss.py
import torch
import torch.nn.functional as F

# wfmmarl_simulator/wfm/contrastive_loss.py
import torch
import torch.nn.functional as F

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    """
    NT-Xent (SimCLR) loss.
    z1: (B, D) embeddings for view 1
    z2: (B, D) embeddings for view 2
    Returns scalar loss.
    """
    assert z1.dim() == 2 and z2.dim() == 2 and z1.size(0) == z2.size(0)
    B = z1.size(0)

    # L2-normalize
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    # Concatenate along batch: (2B, D)
    z = torch.cat([z1, z2], dim=0)

    # Cosine similarity matrix: (2B, 2B)
    sim = torch.matmul(z, z.T) / temperature

    # Mask self-similarity
    diag = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(diag, -float('inf'))

    # For each i in [0, 2B-1], the positive index is i^B (swap pair)
    pos_idx = torch.arange(2 * B, device=z.device)
    pos_idx = (pos_idx + B) % (2 * B)  # partner indices

    # The positive logit for row i is at column pos_idx[i]
    pos_logits = sim[torch.arange(2 * B, device=z.device), pos_idx].unsqueeze(1)  # (2B, 1)

    # Denominator: logsumexp over all (2B-1) non-self entries
    # Equivalent CE with logits = [pos_logit, all_negatives] is:
    #   CE( logits=[pos, others], target=0 )
    # So we build logits cat as [pos, all columns except self and pos]
    # But for numeric stability, we can compute cross-entropy with the whole row and set target=pos_idx
    # if we had not removed self. Since we removed self, we use a more explicit form:

    # Build labels and logits for standard cross_entropy:
    # We'll rebuild per-row logits = [pos, rest] and target=0
    # More efficient alternative: CE over full row using log_softmax + index; here we keep it explicit and clear.

    # Gather negatives: remove self and positive
    mask_pos = torch.zeros_like(sim, dtype=torch.bool)
    mask_pos[torch.arange(2 * B, device=z.device), pos_idx] = True
    mask_all = ~(diag | mask_pos)  # all negatives

    # Build per-row negative lists
    neg_logits_list = []
    for i in range(2 * B):
        neg_logits_list.append(sim[i][mask_all[i]])

    # Stack negatives into (2B, 2B-2)
    neg_logits = torch.stack(neg_logits_list, dim=0)

    # Compose logits: [pos | negatives] => shape (2B, 1 + 2B-2) = (2B, 2B-1)
    logits = torch.cat([pos_logits, neg_logits], dim=1)

    # Targets: positive is always at index 0 in our composed logits
    targets = torch.zeros(2 * B, dtype=torch.long, device=z.device)

    return F.cross_entropy(logits, targets)