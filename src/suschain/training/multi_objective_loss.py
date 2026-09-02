"""Sustainability-aware auxiliary loss (plan C3): moves sustainability from
a post-hoc filter (the base paper's approach, and this project's Phase 2
reasoning-layer-free baseline) into the training objective itself --
directly answering the base paper's own stated future work ("sustainability
objectives could be integrated directly within the training process
through constraint-aware or multi-objective loss formulations", §6).

    L = L_BCE(link) + lambda * E[predicted_prob(edge) * risk(dst)]

The penalty discourages the model from assigning high link-existence
probability to destination nodes with high sustainability risk, across
every edge in the training batch (not just positives) -- pushing down
confidence in risky recommendations generally, not merely making the
loss agree with which edges happen to be real.

Risk convention (unified across both datasets, higher = worse):
- AnonNet: normalized composite carbon-intensity score (already in
  enrichment/anonnet_sustainability.py, itself in a comparable [0, ~2]
  range before a final min-max normalization here).
- LODHalal: 1 - certification_coverage (a manufacturer that certifies
  none of its products has risk 1.0; a fully-certified one has risk 0.0).
"""
from __future__ import annotations

import torch


def sustainability_penalty(scores: torch.Tensor, dst_idx: torch.Tensor, risk_by_dst: torch.Tensor) -> torch.Tensor:
    """scores: raw logits for a batch of (src, dst) candidate edges.
    dst_idx: destination-node local indices for the same batch.
    risk_by_dst: [num_dst_nodes] risk score per destination node, higher = worse.
    """
    probs = torch.sigmoid(scores)
    risk = risk_by_dst[dst_idx]
    return (probs * risk).mean()


def normalize_risk(raw_risk: torch.Tensor) -> torch.Tensor:
    """Min-max normalize to [0, 1], NaN-safe (unmatched nodes get the
    dataset-mean risk rather than propagating NaN into the loss -- an
    explicit, documented choice, not a silent default)."""
    valid = raw_risk[~torch.isnan(raw_risk)]
    if len(valid) == 0:
        return torch.zeros_like(raw_risk)
    lo, hi = valid.min(), valid.max()
    normalized = (raw_risk - lo) / (hi - lo) if hi > lo else torch.full_like(raw_risk, 0.5)
    mean_risk = normalized[~torch.isnan(normalized)].mean() if torch.isnan(normalized).any() else normalized.mean()
    return torch.nan_to_num(normalized, nan=float(mean_risk))
