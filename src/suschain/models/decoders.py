"""Link-prediction decoder heads, swappable via HeteroLinkPredictor's
`decoder_kind`. Plan Phase 1 hypothesis: "I suspect the decoder is the
bottleneck, not the encoder" -- specifically tested here against the
observed GAT ranking anomaly (docs/PHASE1_BASELINES.md): GAT's embeddings
separate real-vs-random pairs well under plain dot-product scoring
(classification metrics ~match SAGE/RGCN) but rank poorly against the full
candidate catalog, which is consistent with a dot-product decoder being a
poor fit for GATv2's embedding geometry (attention-weighted embeddings
aren't guaranteed to live on a scale/direction where inner product tracks
relevance the way a bilinear or MLP decoder could learn to correct for).
"""
from __future__ import annotations

import torch
from torch import nn


class DotProductDecoder(nn.Module):
    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        return (z_src * z_dst).sum(dim=-1)


class BilinearDecoder(nn.Module):
    """score = z_src^T W z_dst -- a learned linear transform before the dot
    product, strictly more expressive than plain dot-product and cheap."""

    def __init__(self, dim: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(dim, dim))
        nn.init.xavier_uniform_(self.W)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bi,ij,bj->b", z_src, self.W, z_dst)


class MLPDecoder(nn.Module):
    """score = MLP([z_src ; z_dst]) -- can learn arbitrary (non-bilinear)
    interactions between the two embeddings, at the cost of more parameters
    and losing the closed-form full-catalog scoring speed-up (dot product
    lets us score against ALL candidates via one matmul; an MLP decoder
    needs one forward pass per candidate, which is still fine at these
    graph sizes but is the reason dot-product is the default elsewhere)."""

    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.ReLU(), nn.Linear(hidden, 1)
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_src, z_dst], dim=-1)).squeeze(-1)

    def score_all(self, z_src_one: torch.Tensor, z_dst_all: torch.Tensor) -> torch.Tensor:
        """Score one source embedding against ALL destination embeddings
        (used by ranking eval, mirroring the dot-product path's `z_src @
        z_dst_all.T` shortcut, just without the matmul speed-up)."""
        src_rep = z_src_one.unsqueeze(0).expand(z_dst_all.shape[0], -1)
        return self.net(torch.cat([src_rep, z_dst_all], dim=-1)).squeeze(-1)


def build_decoder(kind: str, dim: int) -> nn.Module:
    if kind == "dot":
        return DotProductDecoder()
    if kind == "bilinear":
        return BilinearDecoder(dim)
    if kind == "mlp":
        return MLPDecoder(dim)
    raise ValueError(f"unknown decoder kind: {kind}")
