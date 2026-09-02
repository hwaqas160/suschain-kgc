"""Filtered ranking metrics (MRR, Hits@k) for link prediction.

This is the concrete fix for plan limitation B5: the base paper never
states its candidate-set size or whether ranking is filtered (excluding
other known-true edges from the candidate pool) or raw. "Filtered" is
standard KG-completion practice (Bordes et al. 2013 onward) and is what's
implemented here -- each test edge's true tail is ranked against ALL other
destination-type nodes, with any OTHER known-positive tail for that source
masked out of the candidate pool before ranking (so the model isn't
penalized for correctly also scoring a different true relationship highly).
"""
from __future__ import annotations

import numpy as np
import torch


def _score_row(z_src_one: torch.Tensor, z_dst_all: torch.Tensor, decoder) -> torch.Tensor:
    """Score one source embedding against ALL destination embeddings.
    decoder=None -> dot product (fast matmul path, the default everywhere
    this was used before decoders.py existed). Otherwise uses the
    decoder's own `score_all` if present (see decoders.MLPDecoder), else
    falls back to broadcasting `decoder(src_rep, z_dst_all)`."""
    if decoder is None:
        return z_dst_all @ z_src_one
    if hasattr(decoder, "score_all"):
        return decoder.score_all(z_src_one, z_dst_all)
    src_rep = z_src_one.unsqueeze(0).expand(z_dst_all.shape[0], -1)
    return decoder(src_rep, z_dst_all)


def filtered_ranking_metrics(
    z_src_all: torch.Tensor,
    z_dst_all: torch.Tensor,
    test_edge_index: torch.Tensor,
    known_pos_by_src: dict[int, set[int]],
    ks: tuple[int, ...] = (1, 3, 10),
    decoder=None,
) -> tuple[float, dict[int, float]]:
    """
    z_src_all: [num_src, dim] embeddings for the source node type.
    z_dst_all: [num_dst, dim] embeddings for the destination node type.
    test_edge_index: [2, n_test] positive test edges (src_idx, dst_idx), local ids.
    known_pos_by_src: src_idx -> set of ALL known positive dst_idx (train+val+test),
        used to mask out other true edges from the candidate pool (filtered protocol).
    decoder: None for the default dot-product scoring (fast batched matmul);
        otherwise a decoder module from models/decoders.py, scored per-row
        (see _score_row) -- used by the decoder-ablation experiment testing
        whether GAT's ranking anomaly is a decoder-fit issue.
    """
    src_idx, dst_idx = test_edge_index[0].tolist(), test_edge_index[1].tolist()
    if decoder is None:
        scores_all = z_src_all[test_edge_index[0]] @ z_dst_all.T  # [n_test, num_dst]
    else:
        scores_all = torch.stack([_score_row(z_src_all[s], z_dst_all, decoder) for s in src_idx])

    ranks = []
    for i, (s, d) in enumerate(zip(src_idx, dst_idx)):
        row = scores_all[i].clone()
        known = known_pos_by_src.get(s, set())
        if len(known) > 1:
            mask_idx = torch.tensor([k for k in known if k != d], dtype=torch.long, device=row.device)
            if len(mask_idx) > 0:
                row[mask_idx] = float("-inf")
        true_score = row[d].item()
        # standard competition ranking: rank = 1 + #candidates strictly better
        rank = int((row > true_score).sum().item()) + 1
        ranks.append(rank)

    ranks_arr = np.array(ranks, dtype=np.float64)
    mrr = float(np.mean(1.0 / ranks_arr))
    hits = {k: float(np.mean(ranks_arr <= k)) for k in ks}
    return mrr, hits


def sampled_ranking_metrics(
    z_src_all: torch.Tensor,
    z_dst_all: torch.Tensor,
    test_edge_index: torch.Tensor,
    known_pos_by_src: dict[int, set[int]],
    num_negatives: int = 99,
    ks: tuple[int, ...] = (1, 3, 10),
    seed: int = 0,
    decoder=None,
) -> tuple[float, dict[int, float]]:
    """Small-candidate-pool ranking: each positive is ranked among itself
    plus `num_negatives` sampled non-positive candidates (a common
    recommender-systems convention, e.g. num_negatives=99 a la He et al.
    2017's NCF evaluation), rather than the full node catalog used by
    `filtered_ranking_metrics`.

    Exists specifically to test the Phase-0 hypothesis (docs/
    PHASE0_REPRODUCTION.md) that the base paper's much higher Hits@10 on
    AnonNet came from a smaller candidate pool than the full graph, not
    from stronger ranking ability -- comparing this function's output
    against `filtered_ranking_metrics`' output on the same model/data is
    the direct empirical test of that hypothesis (plan limitation B5).
    """
    rng = np.random.default_rng(seed)
    num_dst = z_dst_all.shape[0]
    src_idx, dst_idx = test_edge_index[0].tolist(), test_edge_index[1].tolist()

    ranks = []
    for s, d in zip(src_idx, dst_idx):
        known = known_pos_by_src.get(s, set())
        negatives = []
        while len(negatives) < num_negatives:
            cand = int(rng.integers(0, num_dst))
            if cand != d and cand not in known and cand not in negatives:
                negatives.append(cand)
        candidates = torch.tensor([d] + negatives, dtype=torch.long, device=z_dst_all.device)
        if decoder is None:
            scores = z_dst_all[candidates] @ z_src_all[s]  # [num_negatives + 1]
        else:
            scores = _score_row(z_src_all[s], z_dst_all[candidates], decoder)
        true_score = scores[0].item()
        rank = int((scores > true_score).sum().item()) + 1
        ranks.append(rank)

    ranks_arr = np.array(ranks, dtype=np.float64)
    mrr = float(np.mean(1.0 / ranks_arr))
    hits = {k: float(np.mean(ranks_arr <= k)) for k in ks}
    return mrr, hits


def build_known_positives(*edge_indices: torch.Tensor) -> dict[int, set[int]]:
    """Union known positive (src -> {dst}) pairs across any number of edge_index
    tensors (typically train+val+test), for use as the filtering mask above."""
    known: dict[int, set[int]] = {}
    for ei in edge_indices:
        for s, d in zip(ei[0].tolist(), ei[1].tolist()):
            known.setdefault(s, set()).add(d)
    return known
