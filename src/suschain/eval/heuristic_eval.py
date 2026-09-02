"""Evaluates a heuristic scorer (see models/heuristics.py) using the exact
same protocol as the GNN training loop: validation-tuned threshold, filtered
full-catalog ranking, and sampled-100-candidate ranking -- so heuristic and
GNN results are directly, fairly comparable (plan C2: heuristic floors).
"""
from __future__ import annotations

import numpy as np
import torch

from suschain.eval.metrics import best_f1_threshold, binary_link_metrics


class PopularityScorer:
    """Adapter so the bipartite popularity baseline (LODHalal) can reuse the
    same evaluation code path as the homogeneous CN/AA/PA scorers (AnonNet):
    the score row is independent of `src` -- every source gets the same
    destination-popularity vector."""

    def __init__(self, popularity_vector: np.ndarray):
        self.popularity_vector = popularity_vector

    def score_row(self, src: int, method: str = "popularity") -> np.ndarray:
        return self.popularity_vector


def _score_pairs(scorer, method: str, edge_label_index: torch.Tensor) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    src = edge_label_index[0].numpy()
    dst = edge_label_index[1].numpy()
    row_cache: dict[int, np.ndarray] = {}
    scores = np.empty(len(src), dtype=np.float64)
    for i, (s, d) in enumerate(zip(src, dst)):
        if s not in row_cache:
            row_cache[s] = scorer.score_row(int(s), method)
        scores[i] = row_cache[s][d]
    return scores, row_cache


def evaluate_heuristic(
    scorer,
    method: str,
    val_edge_label_index: torch.Tensor,
    val_edge_label: torch.Tensor,
    test_edge_label_index: torch.Tensor,
    test_edge_label: torch.Tensor,
    known_pos_by_src: dict[int, set[int]],
    num_nodes: int,
    ks: tuple[int, ...] = (1, 3, 10),
    num_negatives_sampled: int = 99,
    seed: int = 0,
) -> dict:
    val_scores, _ = _score_pairs(scorer, method, val_edge_label_index)
    val_labels = val_edge_label.numpy()
    threshold = best_f1_threshold(val_labels, val_scores)

    test_scores, row_cache = _score_pairs(scorer, method, test_edge_label_index)
    test_labels = test_edge_label.numpy()
    binary_metrics = binary_link_metrics(test_labels, test_scores, threshold=threshold)

    pos_mask = test_edge_label == 1.0
    pos_src = test_edge_label_index[0][pos_mask].numpy()
    pos_dst = test_edge_label_index[1][pos_mask].numpy()

    ranks, ranks_sampled = [], []
    rng = np.random.default_rng(seed)
    for s, d in zip(pos_src, pos_dst):
        s = int(s)
        row = row_cache.get(s)
        if row is None:
            row = scorer.score_row(s, method)
            row_cache[s] = row

        # Full-catalog filtered ranking.
        row_f = row.copy()
        known = known_pos_by_src.get(s, set())
        mask_idx = [k for k in known if k != d]
        if mask_idx:
            row_f[mask_idx] = -np.inf
        true_score = row_f[d]
        ranks.append(int((row_f > true_score).sum()) + 1)

        # Sampled 1+K candidate-pool ranking.
        negatives = []
        while len(negatives) < num_negatives_sampled:
            cand = int(rng.integers(0, num_nodes))
            if cand != d and cand not in known and cand not in negatives:
                negatives.append(cand)
        cand_scores = row[[d] + negatives]
        ranks_sampled.append(int((cand_scores > cand_scores[0]).sum()) + 1)

    ranks_arr = np.array(ranks, dtype=np.float64)
    ranks_s_arr = np.array(ranks_sampled, dtype=np.float64)

    return {
        "method": method,
        "threshold": threshold,
        **binary_metrics,
        "mrr": float(np.mean(1.0 / ranks_arr)),
        **{f"hits@{k}": float(np.mean(ranks_arr <= k)) for k in ks},
        "mrr_sampled99": float(np.mean(1.0 / ranks_s_arr)),
        **{f"hits@{k}_sampled99": float(np.mean(ranks_s_arr <= k)) for k in ks},
    }
