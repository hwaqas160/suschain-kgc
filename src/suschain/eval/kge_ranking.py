"""Type-restricted filtered ranking for PyKEEN KGE models.

Why this exists. train_kge_baseline's ranking_metrics come from PyKEEN's own
RankBasedEvaluator, which by default corrupts against the FULL entity
vocabulary (food + manufacturer combined for LODHalal). Every other row in
Table 2 -- GNNs, heuristics -- is ranked only against the correct
DESTINATION TYPE (2,520 manufacturer candidates), via
suschain.eval.ranking.filtered_ranking_metrics. The two protocols are not
the same evaluation, which is a second, independent problem on top of
untyped negative sampling: even a perfectly-trained KGE model is ranked
against ~45,600 candidates instead of 2,520, most of them a different
entity type by construction.

This module gives KGE models the same type-restricted, filtered, full-catalog
ranking protocol as everything else in the benchmark, using the model's own
score_hrt (not a generic dot product, which is only correct for a subset of
KGE scoring functions).
"""
from __future__ import annotations

import numpy as np
import torch


def kge_filtered_ranking_metrics(
    model,
    entity_to_id: dict[str, int],
    relation_id: int,
    src_offset: int,
    dst_offset: int,
    num_dst: int,
    test_pos_edges: torch.Tensor,
    known_pos_by_src: dict[int, set[int]],
    ks: tuple[int, ...] = (1, 3, 10),
    device: str = "cpu",
) -> tuple[float, dict[int, float]]:
    """Filtered MRR/Hits@k for a trained PyKEEN model, candidates restricted
    to the destination entity type only (local ids 0..num_dst-1)."""
    model.eval()
    src_local = test_pos_edges[0].tolist()
    dst_local = test_pos_edges[1].tolist()

    dst_global = torch.arange(num_dst, device=device) + dst_offset
    ranks = []
    with torch.no_grad():
        for s, d in zip(src_local, dst_local):
            s_global = s + src_offset
            heads = torch.full((num_dst,), s_global, dtype=torch.long, device=device)
            rels = torch.full((num_dst,), relation_id, dtype=torch.long, device=device)
            hrt = torch.stack([heads, rels, dst_global], dim=1)
            scores = model.score_hrt(hrt).squeeze(-1)

            known = known_pos_by_src.get(s, set())
            if len(known) > 1:
                mask_idx = torch.tensor([k for k in known if k != d], dtype=torch.long, device=device)
                if len(mask_idx) > 0:
                    scores[mask_idx] = float("-inf")

            true_score = scores[d].item()
            rank = int((scores > true_score).sum().item()) + 1
            ranks.append(rank)

    ranks_arr = np.array(ranks, dtype=np.float64)
    mrr = float(np.mean(1.0 / ranks_arr))
    hits = {k: float(np.mean(ranks_arr <= k)) for k in ks}
    return mrr, hits
