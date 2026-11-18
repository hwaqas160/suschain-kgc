"""v4 filtered ranking, shared by GNNs, KGE models and heuristics.

Changes from eval/ranking.py (kept unchanged so older scripts reproduce):
  1. Ties are split: realistic rank = 1 + #greater + #equal_others / 2.
     Ties are rare for continuous scores but common for heuristics and for
     poorly trained models.
  2. The filter set is all known positives of the evaluated split:
     train message-passing + train supervision + val (+ test). The old code
     filtered with split.train.edge_index, which in a disjoint_train_ratio
     split holds only the message-passing edges, so train supervision edges
     were left in the candidate pool.
  3. One scoring interface, `score_rows(src_ids) -> [b, num_dst]`, so every
     model family is ranked by the same code.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import torch


def known_from_edges(*edge_indices: torch.Tensor) -> dict[int, set[int]]:
    known: dict[int, set[int]] = {}
    for ei in edge_indices:
        for s, d in ei.t().tolist():
            known.setdefault(int(s), set()).add(int(d))
    return known


@torch.no_grad()
def filtered_ranks(
    score_rows: Callable[[torch.Tensor], torch.Tensor],
    query_edges: torch.Tensor,
    known_by_src: dict[int, set[int]],
    batch_size: int = 512,
    device: str = "cpu",
) -> dict[str, np.ndarray]:
    """Optimistic, realistic and pessimistic filtered ranks of each query
    edge's true destination among all destinations of its type."""
    src = query_edges[0].tolist()
    dst = query_edges[1].tolist()
    out = {"optimistic": [], "realistic": [], "pessimistic": []}
    for b0 in range(0, len(src), batch_size):
        bs, bd = src[b0:b0 + batch_size], dst[b0:b0 + batch_size]
        scores = score_rows(torch.tensor(bs, device=device)).float().clone()
        rows, cols = [], []
        for i, (s, d) in enumerate(zip(bs, bd)):
            for k in known_by_src.get(s, ()):
                if k != d:
                    rows.append(i)
                    cols.append(k)
        if rows:
            scores[torch.tensor(rows, device=scores.device), torch.tensor(cols, device=scores.device)] = float("-inf")
        idx = torch.arange(len(bs), device=scores.device)
        true = scores[idx, torch.tensor(bd, device=scores.device)].unsqueeze(1)
        greater = (scores > true).sum(1).double()
        equal = (scores == true).sum(1).double() - 1.0
        out["optimistic"].append((1 + greater).cpu().numpy())
        out["realistic"].append((1 + greater + equal / 2).cpu().numpy())
        out["pessimistic"].append((1 + greater + equal).cpu().numpy())
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}


def summarize_ranks(ranks: np.ndarray, ks=(1, 3, 10)) -> dict[str, float]:
    r = np.asarray(ranks, dtype=np.float64)
    return {"mrr": float(np.mean(1.0 / r)), **{f"hits@{k}": float(np.mean(r <= k)) for k in ks}}
