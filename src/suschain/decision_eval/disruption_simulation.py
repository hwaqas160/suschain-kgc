"""Decision-level evaluation via disruption simulation (plan C5): converts
prediction accuracy into managerial outcomes -- "our method finds a
compliant substitute for X% of disrupted links, at Y% lower risk than the
removed supplier" is a decision-relevant claim; "AUC=0.91" is not. Directly
answers plan B13: the base paper's stated purpose is decision support, yet
everything it reports is prediction accuracy, never decision quality.

Simulates supplier disruption (randomly removing x% of food->manufacturer
edges, as if those manufacturers stopped supplying), asks each candidate
method for top-k substitute recommendations, and reports:
- substitute_recall_at_k: does the model still recognize the disrupted
  manufacturer as plausible (a sanity/validity check on the ranking, not
  the decision-relevant metric on its own)
- mean_risk_at_k: average sustainability risk of the recommended set
- pct_fully_certified_at_k: % of recommendations meeting the certification
  bar (a concrete "compliance constraint satisfaction rate")
- delta_risk_vs_removed: recommended-set risk minus the disrupted
  manufacturer's own risk (negative = recommendations improve compliance)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class DisruptionScenario:
    disrupted_edges: torch.Tensor  # [2, n] the removed (src, dst) edges
    remaining_edge_index: torch.Tensor  # training graph with disrupted edges removed


def simulate_disruption(edge_index: torch.Tensor, fraction: float = 0.1, seed: int = 0) -> DisruptionScenario:
    rng = np.random.default_rng(seed)
    n = edge_index.shape[1]
    n_disrupt = int(round(n * fraction))
    disrupt_idx = rng.choice(n, size=n_disrupt, replace=False)
    mask = np.zeros(n, dtype=bool)
    mask[disrupt_idx] = True
    return DisruptionScenario(
        disrupted_edges=edge_index[:, mask],
        remaining_edge_index=edge_index[:, ~mask],
    )


def evaluate_recommendations(
    recommendations: np.ndarray,  # [n_disrupted, k] recommended dst indices per disrupted edge
    disrupted_edges: torch.Tensor,  # [2, n_disrupted] the (src, true_dst) pairs that were removed
    risk_by_dst: np.ndarray,  # [num_dst] risk score per destination, higher = worse
    fully_certified_mask: np.ndarray,  # [num_dst] bool, True = meets compliance bar
) -> dict:
    true_dst = disrupted_edges[1].numpy()
    k = recommendations.shape[1]

    recall_at_k = float(np.mean([true_dst[i] in recommendations[i] for i in range(len(true_dst))]))
    rec_risk = risk_by_dst[recommendations]  # [n_disrupted, k]
    mean_risk_at_k = float(rec_risk.mean())
    pct_certified = float(fully_certified_mask[recommendations].mean())
    removed_risk = risk_by_dst[true_dst]
    delta_risk = float(rec_risk.mean(axis=1).mean() - removed_risk.mean())

    return {
        "k": k,
        "n_disrupted": len(true_dst),
        "substitute_recall_at_k": recall_at_k,
        "mean_risk_at_k": mean_risk_at_k,
        "pct_fully_certified_at_k": pct_certified,
        "delta_risk_vs_removed": delta_risk,
    }


def recommend_by_scores(scores_all: torch.Tensor, k: int, exclude_dst: dict[int, set[int]] | None = None) -> np.ndarray:
    """scores_all: [n_disrupted, num_dst] score matrix. Returns [n_disrupted, k]
    top-k destination indices per row, optionally masking out known-invalid
    candidates (e.g. the disrupted source's other known-true destinations,
    to keep recommendations from trivially repeating an already-known link)."""
    scores = scores_all.clone()
    if exclude_dst is not None:
        for i, excl in exclude_dst.items():
            if excl:
                idx = torch.tensor(list(excl), dtype=torch.long)
                scores[i, idx] = float("-inf")
    return torch.topk(scores, k=k, dim=1).indices.numpy()


def recommend_random(n_disrupted: int, num_dst: int, k: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.array([rng.choice(num_dst, size=k, replace=False) for _ in range(n_disrupted)])


def recommend_posthoc_filter(
    scores_all: torch.Tensor, risk_by_dst: np.ndarray, k: int, pool_size: int = 50,
    exclude_dst: dict[int, set[int]] | None = None,
) -> np.ndarray:
    """Two-stage retrieve-then-filter baseline, matching the post-hoc
    filtering mechanism the base paper describes but does not implement or
    evaluate: predicted alternatives are ranked by link-prediction score,
    then filtered or re-ranked by a sustainability attribute (Peeris et al.
    2026, Section 3.3). Retrieves the top pool_size candidates by plain
    link-prediction score, then re-ranks that pool by risk (ascending) and
    keeps the top k. Uses the same trained scores as the plain (unconstrained)
    model, so any difference from that arm is attributable to the filtering
    step alone rather than to a different encoder."""
    scores = scores_all.clone()
    if exclude_dst is not None:
        for i, excl in exclude_dst.items():
            if excl:
                idx = torch.tensor(list(excl), dtype=torch.long)
                scores[i, idx] = float("-inf")
    pool_size = min(pool_size, scores.shape[1])
    pool_idx = torch.topk(scores, k=pool_size, dim=1).indices.numpy()  # [n, pool_size]
    pool_risk = risk_by_dst[pool_idx]  # [n, pool_size]
    order = np.argsort(pool_risk, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(pool_idx, order, axis=1)
