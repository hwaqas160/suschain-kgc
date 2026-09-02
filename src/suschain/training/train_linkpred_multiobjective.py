"""Sustainability-aware link-prediction training (plan C3): a separate
training loop, not a modification of train_linkpred.py, so the already-
validated Phase 0/1 pipeline (and its passing tests / reproduced results)
carries zero risk of regression from this Phase 3 addition. Reuses the
same model/split/metric infrastructure; adds the auxiliary sustainability
penalty (multi_objective_loss.py) to the training loss and a risk-aware
evaluation metric (mean risk of top-k recommended candidates) alongside
the standard ranking metrics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from suschain.data.splitters import EdgeType, assert_no_leakage, split_link_prediction
from suschain.eval.metrics import best_f1_threshold, binary_link_metrics
from suschain.eval.ranking import build_known_positives, filtered_ranking_metrics
from suschain.models.link_predictor import HeteroLinkPredictor
from suschain.training.multi_objective_loss import normalize_risk, sustainability_penalty
from suschain.training.train_linkpred import TrainConfig, _forward_scores


@dataclass
class MultiObjectiveResult:
    config: TrainConfig
    sustainability_lambda: float = 0.0
    test_metrics: dict = field(default_factory=dict)
    test_ranking: dict = field(default_factory=dict)
    mean_risk_at_10: float = 0.0  # mean risk of the top-10 ranked candidates, averaged over test sources
    wall_clock_seconds: float = 0.0


def _mean_risk_at_k(z_src_all, z_dst_all, test_src_idx, risk_by_dst_norm, k: int = 10) -> float:
    """For each test source node, scores ALL destination candidates, takes
    the top-k by predicted score, and averages their risk. Aggregated
    (mean) over all test source nodes -- the y-axis of the Pareto plot."""
    scores_all = z_src_all[test_src_idx] @ z_dst_all.T  # [n_src, num_dst]
    topk_idx = torch.topk(scores_all, k=min(k, scores_all.shape[1]), dim=1).indices  # [n_src, k]
    topk_risk = risk_by_dst_norm[topk_idx]  # [n_src, k]
    return float(topk_risk.mean().item())


def train_link_prediction_multiobjective(
    data: HeteroData,
    edge_type: EdgeType,
    rev_edge_type: EdgeType | None,
    config: TrainConfig,
    raw_risk_by_dst: torch.Tensor,
    sustainability_lambda: float = 0.0,
) -> MultiObjectiveResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = config.device

    split = split_link_prediction(data, edge_type, rev_edge_type, seed=config.seed)
    assert_no_leakage(split)

    risk_norm = normalize_risk(raw_risk_by_dst.to(device))

    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    model = HeteroLinkPredictor(
        metadata=data.metadata(), num_nodes_dict=num_nodes_dict,
        embedding_dim=config.embedding_dim, hidden_channels=config.hidden_channels,
        out_channels=config.out_channels, num_layers=config.num_layers,
        conv_kind=config.conv_kind, decoder_kind=config.decoder_kind,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    src_type, _, dst_type = edge_type
    t0 = time.perf_counter()
    best_val_auc, best_state, epochs_no_improve = 0.0, None, 0
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        scores, _ = _forward_scores(model, split.train, edge_type, device)
        labels = split.train[edge_type].edge_label.to(device)
        bce = F.binary_cross_entropy_with_logits(scores, labels)

        if sustainability_lambda > 0:
            dst_idx = split.train[edge_type].edge_label_index[1].to(device)
            penalty = sustainability_penalty(scores, dst_idx, risk_norm)
            loss = bce + sustainability_lambda * penalty
        else:
            loss = bce
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_scores, _ = _forward_scores(model, split.val, edge_type, device)
            val_labels = split.val[edge_type].edge_label.to(device)
            val_probs = torch.sigmoid(val_scores).cpu().numpy()
            val_auc = binary_link_metrics(val_labels.cpu().numpy(), val_probs)["auc_roc"]

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    wall_clock = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        val_scores, _ = _forward_scores(model, split.val, edge_type, device)
        val_labels = split.val[edge_type].edge_label.cpu().numpy()
        val_probs = torch.sigmoid(val_scores).cpu().numpy()
        threshold = best_f1_threshold(val_labels, val_probs)

        test_scores, z_dict = _forward_scores(model, split.test, edge_type, device)
        test_labels = split.test[edge_type].edge_label.cpu().numpy()
        test_probs = torch.sigmoid(test_scores).cpu().numpy()
        test_metrics = binary_link_metrics(test_labels, test_probs, threshold=threshold)

        pos_mask = split.test[edge_type].edge_label == 1.0
        test_pos_edges = split.test[edge_type].edge_label_index[:, pos_mask]
        known_pos = build_known_positives(
            split.train[edge_type].edge_index,
            split.val[edge_type].edge_label_index[:, split.val[edge_type].edge_label == 1.0],
            test_pos_edges,
        )
        mrr, hits = filtered_ranking_metrics(
            z_dict[src_type], z_dict[dst_type], test_pos_edges.to(device), known_pos, ks=(1, 3, 10)
        )

        test_src_idx = test_pos_edges[0].unique().to(device)
        mean_risk_10 = _mean_risk_at_k(z_dict[src_type], z_dict[dst_type], test_src_idx, risk_norm, k=10)

    return MultiObjectiveResult(
        config=config,
        sustainability_lambda=sustainability_lambda,
        test_metrics=test_metrics,
        test_ranking={"mrr": mrr, **{f"hits@{k}": v for k, v in hits.items()}},
        mean_risk_at_10=mean_risk_10,
        wall_clock_seconds=wall_clock,
    )
