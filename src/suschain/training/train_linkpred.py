"""Full-batch link-prediction training loop for a single target relation.

Full-batch (not neighbor-sampled mini-batching) is used deliberately: both
graphs are small (LODHalal ~68K nodes / ~176K undirected edges, AnonNet
~7K nodes / ~82K undirected edges after symmetrizing) and comfortably fit
in the local Quadro P2000's 5GB VRAM with embedding_dim=128 -- mini-batch
neighbor sampling would add complexity with no benefit at this scale.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from suschain.data.splitters import EdgeType, LinkSplit, assert_no_leakage, split_link_prediction
from suschain.eval.metrics import best_f1_threshold, binary_link_metrics
from suschain.eval.ranking import build_known_positives, filtered_ranking_metrics, sampled_ranking_metrics
from suschain.models.link_predictor import HeteroLinkPredictor


@dataclass
class TrainConfig:
    conv_kind: str = "sage"  # 'sage' | 'gat' | 'rgcn'
    decoder_kind: str = "dot"  # 'dot' | 'bilinear' | 'mlp'
    embedding_dim: int = 128
    hidden_channels: int = 128
    out_channels: int = 64
    num_layers: int = 2
    lr: float = 0.005
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainResult:
    config: TrainConfig
    val_metrics: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    test_ranking: dict = field(default_factory=dict)
    tuned_threshold: float = 0.5
    best_val_auc: float = 0.0
    epochs_run: int = 0
    wall_clock_seconds: float = 0.0
    peak_vram_mb: float = 0.0


def _forward_scores(model: HeteroLinkPredictor, data: HeteroData, edge_type: EdgeType, device: str):
    node_id_dict = {ntype: data[ntype].node_id.to(device) for ntype in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    z_dict = model.encode(node_id_dict, edge_index_dict)
    src_type, _, dst_type = edge_type
    label_index = data[edge_type].edge_label_index.to(device)
    scores = model.decode(z_dict, label_index, src_type, dst_type)
    return scores, z_dict


def train_link_prediction(
    data: HeteroData,
    edge_type: EdgeType,
    rev_edge_type: EdgeType | None,
    config: TrainConfig,
) -> TrainResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    split = split_link_prediction(data, edge_type, rev_edge_type, seed=config.seed)
    assert_no_leakage(split)

    device = config.device
    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    model = HeteroLinkPredictor(
        metadata=data.metadata(),
        num_nodes_dict=num_nodes_dict,
        embedding_dim=config.embedding_dim,
        hidden_channels=config.hidden_channels,
        out_channels=config.out_channels,
        num_layers=config.num_layers,
        conv_kind=config.conv_kind,
        decoder_kind=config.decoder_kind,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    best_val_auc, best_state, epochs_no_improve, epochs_run = 0.0, None, 0, 0
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        scores, _ = _forward_scores(model, split.train, edge_type, device)
        labels = split.train[edge_type].edge_label.to(device)
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()
        epochs_run = epoch + 1

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
    peak_vram = torch.cuda.max_memory_allocated() / 1e6 if device == "cuda" else 0.0

    model.eval()
    with torch.no_grad():
        val_scores, _ = _forward_scores(model, split.val, edge_type, device)
        val_labels = split.val[edge_type].edge_label.cpu().numpy()
        val_probs = torch.sigmoid(val_scores).cpu().numpy()
        tuned_threshold = best_f1_threshold(val_labels, val_probs)

        test_scores, z_dict = _forward_scores(model, split.test, edge_type, device)
        test_labels = split.test[edge_type].edge_label.cpu().numpy()
        test_probs = torch.sigmoid(test_scores).cpu().numpy()
        test_metrics = binary_link_metrics(test_labels, test_probs, threshold=tuned_threshold)
        test_metrics_fixed = binary_link_metrics(test_labels, test_probs, threshold=0.5)
        test_metrics["f1_at_threshold_0.5"] = test_metrics_fixed["f1"]

        # Filtered ranking (MRR / Hits@k) on the positive test edges only.
        src_type, _, dst_type = edge_type
        pos_mask = split.test[edge_type].edge_label == 1.0
        test_pos_edges = split.test[edge_type].edge_label_index[:, pos_mask]
        known_pos = build_known_positives(
            split.train[edge_type].edge_index,
            split.val[edge_type].edge_label_index[:, split.val[edge_type].edge_label == 1.0],
            test_pos_edges,
        )
        # decoder=None uses the fast dot-product matmul path; a non-dot
        # decoder (bilinear/mlp) must be scored per-row via the model's own
        # decoder module (see eval/ranking.py::_score_row).
        ranking_decoder = None if config.decoder_kind == "dot" else model.decoder
        mrr, hits = filtered_ranking_metrics(
            z_dict[src_type], z_dict[dst_type], test_pos_edges.to(device), known_pos,
            ks=(1, 3, 10), decoder=ranking_decoder,
        )
        # Small-candidate-pool ranking, for direct comparison against the
        # full-catalog filtered ranking above (see docs/PHASE0_REPRODUCTION.md
        # -- tests the hypothesis that the base paper's much higher Hits@10
        # came from a smaller candidate pool, not stronger ranking ability).
        mrr_s99, hits_s99 = sampled_ranking_metrics(
            z_dict[src_type], z_dict[dst_type], test_pos_edges.to(device), known_pos,
            num_negatives=99, ks=(1, 3, 10), seed=config.seed, decoder=ranking_decoder,
        )

    return TrainResult(
        config=config,
        val_metrics={"auc_roc": best_val_auc},
        test_metrics=test_metrics,
        test_ranking={
            "mrr": mrr, **{f"hits@{k}": v for k, v in hits.items()},
            "mrr_sampled99": mrr_s99, **{f"hits@{k}_sampled99": v for k, v in hits_s99.items()},
        },
        tuned_threshold=tuned_threshold,
        best_val_auc=best_val_auc,
        epochs_run=epochs_run,
        wall_clock_seconds=wall_clock,
        peak_vram_mb=peak_vram,
    )
