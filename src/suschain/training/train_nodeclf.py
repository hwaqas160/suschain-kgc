"""Full-batch node-classification training loop, mirroring
train_linkpred.py's structure (see that module for why full-batch is fine
at this graph scale). Reports the full metric suite the base paper's node
classification never did (plan B10: accuracy only, no macro-F1, no
per-class breakdown, no majority-class baseline) via
eval/metrics.py::node_classification_metrics.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from suschain.data.splitters import NodeClassificationSplit
from suschain.eval.metrics import majority_class_baseline_accuracy, node_classification_metrics
from suschain.models.node_classifier import HeteroNodeClassifier


@dataclass
class NodeClfConfig:
    conv_kind: str = "sage"
    embedding_dim: int = 128
    hidden_channels: int = 128
    out_channels: int = 64
    num_layers: int = 2
    lr: float = 0.005
    weight_decay: float = 1e-5
    epochs: int = 150
    patience: int = 20
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class NodeClfResult:
    config: NodeClfConfig
    class_names: list[str] = field(default_factory=list)
    majority_baseline: float = 0.0
    val_macro_f1: float = 0.0
    test_metrics: dict = field(default_factory=dict)
    epochs_run: int = 0
    wall_clock_seconds: float = 0.0


def train_node_classification(
    data: HeteroData,
    target_node_type: str,
    split: NodeClassificationSplit,
    config: NodeClfConfig,
) -> NodeClfResult:
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = config.device

    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    model = HeteroNodeClassifier(
        metadata=data.metadata(),
        num_nodes_dict=num_nodes_dict,
        target_node_type=target_node_type,
        num_classes=len(split.class_names),
        embedding_dim=config.embedding_dim,
        hidden_channels=config.hidden_channels,
        out_channels=config.out_channels,
        num_layers=config.num_layers,
        conv_kind=config.conv_kind,
    ).to(device)

    node_id_dict = {ntype: data[ntype].node_id.to(device) for ntype in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    labels = torch.from_numpy(split.labels).long().to(device)

    train_idx = torch.from_numpy(split.train_idx).long().to(device)
    val_idx = torch.from_numpy(split.val_idx).long().to(device)
    test_idx = torch.from_numpy(split.test_idx).long().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Inverse-frequency class weights from the TRAINING labels only (never
    # val/test, to avoid leaking test-set class balance into training).
    # Without this, plain cross-entropy under severe imbalance (verified:
    # LODHalal's cert-coverage task is 96.2/3.1/0.75 split) collapses to
    # always predicting the majority class -- confirmed directly: all
    # three architectures produced the IDENTICAL degenerate confusion
    # matrix (macro-F1 exactly 0.3266, the value for "always predict
    # class 0") before this weighting was added.
    train_labels_np = labels[train_idx].cpu().numpy()
    class_counts = np.bincount(train_labels_np, minlength=len(split.class_names)).astype(np.float64)
    class_weights = torch.from_numpy(
        (train_labels_np.shape[0] / (len(split.class_names) * np.maximum(class_counts, 1))).astype(np.float32)
    ).to(device)

    t0 = time.perf_counter()
    best_val_f1, best_state, epochs_no_improve, epochs_run = -1.0, None, 0, 0
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(node_id_dict, edge_index_dict)
        loss = F.cross_entropy(logits[train_idx], labels[train_idx], weight=class_weights)
        loss.backward()
        optimizer.step()
        epochs_run = epoch + 1

        model.eval()
        with torch.no_grad():
            logits = model(node_id_dict, edge_index_dict)
            val_pred = logits[val_idx].argmax(dim=-1).cpu().numpy()
            val_true = labels[val_idx].cpu().numpy()
            val_metrics = node_classification_metrics(val_true, val_pred, labels=list(range(len(split.class_names))))
            val_f1 = val_metrics["macro_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
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
        logits = model(node_id_dict, edge_index_dict)
        test_pred = logits[test_idx].argmax(dim=-1).cpu().numpy()
        test_true = labels[test_idx].cpu().numpy()
        test_metrics = node_classification_metrics(test_true, test_pred, labels=list(range(len(split.class_names))))

    majority_baseline = majority_class_baseline_accuracy(labels.cpu().numpy())

    return NodeClfResult(
        config=config,
        class_names=split.class_names,
        majority_baseline=majority_baseline,
        val_macro_f1=best_val_f1,
        test_metrics=test_metrics,
        epochs_run=epochs_run,
        wall_clock_seconds=wall_clock,
    )
