"""Link-prediction and (later) node-classification splitters, with the
protocol explicitly stated and versioned here -- fixes plan limitation B5:
the base paper never discloses its negative-sampling ratio, candidate-set
size, or whether ranking is filtered vs. raw. Every choice below is a
deliberate, documented decision, not a silent default.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch_geometric.data import HeteroData
from torch_geometric.transforms import RandomLinkSplit

EdgeType = tuple[str, str, str]

# Frozen protocol for the transductive link-prediction split (Phase 0/1
# default). neg_sampling_ratio is swept in {1, 5, 10} as a Phase-1 ablation;
# 1.0 here is the training-time default, not a claim that 1:1 is correct.
LINK_SPLIT_PROTOCOL = dict(
    num_val=0.1,
    num_test=0.1,
    disjoint_train_ratio=0.3,
    neg_sampling_ratio=1.0,
    add_negative_train_samples=True,
    is_undirected=False,
)


@dataclass
class LinkSplit:
    train: HeteroData
    val: HeteroData
    test: HeteroData
    edge_type: EdgeType
    protocol: dict


def split_link_prediction(
    data: HeteroData,
    edge_type: EdgeType,
    rev_edge_type: EdgeType | None = None,
    seed: int = 0,
    neg_sampling_ratio: float | None = None,
) -> LinkSplit:
    """Transductive edge-level split on a single target relation.

    disjoint_train_ratio=0.3 means 30% of training edges are held out as
    "supervision-only" edges (not used for message passing during training),
    which prevents the model from trivially seeing the edge it's asked to
    predict via its own 1-hop neighborhood -- the leakage check called for
    in the plan's verification section.
    """
    torch.manual_seed(seed)
    protocol = dict(LINK_SPLIT_PROTOCOL)
    if neg_sampling_ratio is not None:
        protocol["neg_sampling_ratio"] = neg_sampling_ratio

    transform = RandomLinkSplit(
        num_val=protocol["num_val"],
        num_test=protocol["num_test"],
        disjoint_train_ratio=protocol["disjoint_train_ratio"],
        neg_sampling_ratio=protocol["neg_sampling_ratio"],
        add_negative_train_samples=protocol["add_negative_train_samples"],
        is_undirected=protocol["is_undirected"],
        edge_types=edge_type,
        rev_edge_types=rev_edge_type,
    )
    train_data, val_data, test_data = transform(data)
    return LinkSplit(train=train_data, val=val_data, test=test_data, edge_type=edge_type, protocol=protocol)


def assert_no_leakage(split: LinkSplit) -> None:
    """Verification helper (per the plan's Phase-1 verification section):
    positive edges in val/test must not appear as positive supervision
    edges in train."""
    et = split.edge_type

    def pos_edges(d, label_val=1.0):
        store = d[et]
        mask = store.edge_label == label_val
        idx = store.edge_label_index[:, mask]
        return set(map(tuple, idx.t().tolist()))

    train_pos = pos_edges(split.train)
    val_pos = pos_edges(split.val)
    test_pos = pos_edges(split.test)

    assert train_pos.isdisjoint(val_pos), "train/val positive edge leakage detected"
    assert train_pos.isdisjoint(test_pos), "train/test positive edge leakage detected"
    assert val_pos.isdisjoint(test_pos), "val/test positive edge leakage detected"

    # negatives must never collide with ANY known positive across all splits
    all_pos = train_pos | val_pos | test_pos

    def neg_edges(d, label_val=0.0):
        store = d[et]
        mask = store.edge_label == label_val
        idx = store.edge_label_index[:, mask]
        return set(map(tuple, idx.t().tolist()))

    for name, d in [("train", split.train), ("val", split.val), ("test", split.test)]:
        negs = neg_edges(d)
        collision = negs & all_pos
        assert not collision, f"{name} negative samples collide with known positives: {len(collision)} cases"


@dataclass
class NodeClassificationSplit:
    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    labels: np.ndarray  # graph-node-id-indexed: labels[graph_node_id] is that node's class.
    class_names: list[str]


def split_node_classification(
    node_ids: np.ndarray,
    string_labels: list[str],
    num_nodes_total: int,
    val_size: float = 0.2,
    test_size: float = 0.2,
    seed: int = 0,
) -> NodeClassificationSplit:
    """Stratified 60/20/20 split (plan: "must not use plain random split"
    given class imbalance -- confirmed necessary directly, see
    docs/PHASE2_SUSTAINABILITY.md's cert_coverage skew finding).

    node_ids / string_labels must be the SAME length and index-aligned
    (typically: only the subset of nodes that actually have a label, e.g.
    AnonNet nodes with a non-null composite_score). node_ids are GRAPH node
    indices (0..num_nodes_total-1), used directly to index a model's full
    per-node logits tensor -- so `labels` is built as a num_nodes_total-
    length array indexed the same way, not a len(node_ids)-length array
    positionally aligned to the labeled subset. Conflating these two index
    spaces was a real bug caught directly (IndexError: train_idx values up
    to 7243 used against a 5688-length labels array) before being fixed
    here -- see tests/test_node_split.py for the regression test."""
    class_names = sorted(set(string_labels))
    name_to_idx = {name: i for i, name in enumerate(class_names)}
    y = np.array([name_to_idx[s] for s in string_labels])

    full_labels = np.full(num_nodes_total, -1, dtype=np.int64)
    full_labels[node_ids] = y

    splitter1 = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(splitter1.split(node_ids, y))

    relative_val_size = val_size / (1 - test_size)
    splitter2 = StratifiedShuffleSplit(n_splits=1, test_size=relative_val_size, random_state=seed)
    train_idx_rel, val_idx_rel = next(splitter2.split(train_val_idx, y[train_val_idx]))
    train_idx = train_val_idx[train_idx_rel]
    val_idx = train_val_idx[val_idx_rel]

    return NodeClassificationSplit(
        train_idx=node_ids[train_idx], val_idx=node_ids[val_idx], test_idx=node_ids[test_idx],
        labels=full_labels, class_names=class_names,
    )
