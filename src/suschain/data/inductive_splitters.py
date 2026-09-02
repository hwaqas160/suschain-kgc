"""Inductive entity-holdout split (plan C4/B7): tests the base paper's own
claim -- GraphSAGE generalizes "to previously unseen entities without
retraining" (§3.2) -- which it asserts but never actually tests (no
held-out unseen nodes, no cold-start scenario, plan limitation B7,
described there as "the single strongest methodological attack" available
against the paper).

Holds out a fraction of one node type's ENTITIES completely: every edge
touching a held-out entity is removed from the training graph (not just
held out as a supervision label, as RandomLinkSplit does -- the entity is
structurally invisible during training, degree zero). At evaluation, the
held-out entities' true edges are scored using ONLY that entity's own
node features (an nn.Embedding-based model has an untrained/meaningless
feature for it; a text-feature-based model still has a real one) -- this
is what actually tests inductive generalization, not just held-out edges
between otherwise-fully-observed nodes.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import HeteroData

EdgeType = tuple[str, str, str]


@dataclass
class InductiveSplit:
    train_data: HeteroData  # held-out entities' edges fully removed
    heldout_entity_ids: np.ndarray  # local ids of the held-out entities (of holdout_node_type)
    heldout_test_edges: torch.Tensor  # [2, n] edges touching held-out entities, for cold-start eval
    holdout_node_type: str


def inductive_entity_holdout(
    data: HeteroData,
    edge_type: EdgeType,
    holdout_node_type: str,
    holdout_fraction: float = 0.1,
    seed: int = 0,
) -> InductiveSplit:
    """holdout_node_type must be either edge_type[0] or edge_type[2]."""
    assert holdout_node_type in (edge_type[0], edge_type[2]), \
        f"holdout_node_type {holdout_node_type} must be the source or destination of {edge_type}"

    rng = np.random.default_rng(seed)
    num_nodes = data[holdout_node_type].num_nodes
    n_heldout = int(round(num_nodes * holdout_fraction))
    heldout_ids = rng.choice(num_nodes, size=n_heldout, replace=False)
    heldout_set = set(heldout_ids.tolist())

    train_data = data.clone()
    src_type, rel, dst_type = edge_type
    holdout_position = 0 if holdout_node_type == src_type else 1

    original_edge_index = data[edge_type].edge_index
    endpoint = original_edge_index[holdout_position]
    is_heldout_edge = torch.tensor([int(e.item()) in heldout_set for e in endpoint])

    train_data[edge_type].edge_index = original_edge_index[:, ~is_heldout_edge]
    heldout_test_edges = original_edge_index[:, is_heldout_edge]

    # Also strip the held-out entities' edges from the reverse relation, if
    # present (ToUndirected()-created), so they're truly degree-zero in the
    # training message-passing graph, not half-visible via the reverse edge.
    rev_edge_type = (dst_type, f"rev_{rel}", src_type)
    if rev_edge_type in data.edge_types:
        rev_ei = data[rev_edge_type].edge_index
        rev_holdout_position = 1 if holdout_node_type == src_type else 0
        rev_endpoint = rev_ei[rev_holdout_position]
        rev_is_heldout = torch.tensor([int(e.item()) in heldout_set for e in rev_endpoint])
        train_data[rev_edge_type].edge_index = rev_ei[:, ~rev_is_heldout]

    return InductiveSplit(
        train_data=train_data,
        heldout_entity_ids=heldout_ids,
        heldout_test_edges=heldout_test_edges,
        holdout_node_type=holdout_node_type,
    )
