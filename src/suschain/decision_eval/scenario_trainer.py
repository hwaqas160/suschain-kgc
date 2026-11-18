"""Trains a link predictor on one disruption scenario.

This is the trainer from scripts/20_disruption_multiseed.py, moved here so the
v4 scripts share one copy. With penalty_on="all" and the same seed it performs
exactly the same computation as script 20, so its lambda=0 and lambda=1 results
reproduce phase4_disruption_multiseed_lodhalal.json.

The model never sees the disrupted edges: they are removed from both the
message-passing graph and the supervision set.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from suschain.models.link_predictor import HeteroLinkPredictor
from suschain.training.multi_objective_loss import sustainability_penalty

PENALTY_TARGETS = ("all", "negatives")


def train_scenario_model(data, edge_type, rev_edge_type, remaining_edges, risk_norm,
                         sustainability_lambda: float, seed: int, *, device: str,
                         epochs: int = 100, conv_kind: str = "rgcn",
                         penalty_on: str = "all", mp_edge_index=None, hparams: dict | None = None):
    """Returns (z_src, z_dst): final embeddings for the source and destination
    node types of `edge_type`.

    penalty_on: "all" applies the sustainability penalty to every scored pair
    (the paper's Eq. 12); "negatives" applies it to sampled negatives only, so
    true links to risky suppliers are not pushed down.

    hparams (v4): optional overrides of embedding_dim, hidden_channels,
    out_channels, num_layers, lr and weight_decay, e.g. a configuration tuned
    by scripts/41. None keeps the fixed defaults used by scripts 20, 34, 36, 37.

    mp_edge_index: if given, the message-passing edges for `edge_type`, used in
    place of `remaining_edges`. AnonNet needs this: it has one node type and one
    relation, so message passing must run over both directions of each visible
    link while supervision uses the oriented links only.
    """
    if penalty_on not in PENALTY_TARGETS:
        raise ValueError(f"penalty_on must be one of {PENALTY_TARGETS}")
    torch.manual_seed(seed)
    src_type, _, dst_type = edge_type
    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    hp = dict(embedding_dim=128, hidden_channels=128, out_channels=64, num_layers=2,
              lr=0.005, weight_decay=1e-5)
    if hparams:
        unknown = set(hparams) - set(hp)
        if unknown:
            raise ValueError(f"unknown hparams {unknown}")
        hp.update(hparams)
    model = HeteroLinkPredictor(
        metadata=data.metadata(), num_nodes_dict=num_nodes_dict,
        embedding_dim=hp["embedding_dim"], hidden_channels=hp["hidden_channels"],
        out_channels=hp["out_channels"], num_layers=hp["num_layers"], conv_kind=conv_kind,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])

    node_id_dict = {nt: data[nt].node_id.to(device) for nt in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(device) for et in data.edge_types}
    edge_index_dict[edge_type] = (remaining_edges if mp_edge_index is None
                                  else mp_edge_index).to(device)
    if mp_edge_index is None and rev_edge_type in edge_index_dict:
        edge_index_dict[rev_edge_type] = remaining_edges.flip(0).to(device)

    num_dst = data[dst_type].num_nodes
    known: dict[int, set[int]] = {}
    for s, d in zip(remaining_edges[0].tolist(), remaining_edges[1].tolist()):
        known.setdefault(s, set()).add(d)

    # One fixed negative per positive, drawn once (as in script 20).
    rng = np.random.default_rng(seed)
    neg_dst = []
    for s in remaining_edges[0].tolist():
        ks = known.get(s, set())
        while True:
            c = int(rng.integers(0, num_dst))
            if c not in ks:
                neg_dst.append(c)
                break
    neg_dst = torch.tensor(neg_dst, dtype=torch.long, device=device)
    pos_edges = remaining_edges.to(device)

    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        z = model.encode(node_id_dict, edge_index_dict)
        pos = model.decode(z, pos_edges, src_type, dst_type)
        neg = model.decode(z, torch.stack([pos_edges[0], neg_dst]), src_type, dst_type)
        scores = torch.cat([pos, neg])
        labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        if sustainability_lambda > 0:
            if penalty_on == "all":
                loss = loss + sustainability_lambda * sustainability_penalty(
                    scores, torch.cat([pos_edges[1], neg_dst]), risk_norm)
            else:
                loss = loss + sustainability_lambda * sustainability_penalty(
                    neg, neg_dst, risk_norm)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(node_id_dict, edge_index_dict)
    return z[src_type], z[dst_type]
