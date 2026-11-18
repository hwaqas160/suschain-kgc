"""Trains and evaluates a link predictor under the inductive entity-holdout
split (plan C4/B7), comparing the embedding-table model (this project's
default everywhere else, matching the base paper's one-hot/categorical
features) against the text-feature model on a TRUE cold-start task: entities
with zero edges in the training graph, scored using only their own node
features. This is the concrete test of the base paper's claim that
GraphSAGE generalizes "to previously unseen entities without retraining."
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from suschain.data.inductive_splitters import EdgeType, InductiveSplit
from suschain.eval.metrics import binary_link_metrics
from suschain.models.inductive_link_predictor import HeteroInductiveLinkPredictor


@dataclass
class InductiveTrainConfig:
    conv_kind: str = "sage"
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
class InductiveResult:
    feature_mode: str  # 'embedding' | 'text'
    transductive_metrics: dict = field(default_factory=dict)
    coldstart_metrics: dict = field(default_factory=dict)
    n_heldout_entities: int = 0
    n_coldstart_edges: int = 0
    coldstart_controls: dict = field(default_factory=dict)
    wall_clock_seconds: float = 0.0


def _sample_negatives(num_dst: int, n: int, exclude: set[int], seed: int, device) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    negs = []
    while len(negs) < n:
        cand = int(rng.integers(0, num_dst))
        if cand not in exclude:
            negs.append(cand)
    return torch.tensor(negs, dtype=torch.long, device=device)


def train_and_eval_inductive(
    split: InductiveSplit,
    edge_type: EdgeType,
    metadata,
    num_nodes_dict: dict[str, int],
    config: InductiveTrainConfig,
    text_features: dict[str, torch.Tensor] | None,
    feature_mode: str,
    coldstart_controls: bool = False,
) -> InductiveResult:
    """coldstart_controls (v4): also score the cold-start edges after removing
    query-specific information from the held-out source entities, to test
    whether the cold-start AUC uses anything about the query entity at all."""
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = config.device
    src_type, _, dst_type = edge_type

    model = HeteroInductiveLinkPredictor(
        metadata=metadata, num_nodes_dict=num_nodes_dict, text_features=text_features,
        embedding_dim=config.embedding_dim, hidden_channels=config.hidden_channels,
        out_channels=config.out_channels, num_layers=config.num_layers, conv_kind=config.conv_kind,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    train_data = split.train_data
    node_id_dict = {ntype: train_data[ntype].node_id.to(device) for ntype in train_data.node_types}
    edge_index_dict = {et: train_data[et].edge_index.to(device) for et in train_data.edge_types}

    # Simple 90/10 train/eval split of the REMAINING (non-heldout) positive
    # edges, with 1:1 random negatives, for the "transductive" control metric.
    all_edges = train_data[edge_type].edge_index
    n_edges = all_edges.shape[1]
    perm = torch.randperm(n_edges, generator=torch.Generator().manual_seed(config.seed))
    n_eval = max(1, int(0.1 * n_edges))
    eval_pos = all_edges[:, perm[:n_eval]]
    mp_edges = all_edges[:, perm[n_eval:]]
    # message-passing graph for training excludes the transductive eval edges
    train_mp_edge_index_dict = dict(edge_index_dict)
    train_mp_edge_index_dict[edge_type] = mp_edges.to(device)

    num_dst = num_nodes_dict[dst_type]
    known_dst_by_src: dict[int, set[int]] = {}
    for s, d in zip(all_edges[0].tolist(), all_edges[1].tolist()):
        known_dst_by_src.setdefault(s, set()).add(d)

    def sample_neg_edges(pos_edges: torch.Tensor, seed_offset: int) -> torch.Tensor:
        negs = []
        rng = np.random.default_rng(config.seed + seed_offset)
        for s in pos_edges[0].tolist():
            known = known_dst_by_src.get(s, set())
            while True:
                cand = int(rng.integers(0, num_dst))
                if cand not in known:
                    negs.append(cand)
                    break
        return torch.tensor(negs, dtype=torch.long)

    eval_neg_dst = sample_neg_edges(eval_pos, seed_offset=100)
    train_pos = mp_edges
    train_neg_dst = sample_neg_edges(train_pos, seed_offset=200)

    t0 = time.perf_counter()
    for epoch in range(config.epochs):
        model.train()
        optimizer.zero_grad()
        z_dict = model.encode(node_id_dict, train_mp_edge_index_dict)

        pos_scores = model.decode(z_dict, train_pos.to(device), src_type, dst_type)
        neg_edge_index = torch.stack([train_pos[0], train_neg_dst]).to(device)
        neg_scores = model.decode(z_dict, neg_edge_index, src_type, dst_type)

        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()

    wall_clock = time.perf_counter() - t0

    model.eval()
    with torch.no_grad():
        # Transductive control: eval edges between entities the model DID see.
        z_dict = model.encode(node_id_dict, train_mp_edge_index_dict)
        pos_scores = model.decode(z_dict, eval_pos.to(device), src_type, dst_type)
        neg_edge_index = torch.stack([eval_pos[0], eval_neg_dst]).to(device)
        neg_scores = model.decode(z_dict, neg_edge_index, src_type, dst_type)
        scores = torch.sigmoid(torch.cat([pos_scores, neg_scores])).cpu().numpy()
        labels_np = np.concatenate([np.ones(pos_scores.shape[0]), np.zeros(neg_scores.shape[0])])
        transductive_metrics = binary_link_metrics(labels_np, scores, threshold=0.5)

        # Cold-start: edges touching entities the model NEVER saw during
        # training (zero-degree in the message-passing graph used above).
        heldout_pos = split.heldout_test_edges
        if heldout_pos.shape[1] > 0:
            heldout_neg_dst = sample_neg_edges(heldout_pos, seed_offset=300)
            pos_scores_h = model.decode(z_dict, heldout_pos.to(device), src_type, dst_type)
            neg_edge_index_h = torch.stack([heldout_pos[0], heldout_neg_dst]).to(device)
            neg_scores_h = model.decode(z_dict, neg_edge_index_h, src_type, dst_type)
            scores_h = torch.sigmoid(torch.cat([pos_scores_h, neg_scores_h])).cpu().numpy()
            labels_h = np.concatenate([np.ones(pos_scores_h.shape[0]), np.zeros(neg_scores_h.shape[0])])
            coldstart_metrics = binary_link_metrics(labels_h, scores_h, threshold=0.5)
        else:
            coldstart_metrics = {}

        controls = {}
        if coldstart_controls and heldout_pos.shape[1] > 0:
            from sklearn.metrics import roc_auc_score
            h_src = heldout_pos[0].to(device)
            ids = h_src.unique()

            def auc_with(z_src):
                zd = dict(z_dict)
                zd[src_type] = z_src
                ps = model.decode(zd, heldout_pos.to(device), src_type, dst_type)
                ns = model.decode(zd, neg_edge_index_h, src_type, dst_type)
                return float(roc_auc_score(labels_h, torch.cat([ps, ns]).cpu().numpy()))

            zs = z_dict[src_type].clone()
            zs[ids] = zs[ids].mean(dim=0, keepdim=True)
            controls["mean_query_vector"] = auc_with(zs)
            if src_type in model.embeddings:
                W = model.embeddings[src_type].weight
                saved = W.data.clone()
                gen = torch.Generator(device=W.device).manual_seed(config.seed + 999)
                for name in ("inputs_zeroed", "inputs_redrawn"):
                    W.data[ids] = (torch.zeros(len(ids), W.shape[1], device=W.device) if name == "inputs_zeroed"
                                   else torch.randn(len(ids), W.shape[1], device=W.device, generator=gen))
                    controls[name] = auc_with(model.encode(node_id_dict, train_mp_edge_index_dict)[src_type])
                    W.data.copy_(saved)

    return InductiveResult(
        feature_mode=feature_mode,
        transductive_metrics=transductive_metrics,
        coldstart_metrics=coldstart_metrics,
        n_heldout_entities=len(split.heldout_entity_ids),
        n_coldstart_edges=heldout_pos.shape[1],
        coldstart_controls=controls,
        wall_clock_seconds=wall_clock,
    )
