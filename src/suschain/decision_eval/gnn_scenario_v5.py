"""v5 GNN trainer for the substitution scenario with the loss, the number of
negatives and the number of gradient steps as explicit hyperparameters.

Why this exists. The v4 GNN trainer (scenario_trainer.py) used binary
cross-entropy with one fixed negative per positive, a full-batch step per epoch
(100 steps in all) and a dot-product decoder. The v4 embedding models used a
softmax over hundreds of sampled negatives and thousands of minibatch steps.
Architecture was therefore confounded with the loss, the negatives and the
number of updates. Here the GNN gets the same freedoms:

  loss        "bce"      BCE on each positive and `num_neg` uniform negatives
              "softmax"  cross-entropy of the true supplier against `num_neg`
                         sampled suppliers shared by the batch (as in kge_torch)
  num_neg     negatives per positive (bce) or shared negatives per batch (softmax)
  batch_size  positives per gradient step; the encoder runs on the full graph at
              every step, so the number of steps per epoch is n_pos / batch_size
  epochs      passes over the positives

Score of (customer c, supplier v) is the dot product of their encoder outputs,
as before. Message passing runs over both directions of every visible link.
The optional sustainability penalty is Eq. (1) of the paper: the mean of
sigmoid(score) * risk(supplier) over the scored positive and negative pairs.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F

from suschain.models.link_predictor import HeteroLinkPredictor

HP_DEFAULT = dict(embedding_dim=128, hidden_channels=128, out_channels=64, num_layers=2,
                  lr=0.005, weight_decay=1e-5, loss="bce", num_neg=1, batch_size=2048, epochs=60)


def train_gnn_v5(data, edge_type, train_edges, risk_norm, lam: float, seed: int, *, device: str,
                 conv_kind: str, hp: dict, mp_edge_index=None):
    """train_edges: [2, m] supplier -> customer, visible. Returns
    (z_customer, z_supplier, seconds): encoder outputs for every node (the node
    set has one type) and the training wall-clock time."""
    t0 = time.perf_counter()
    h = dict(HP_DEFAULT)
    unknown = set(hp) - set(h)
    if unknown:
        raise ValueError(f"unknown hparams {unknown}")
    h.update(hp)
    torch.manual_seed(seed)
    src_type, _, dst_type = edge_type
    n = data[dst_type].num_nodes
    model = HeteroLinkPredictor(
        metadata=data.metadata(), num_nodes_dict={nt: data[nt].num_nodes for nt in data.node_types},
        embedding_dim=h["embedding_dim"], hidden_channels=h["hidden_channels"],
        out_channels=h["out_channels"], num_layers=h["num_layers"], conv_kind=conv_kind).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=h["lr"], weight_decay=h["weight_decay"])

    node_id = {nt: data[nt].node_id.to(device) for nt in data.node_types}
    ei = {et: data[et].edge_index.to(device) for et in data.edge_types}
    if mp_edge_index is None:
        mp_edge_index = torch.cat([train_edges, train_edges.flip(0)], 1)
    ei[edge_type] = mp_edge_index.to(device)

    sup = train_edges[0].to(device)           # supplier of each positive
    cus = train_edges[1].to(device)           # customer of each positive
    m = sup.numel()
    risk = risk_norm.to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    K, bs = int(h["num_neg"]), int(min(h["batch_size"], m))

    for _ in range(int(h["epochs"])):
        perm = torch.randperm(m, device=device, generator=gen)
        for i in range(0, m, bs):
            idx = perm[i:i + bs]
            c, s = cus[idx], sup[idx]
            model.train()
            opt.zero_grad()
            z = model.encode(node_id, ei)[dst_type]
            zc = z[c]
            pos = (zc * z[s]).sum(-1)
            if h["loss"] == "bce":
                neg_i = torch.randint(0, n, (len(idx), K), device=device, generator=gen)
                neg = (zc.unsqueeze(1) * z[neg_i]).sum(-1)
                scores = torch.cat([pos, neg.reshape(-1)])
                labels = torch.cat([torch.ones_like(pos), torch.zeros_like(neg.reshape(-1))])
                loss = F.binary_cross_entropy_with_logits(scores, labels)
                pen_dst = torch.cat([s, neg_i.reshape(-1)])
            else:
                neg_i = torch.randint(0, n, (K,), device=device, generator=gen)
                logits = torch.cat([pos.unsqueeze(1), zc @ z[neg_i].T], 1)
                loss = F.cross_entropy(logits, torch.zeros(len(idx), dtype=torch.long, device=device))
                scores = logits.reshape(-1)
                pen_dst = torch.cat([s.unsqueeze(1), neg_i.unsqueeze(0).expand(len(idx), -1)], 1).reshape(-1)
            if lam > 0:
                loss = loss + lam * (torch.sigmoid(scores) * risk[pen_dst]).mean()
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        z = model.encode(node_id, ei)[dst_type]
    return z, z, time.perf_counter() - t0
