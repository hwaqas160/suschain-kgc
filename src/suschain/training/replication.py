"""v4 work package 9: training and evaluation under the protocol of
Peeris et al. (2026), as described by their first author:
  - two negatives per positive;
  - reported metrics computed on the VALIDATION split;
  - LODHalal: food and manufacturer files only.

The training loop is the one in training/train_linkpred.py (full batch, BCE,
early stopping on validation AUC). It is repeated here, rather than changed
there, so that running experiments that import train_linkpred are unaffected.
Evaluation is wider: every metric on both validation and test, accuracy and F1
at the fixed threshold 0.5 and at the F1-optimal validation threshold, and
filtered ranking over the full catalogue and over 99 sampled negatives, in
both directions of the target relation.
"""
from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

from suschain.eval.metrics import best_f1_threshold, binary_link_metrics
from suschain.eval.ranking_v4 import known_from_edges, summarize_ranks
from suschain.models.link_predictor import HeteroLinkPredictor
from suschain.training.train_linkpred import TrainConfig


def _embed(model, store, device):
    nid = {nt: store[nt].node_id.to(device) for nt in store.node_types}
    return model.encode(nid, {et: store[et].edge_index.to(device) for et in store.edge_types})


def _ranks(score_rows, queries, known, n_cand, sampled, rng):
    """Realistic filtered ranks; if sampled, against 99 random non-known candidates."""
    out = []
    q = queries.t().tolist()
    for b0 in range(0, len(q), 512):
        batch = q[b0:b0 + 512]
        S = score_rows(torch.tensor([s for s, _ in batch])).cpu().numpy().astype(np.float64)
        for i, (s, d) in enumerate(batch):
            row = S[i]
            others = [k for k in known.get(s, ()) if k != d]
            if sampled:
                mask = np.ones(n_cand, dtype=bool)
                mask[others] = False
                mask[d] = False
                pool = np.flatnonzero(mask)
                pool = rng.choice(pool, size=min(99, len(pool)), replace=False)
                cand = row[pool]
            else:
                cand = row.copy()
                cand[others] = -np.inf
                cand = np.delete(cand, d)
            t = row[d]
            out.append(1 + (cand > t).sum() + (cand == t).sum() / 2)
    return np.array(out, dtype=np.float64)


def train_replication(data, edge_type, rev_edge_type, split, config: TrainConfig, both_directions: bool):
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    dev = config.device
    model = HeteroLinkPredictor(
        metadata=data.metadata(), num_nodes_dict={nt: data[nt].num_nodes for nt in data.node_types},
        embedding_dim=config.embedding_dim, hidden_channels=config.hidden_channels,
        out_channels=config.out_channels, num_layers=config.num_layers, conv_kind=config.conv_kind,
    ).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    st, dt = edge_type[0], edge_type[2]

    def scores(store):
        z = _embed(model, store, dev)
        return model.decode(z, store[edge_type].edge_label_index.to(dev), st, dt), z

    t0 = time.perf_counter()
    best_auc, best_state, bad, epochs = 0.0, None, 0, 0
    for epoch in range(config.epochs):
        model.train()
        opt.zero_grad()
        s, _ = scores(split.train)
        F.binary_cross_entropy_with_logits(s, split.train[edge_type].edge_label.to(dev)).backward()
        opt.step()
        epochs = epoch + 1
        model.eval()
        with torch.no_grad():
            vs, _ = scores(split.val)
            auc = binary_link_metrics(split.val[edge_type].edge_label.numpy(), torch.sigmoid(vs).cpu().numpy())["auc_roc"]
        if auc > best_auc:
            best_auc, bad = auc, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= config.patience:
                break
    model.load_state_dict(best_state)
    model.eval()

    def pos(store):
        return store[edge_type].edge_label_index[:, store[edge_type].edge_label == 1.0]
    val_pos, test_pos = pos(split.val), pos(split.test)
    res = {"epochs_run": epochs, "seconds": time.perf_counter() - t0}
    with torch.no_grad():
        vs, _ = scores(split.val)
        vp = torch.sigmoid(vs).cpu().numpy()
        vl = split.val[edge_type].edge_label.numpy()
        thr = best_f1_threshold(vl, vp)
        for name, store, known_edges in (("val", split.val, (val_pos,)),
                                         ("test", split.test, (val_pos, test_pos))):
            s, z = scores(store)
            p = torch.sigmoid(s).cpu().numpy()
            lab = store[edge_type].edge_label.numpy()
            m05 = binary_link_metrics(lab, p, threshold=0.5)
            mf1 = binary_link_metrics(lab, p, threshold=thr)
            r = {"auc_roc": m05["auc_roc"], "pr_auc": m05["pr_auc"],
                 "accuracy_0.5": m05["accuracy"], "f1_0.5": m05["f1"],
                 "accuracy_f1thr": mf1["accuracy"], "f1_f1thr": mf1["f1"]}
            q = pos(store)
            mp = store[edge_type].edge_index
            dirs = [("tail", z[st], z[dt], q, known_from_edges(mp, *known_edges), store[dt].num_nodes)]
            if both_directions:
                dirs.append(("head", z[dt], z[st], q.flip(0),
                             known_from_edges(mp.flip(0), *[k.flip(0) for k in known_edges]), store[st].num_nodes))
            for dname, za, zb, qq, kn, nc in dirs:
                rows = (lambda za_, zb_: (lambda ids: za_[ids.to(za_.device)] @ zb_.T))(za, zb)
                for pool in ("full", "sampled99"):
                    rk = _ranks(rows, qq, kn, nc, pool == "sampled99", np.random.default_rng(config.seed))
                    sm = summarize_ranks(rk)
                    r[f"{dname}_{pool}_mrr"] = sm["mrr"]
                    r[f"{dname}_{pool}_hits@10"] = sm["hits@10"]
            res[name] = r
    return res
