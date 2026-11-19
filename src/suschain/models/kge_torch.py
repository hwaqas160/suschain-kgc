"""v4 knowledge-graph-embedding baselines in plain PyTorch.

Why not PyKEEN (models/kge.py)? One PyKEEN run took about 20 minutes on
LODHalal, which rules out the 25-trial search the GNNs get. The PyKEEN
runs also trained on the target relation only, while the GNNs pass messages
over every relation, so the KGE models never saw the certificate links.

Setup, identical for the four scoring functions:
  - entities: every node of every type, one table with per-type offsets;
  - triples: every forward relation of the training graph (relations named
    "rev_*" are the GNN's reverse message-passing edges and are skipped),
    with the target relation restricted to training edges;
  - reciprocal training: each triple (h, r, t) also yields (t, r^-1, h)
    (Lacroix et al., 2018), so the model learns to predict both ends;
  - loss: softmax cross-entropy of the true entity against `num_neg`
    entities of the correct type, sampled once per batch and shared by its
    rows (all entities of the type when there are no more than num_neg);
  - L2 penalty on the embeddings used in the batch;
  - early stopping on validation filtered MRR of the target relation.

Scoring functions
  TransE   gamma - ||h + r - t||_2
  DistMult <h, r, t>
  ComplEx  Re(<h, r, conj(t)>)
  RotatE   gamma - ||h o exp(i theta) - t||_2  (L2 over the complex vector,
           not the original sum of moduli: the original, computed exactly,
           took about 30x longer per epoch and ruled out the tuning budget)
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from suschain.eval.metrics import best_accuracy_threshold, best_f1_threshold, binary_link_metrics
from suschain.eval.ranking_v4 import filtered_ranks, known_from_edges, summarize_ranks

KGE_MODELS = ("TransE", "DistMult", "ComplEx", "RotatE")


@dataclass
class KGEConfig:
    model: str = "DistMult"
    dim: int = 128
    lr: float = 1e-3
    reg: float = 1e-6
    batch_size: int = 512
    num_neg: int = 1024
    gamma: float = 6.0
    max_epochs: int = 200
    eval_every: int = 5
    patience: int = 4          # evaluations without improvement
    loss: str = "softmax"      # "softmax" (default, all v4 results) or "bce" (v5 ablation)
    seed: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class KGEResultV4:
    config: KGEConfig
    val_ranking_v4: dict = field(default_factory=dict)
    test_ranking_v4: dict = field(default_factory=dict)
    test_metrics: dict = field(default_factory=dict)
    test_metrics_acc_threshold: dict = field(default_factory=dict)
    test_ranks_v4: dict = field(default_factory=dict)
    test_scores_v4: dict = field(default_factory=dict)
    epochs_run: int = 0
    wall_clock_seconds: float = 0.0


class KGEModel(nn.Module):
    def __init__(self, kind: str, num_entities: int, num_relations: int, dim: int, gamma: float):
        super().__init__()
        if kind not in KGE_MODELS:
            raise ValueError(kind)
        self.kind, self.gamma = kind, gamma
        if kind in ("ComplEx", "RotatE") and dim % 2:
            raise ValueError("complex models need an even dim")
        self.ent = nn.Embedding(num_entities, dim)
        rel_dim = dim // 2 if kind == "RotatE" else dim
        self.rel = nn.Embedding(num_relations, rel_dim)
        bound = 6 / math.sqrt(dim)
        nn.init.uniform_(self.ent.weight, -bound, bound)
        if kind == "RotatE":
            nn.init.uniform_(self.rel.weight, -math.pi, math.pi)
        else:
            nn.init.uniform_(self.rel.weight, -bound, bound)

    def score(self, h: torch.Tensor, r: torch.Tensor, cand: torch.Tensor) -> torch.Tensor:
        """h, r: [b] ids; cand: [m] entity ids. Returns [b, m] scores."""
        eh, er, ec = self.ent(h), self.rel(r), self.ent(cand)
        if self.kind == "DistMult":
            return (eh * er) @ ec.T
        if self.kind == "ComplEx":
            hr_, hi = eh.chunk(2, -1)
            rr, ri = er.chunk(2, -1)
            qr, qi = hr_ * rr - hi * ri, hr_ * ri + hi * rr
            cr, ci = ec.chunk(2, -1)
            return qr @ cr.T + qi @ ci.T
        if self.kind == "TransE":
            q = eh + er
            d2 = (q * q).sum(-1, keepdim=True) + (ec * ec).sum(-1).unsqueeze(0) - 2 * q @ ec.T
            return self.gamma - d2.clamp_min(1e-9).sqrt()
        # RotatE, L2 variant: gamma - ||h o r - t||_2 over the complex vector,
        # computed through the expansion |q|^2 + |t|^2 - 2 Re<q, conj t>
        hr_, hi = eh.chunk(2, -1)
        c, s = torch.cos(er), torch.sin(er)
        q = torch.cat([hr_ * c - hi * s, hr_ * s + hi * c], -1)
        d2 = (q * q).sum(-1, keepdim=True) + (ec * ec).sum(-1).unsqueeze(0) - 2 * q @ ec.T
        return self.gamma - d2.clamp_min(1e-9).sqrt()

    def score_aligned(self, h, r, t):
        """Scores of aligned triples, any shape of t that broadcasts against h, r
        ([b] with [b, k]); returns the shape of t. Used by the BCE loss."""
        eh, er, et = self.ent(h), self.rel(r), self.ent(t)
        if t.dim() == 2:
            eh, er = eh.unsqueeze(1), er.unsqueeze(1)
        if self.kind == "DistMult":
            return (eh * er * et).sum(-1)
        if self.kind == "ComplEx":
            hr_, hi = eh.chunk(2, -1)
            rr, ri = er.chunk(2, -1)
            tr, ti = et.chunk(2, -1)
            return ((hr_ * rr - hi * ri) * tr + (hr_ * ri + hi * rr) * ti).sum(-1)
        if self.kind == "TransE":
            return self.gamma - (eh + er - et).pow(2).sum(-1).clamp_min(1e-9).sqrt()
        hr_, hi = eh.chunk(2, -1)
        c, s = torch.cos(er), torch.sin(er)
        q = torch.cat([hr_ * c - hi * s, hr_ * s + hi * c], -1)
        return self.gamma - (q - et).pow(2).sum(-1).clamp_min(1e-9).sqrt()

    def score_pairs(self, h, r, t):
        """Score aligned triples, [n]."""
        out = []
        for i in range(0, len(h), 4096):
            hh, rr, tt = h[i:i + 4096], r[i:i + 4096], t[i:i + 4096]
            u, inv = torch.unique(tt, return_inverse=True)
            s = self.score(hh, rr, u)
            out.append(s.gather(1, inv.unsqueeze(1)).squeeze(1))
        return torch.cat(out)


class _Vocab:
    def __init__(self, data):
        self.offset, n = {}, 0
        for nt in data.node_types:
            self.offset[nt] = n
            n += data[nt].num_nodes
        self.num_entities = n
        self.range = {nt: (self.offset[nt], self.offset[nt] + data[nt].num_nodes) for nt in data.node_types}


def _training_triples(split, edge_type, vocab):
    """(h, r, t, candidate-type of t) for every forward relation, with
    reciprocal relations appended. Relation 2k is forward, 2k+1 reciprocal."""
    rels = [et for et in split.train.edge_types if not et[1].startswith("rev_")]
    rels.remove(edge_type)
    rels = [edge_type] + rels  # target relation is id 0 (forward), 1 (reciprocal)
    hs, rs, ts, tt = [], [], [], []
    for k, et in enumerate(rels):
        st, _, dt = et
        if et == edge_type:
            store = split.train[et]
            ei = torch.cat([store.edge_index, store.edge_label_index[:, store.edge_label == 1.0]], 1)
            ei = torch.unique(ei, dim=1)
        else:
            ei = split.train[et].edge_index
        h = ei[0] + vocab.offset[st]
        t = ei[1] + vocab.offset[dt]
        hs += [h, t]
        ts += [t, h]
        rs += [torch.full_like(h, 2 * k), torch.full_like(h, 2 * k + 1)]
        tt += [(dt, len(h)), (st, len(h))]
    rel_dst = {}
    for k, et in enumerate(rels):
        rel_dst[2 * k], rel_dst[2 * k + 1] = et[2], et[0]
    return torch.cat(hs), torch.cat(rs), torch.cat(ts), rel_dst, 2 * len(rels)


def _train_epoch(model, opt, H, R, T, rel_dst, vocab, cfg, gen, dev):
    """One pass over the training triples (shared by train_kge_v4 and train_kge_on_graph)."""
    model.train()
    perm = torch.randperm(len(H), generator=gen).to(dev)
    # group a batch by relation so negatives come from one entity type
    r_sorted = R[perm]
    order = perm[torch.argsort(r_sorted, stable=True)]
    rel_of = R[order]
    bounds = torch.nonzero(torch.diff(rel_of)).view(-1) + 1
    starts = [0] + bounds.tolist()
    ends = bounds.tolist() + [len(order)]
    batches = []
    for a, b in zip(starts, ends):
        seg = order[a:b]
        for i in range(0, len(seg), cfg.batch_size):
            batches.append(seg[i:i + cfg.batch_size])
    for j in torch.randperm(len(batches), generator=gen).tolist():
        idx = batches[j]
        h, r, t = H[idx], R[idx], T[idx]
        lo, hi = vocab.range[rel_dst[int(r[0])]]
        n_type = hi - lo
        if cfg.loss == "bce":
            # binary cross-entropy: each true triple against num_neg tails drawn
            # uniformly from the entities of the correct type, per row
            b = len(t)
            neg = torch.randint(lo, hi, (b, cfg.num_neg), generator=gen).to(dev)
            pos_s = model.score_aligned(h, r, t)
            neg_s = model.score_aligned(h, r, neg)
            s_all = torch.cat([pos_s, neg_s.reshape(-1)])
            y = torch.cat([torch.ones_like(pos_s), torch.zeros_like(neg_s.reshape(-1))])
            loss = F.binary_cross_entropy_with_logits(s_all, y)
            if cfg.reg > 0:
                loss = loss + cfg.reg * (model.ent(h).pow(2).sum(-1).mean() + model.ent(t).pow(2).sum(-1).mean()
                                         + model.rel(r).pow(2).sum(-1).mean())
            opt.zero_grad()
            loss.backward()
            opt.step()
            continue
        if n_type <= cfg.num_neg:
            cand = torch.arange(lo, hi, device=dev)
            s = model.score(h, r, cand)
            target = t - lo
        else:
            neg = torch.randint(lo, hi, (cfg.num_neg,), generator=gen).to(dev)
            cand = torch.cat([t, neg])
            s_all = model.score(h, r, cand)            # [b, b + num_neg]
            b = len(t)
            s = torch.cat([s_all[torch.arange(b), torch.arange(b)].unsqueeze(1), s_all[:, b:]], 1)
            target = torch.zeros(b, dtype=torch.long, device=dev)
        loss = F.cross_entropy(s, target)
        if cfg.reg > 0:
            loss = loss + cfg.reg * (model.ent(h).pow(2).sum(-1).mean() + model.ent(t).pow(2).sum(-1).mean()
                                     + model.rel(r).pow(2).sum(-1).mean())
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_kge_v4(split, edge_type, cfg: KGEConfig) -> KGEResultV4:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device
    t0 = time.perf_counter()
    vocab = _Vocab(split.train)
    H, R, T, rel_dst, n_rel = _training_triples(split, edge_type, vocab)
    model = KGEModel(cfg.model, vocab.num_entities, n_rel, cfg.dim, cfg.gamma).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    H, R, T = H.to(dev), R.to(dev), T.to(dev)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)

    src_t, _, dst_t = edge_type
    so, do = vocab.offset[src_t], vocab.offset[dst_t]
    n_dst = split.train[dst_t].num_nodes
    dst_all = torch.arange(n_dst, device=dev) + do

    def pos(store):
        return store.edge_label_index[:, store.edge_label == 1.0]
    val_pos, test_pos = pos(split.val[edge_type]), pos(split.test[edge_type])
    known_val = known_from_edges(split.val[edge_type].edge_index, val_pos)
    known_test = known_from_edges(split.test[edge_type].edge_index, val_pos, test_pos)

    def rows(ids):
        h = ids + so
        return model.score(h, torch.zeros_like(h), dst_all)

    best, best_state, bad, epochs = -1.0, None, 0, 0
    for epoch in range(cfg.max_epochs):
        _train_epoch(model, opt, H, R, T, rel_dst, vocab, cfg, gen, dev)
        epochs = epoch + 1
        if epochs % cfg.eval_every == 0 or epochs == cfg.max_epochs:
            model.eval()
            v = summarize_ranks(filtered_ranks(rows, val_pos, known_val, device=dev)["realistic"])["mrr"]
            if v > best:
                best, bad = v, 0
                best_state = {k: x.detach().clone() for k, x in model.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        vr = filtered_ranks(rows, val_pos, known_val, device=dev)
        tr = filtered_ranks(rows, test_pos, known_test, device=dev)

        def pair_scores(store):
            ei = store.edge_label_index.to(dev)
            h = ei[0] + so
            return model.score_pairs(h, torch.zeros_like(h), ei[1] + do).cpu().numpy()
        vs, ts = pair_scores(split.val[edge_type]), pair_scores(split.test[edge_type])
    vl, tl = split.val[edge_type].edge_label.numpy(), split.test[edge_type].edge_label.numpy()
    return KGEResultV4(
        config=cfg,
        val_ranking_v4=summarize_ranks(vr["realistic"]),
        test_ranking_v4={p: summarize_ranks(r) for p, r in tr.items()},
        test_metrics=binary_link_metrics(tl, ts, threshold=best_f1_threshold(vl, vs)),
        test_metrics_acc_threshold=binary_link_metrics(tl, ts, threshold=best_accuracy_threshold(vl, vs)),
        test_ranks_v4={"realistic": tr["realistic"]},
        test_scores_v4={"labels": tl, "probs": ts, "val_labels": vl, "val_probs": vs},
        epochs_run=epochs,
        wall_clock_seconds=time.perf_counter() - t0,
    )


def train_kge_on_graph(train_data, edge_type, cfg: KGEConfig, epochs: int):
    """Train on every forward relation of `train_data` for a fixed number of
    epochs, with no validation data (used where no split exists, e.g. the
    substitution scenario; the epoch count comes from tuning). Returns the
    model and the entity offsets per node type. In train_data the target
    relation's edges are all training edges (edge_label_index may be absent)."""
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    dev = cfg.device

    class _S:  # minimal split view for _training_triples
        pass
    sv = _S()
    sv.train = train_data
    st = train_data[edge_type]
    if "edge_label_index" not in st:
        st.edge_label_index = torch.zeros(2, 0, dtype=torch.long)
        st.edge_label = torch.zeros(0)
    vocab = _Vocab(train_data)
    H, R, T, rel_dst, n_rel = _training_triples(sv, edge_type, vocab)
    model = KGEModel(cfg.model, vocab.num_entities, n_rel, cfg.dim, cfg.gamma).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    H, R, T = H.to(dev), R.to(dev), T.to(dev)
    gen = torch.Generator(device="cpu").manual_seed(cfg.seed)
    for _ in range(epochs):
        _train_epoch(model, opt, H, R, T, rel_dst, vocab, cfg, gen, dev)
    model.eval()
    return model, vocab.offset
