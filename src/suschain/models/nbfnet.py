"""A compact Neural Bellman-Ford network (Zhu et al., NeurIPS 2021) for one
relation and its inverse, written for the substitution scenario.

NBFNet scores a candidate by running a generalised Bellman-Ford iteration that
starts at the query head and sums path representations; it is the model used
for supply-chain link prediction by Kosasih et al. (IJPR 2025). This is our own
implementation of the published idea, not the authors' code, and differs in
four ways that keep it small and cheap enough for a 5 GB GPU:

  - MESSAGE is the DistMult product with one learned vector per layer and
    relation (static, not conditioned on the query: there is one query relation);
  - AGGREGATE is a sum divided by sqrt(in-degree + 1), not PNA;
  - the update is ReLU(LayerNorm(Linear([aggregate + boundary ; previous]))),
    the final score an MLP of [state ; query vector];
  - training removes the supervised link and its reverse from the graph for
    each query, as in the paper, and uses a softmax over all nodes with the
    customer's other visible suppliers filtered out.

Query: head = customer, relation = "supplied by"; the answer is the supplier.
Edge ids: forward edge j is (supplier_j -> customer_j) with relation 0, its
reverse j + m is (customer_j -> supplier_j) with relation 1.
"""
from __future__ import annotations

import time

import torch
import torch.nn.functional as F
from torch import nn


class NBFNet(nn.Module):
    def __init__(self, n_nodes: int, train_edges: torch.Tensor, dim: int = 32, layers: int = 3):
        super().__init__()
        m = train_edges.shape[1]
        self.n, self.m, self.dim, self.L = n_nodes, m, dim, layers
        src = torch.cat([train_edges[0], train_edges[1]])
        dst = torch.cat([train_edges[1], train_edges[0]])
        rel = torch.cat([torch.zeros(m, dtype=torch.long), torch.ones(m, dtype=torch.long)])
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("rel_of_edge", rel)
        deg = torch.zeros(n_nodes).index_add_(0, dst, torch.ones_like(dst, dtype=torch.float))
        self.register_buffer("norm", (1.0 / torch.sqrt(deg + 1.0)).view(-1, 1, 1))
        self.rel = nn.Parameter(torch.randn(layers, 2, dim) * 0.1 + 1.0)
        self.query = nn.Parameter(torch.randn(dim) * 0.1)
        self.lins = nn.ModuleList(nn.Linear(2 * dim, dim) for _ in range(layers))
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))
        self.out = nn.Sequential(nn.Linear(2 * dim, dim), nn.ReLU(), nn.Linear(dim, 1))

    def forward(self, heads: torch.Tensor, drop_edges: torch.Tensor | None = None) -> torch.Tensor:
        """heads: [B] customer ids. drop_edges: [B] forward-edge ids whose edge
        and reverse are hidden for that batch element (training). Returns [B, N]."""
        B, N, d = len(heads), self.n, self.dim
        dev = heads.device
        boundary = torch.zeros(N, B, d, device=dev)
        boundary[heads, torch.arange(B, device=dev)] = self.query
        mask = None
        if drop_edges is not None:
            mask = torch.ones(self.src.numel(), B, 1, device=dev)
            ar = torch.arange(B, device=dev)
            mask[drop_edges, ar] = 0
            mask[drop_edges + self.m, ar] = 0
        h = boundary
        for layer in range(self.L):
            msg = h[self.src] * self.rel[layer][self.rel_of_edge].unsqueeze(1)
            if mask is not None:
                msg = msg * mask
            agg = torch.zeros(N, B, d, device=dev).index_add_(0, self.dst, msg) * self.norm
            new = F.relu(self.norms[layer](self.lins[layer](torch.cat([agg + boundary, h], -1))))
            h = new + h if layer > 0 else new
        q = self.query.view(1, 1, d).expand(N, B, d)
        return self.out(torch.cat([h, q], -1)).squeeze(-1).T


def train_nbfnet(train_edges: torch.Tensor, n_nodes: int, seed: int, device: str, *,
                 seconds: float = 600.0, batch: int = 16, lr: float = 5e-3, dim: int = 32,
                 layers: int = 3, log=None):
    """Trains for a wall-clock budget. Returns (model, steps, seconds)."""
    t0 = time.perf_counter()
    torch.manual_seed(seed)
    m = train_edges.shape[1]
    model = NBFNet(n_nodes, train_edges, dim, layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    cus, sup = train_edges[1].to(device), train_edges[0].to(device)
    known: dict[int, list[int]] = {}
    for s, c in train_edges.T.tolist():
        known.setdefault(c, []).append(s)
    gen = torch.Generator().manual_seed(seed)
    steps = 0
    while time.perf_counter() - t0 < seconds:
        idx = torch.randint(0, m, (batch,), generator=gen).to(device)
        heads, target = cus[idx], sup[idx]
        model.train()
        opt.zero_grad()
        logits = model(heads, drop_edges=idx)
        filt = torch.zeros_like(logits, dtype=torch.bool)
        for b, (c, t) in enumerate(zip(heads.tolist(), target.tolist())):
            others = [x for x in known[c] if x != t]
            if others:
                filt[b, others] = True
        logits = logits.masked_fill(filt, float("-inf"))
        loss = F.cross_entropy(logits, target)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        steps += 1
        if log and steps % 200 == 0:
            log(f"  nbfnet step {steps} loss {loss.item():.3f}")
    model.eval()
    return model, steps, time.perf_counter() - t0


@torch.no_grad()
def nbfnet_scores(model: NBFNet, customers: list[int], device: str, batch: int = 32):
    """[len(customers), N] scores with the full visible graph."""
    out = []
    for i in range(0, len(customers), batch):
        out.append(model(torch.tensor(customers[i:i + batch], device=device)).cpu())
    return torch.cat(out).numpy()
