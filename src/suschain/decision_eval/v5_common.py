"""Shared setup for the v5 substitution experiments (tuning and confirmatory).

  load_context()    AnonNet graph, product matrix, risk, scenario builder inputs
  scenario_for()    visible graph, queries and tie-breaks of one seed
  train_scorer()    one method -> a [n_customers, n] score matrix, plus timing
  risk_curves()     per-query hit@10 and risk@10 under every shift weight

A method is a dict {"family": "gnn"|"kge"|"nbfnet", ...}. Scores are always for
the customers of the scenario's queries, aligned with `cidx`.
"""
from __future__ import annotations

import time

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata
from suschain.decision_eval.gnn_scenario_v5 import train_gnn_v5
from suschain.decision_eval.substitutes import (build_substitution_scenario, evaluate_substitution,
                                                idf_product_matrix, product_items, rank_candidates)
from suschain.enrichment.anonnet_sustainability import compute_composite_score
from suschain.enrichment.product_taxonomy import classify_products_column
from suschain.models.kge_torch import KGEConfig, train_kge_on_graph
from suschain.models.nbfnet import nbfnet_scores, train_nbfnet
from suschain.training.multi_objective_loss import normalize_risk

ET = ("node", "supplies", "node")
K = 10
BETAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]


class Context:
    pass


def load_context(device: str) -> Context:
    c = Context()
    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=False)
    c.data, nm = built.data, built.id_maps["node"]
    c.n = c.data["node"].num_nodes
    nodes = g.nodes.set_index("id").loc[sorted(nm, key=nm.get)]
    c.X = idf_product_matrix(nodes["products_list"].apply(product_items).tolist())
    c.E = c.data[ET].edge_index.numpy()
    cats = classify_products_column(nodes["products"].reset_index(drop=True))
    comp = compute_composite_score(nodes["location"].reset_index(drop=True), cats)
    c.risk_norm = normalize_risk(torch.tensor(comp["composite_score"].to_numpy(), dtype=torch.float32))
    c.risk = c.risk_norm.numpy()
    c.device = device
    return c


class Scenario:
    pass


def scenario_for(ctx: Context, seed: int, tau: float = 0.1) -> Scenario:
    sc = build_substitution_scenario(ctx.E, ctx.X, tau=tau, seed=seed)
    s = Scenario()
    s.sc, s.qs, s.tr, s.seed = sc, sc.queries, sc.train_edges, seed
    s.tb = np.random.default_rng(seed).random(ctx.n)
    A = sp.csr_matrix((np.ones(s.tr.shape[1]), (s.tr[0], s.tr[1])), shape=(ctx.n, ctx.n))
    A.data[:] = 1.0
    s.pop = np.asarray(A.sum(axis=1)).ravel()
    s.co = (A @ A.T).tocsr()
    s.customers = sorted({q.customer for q in s.qs})
    s.cidx = {c: i for i, c in enumerate(s.customers)}
    return s


def co_supply_scores(sc: Scenario, q):
    return sc.co[q.disrupted].toarray().ravel()


def train_scorer(ctx: Context, sc: Scenario, method: dict, lam: float = 0.0, log=None):
    """Returns (S, seconds): S[cidx[customer]] is a length-n score vector."""
    dev, fam = ctx.device, method["family"]
    t0 = time.perf_counter()
    if fam == "gnn":
        sup = torch.from_numpy(sc.tr.copy())
        mp = torch.from_numpy(np.concatenate([sc.tr, sc.tr[[1, 0]]], axis=1).copy())
        z, _, _ = train_gnn_v5(ctx.data, ET, sup, ctx.risk_norm, lam, sc.seed, device=dev,
                               conv_kind=method["conv"], hp=method["hp"], mp_edge_index=mp)
        S = (z[torch.tensor(sc.customers, device=z.device)] @ z.T).cpu().numpy()
    elif fam == "kge":
        td = HeteroData()
        td["node"].num_nodes = ctx.n
        td["node"].node_id = torch.arange(ctx.n)
        td[ET].edge_index = torch.from_numpy(sc.tr.copy())
        hp = dict(method["hp"])
        epochs = hp.pop("epochs")
        cfg = KGEConfig(model=method["model"], seed=sc.seed, device=dev, **hp)
        model, _ = train_kge_on_graph(td, ET, cfg, epochs)
        with torch.no_grad():
            allv = torch.arange(ctx.n, device=dev)
            h = torch.tensor(sc.customers, device=dev)
            S = torch.cat([model.score(h[i:i + 512], torch.ones_like(h[i:i + 512]), allv)
                           for i in range(0, len(h), 512)]).cpu().numpy()
    elif fam == "nbfnet":
        model, steps, _ = train_nbfnet(torch.from_numpy(sc.tr.copy()), ctx.n, sc.seed, dev,
                                       seconds=method.get("seconds", 600.0), dim=method.get("dim", 32),
                                       layers=method.get("layers", 3), lr=method.get("lr", 5e-3), log=log)
        S = nbfnet_scores(model, sc.customers, dev)
    else:
        raise ValueError(fam)
    return S, time.perf_counter() - t0


def evaluate(ctx, sc, S):
    """Plain ranking by S: per-query arrays from evaluate_substitution."""
    ranked = [rank_candidates(S[sc.cidx[q.customer]], q, sc.tb) for q in sc.qs]
    return evaluate_substitution(ranked, sc.qs, ctx.risk, K)


def risk_curves(ctx, sc, S):
    """Per-query hit@10 and risk@10 for every shift weight in BETAS, after
    standardising the scores within each query's candidates.
    Returns hit, risk: arrays [len(BETAS), n_queries]."""
    hit = np.zeros((len(BETAS), len(sc.qs)))
    rsk = np.zeros_like(hit)
    for j, q in enumerate(sc.qs):
        s = np.asarray(S[sc.cidx[q.customer]], dtype=np.float64)
        x = s[q.candidates]
        sd = x.std()
        z = np.zeros_like(s)
        z[q.candidates] = (x - x.mean()) / (sd if sd > 0 else 1.0)
        for i, b in enumerate(BETAS):
            out = evaluate_substitution([rank_candidates(z - b * ctx.risk, q, sc.tb)], [q], ctx.risk, K)
            hit[i, j], rsk[i, j] = out["hit"][0], out["risk"][0]
    return hit, rsk
