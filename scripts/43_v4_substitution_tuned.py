"""v4: the AnonNet substitution task with TUNED learned rankers.

Script 36 used one fixed GNN configuration. This script reruns the learned
rankers with the configurations tuned on validation MRR by scripts/41
(configs/model_v4/anonnet_*.json), and adds the four KGE models, so that the
strongest link-prediction family is also tested as a substitute ranker.

Same scenarios, tie-breaks and risk as script 36 (same seeds), so results
pair with v4_anonnet_substitution_tau0.1.json seed by seed.

Methods
  co_supply                      reference (recomputed)
  gnn_{conv}_tuned_lam{0,1}      tuned GNN link score, plain and learned penalty
  gnn_{conv}_tuned_shift_beta{B} plain tuned GNN, score - B * risk
  fused_{conv}_tuned             RRF of GNN link score, GNN supplier similarity, co-supply
  kge_{m}                        KGE score of (candidate, supplies, customer)
  kge_{m}_shift_beta{B}          the same, minus B * risk
  fused_kge_{m}                  RRF of KGE score, KGE entity similarity, co-supply

Risk-budget re-ranking (the practical procedure suggested by the shift result)
  For a ranker with score vector s, candidates of each query are ranked by
  z(s) - beta * risk, where z standardises s within the query's candidates,
  so one beta has the same meaning for every ranker. Customers are split at
  random into a calibration half and an evaluation half. For a target risk
  reduction t (5, 10, 20, 30%), beta is the smallest grid value whose mean
  risk@10 on the calibration half is at least t below beta = 0; the
  evaluation half reports the reduction achieved and the hit@10 lost.
  Applied to each tuned GNN (raw scores) and, through rank-position scores, to
  co-supply and to each fused ranking.

KGE models train for a fixed number of epochs: the epoch count at which the
best tuning trial stopped (no validation data exists inside the scenario).

Output: results/metrics/v4_substitution_tuned_tau{tau}.json
Run:    python -u scripts/43_v4_substitution_tuned.py
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata
from suschain.decision_eval.scenario_trainer import train_scenario_model
from suschain.decision_eval.substitutes import (build_substitution_scenario, evaluate_substitution,
                                                idf_product_matrix, product_items, rank_candidates)
from suschain.enrichment.anonnet_sustainability import compute_composite_score
from suschain.enrichment.product_taxonomy import classify_products_column
from suschain.eval.significance import exact_wilcoxon
from suschain.models.kge_torch import KGEConfig, train_kge_on_graph
from suschain.training.multi_objective_loss import normalize_risk

_spec = importlib.util.spec_from_file_location("s36", ROOT / "scripts" / "36_v4_anonnet_substitution.py")
_s36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s36)
rrf_order = _s36.rrf_order

ET = ("node", "supplies", "node")
K = 10
BETAS = [1.0, 3.0, 10.0]
BUDGET_BETAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
TARGETS = [0.05, 0.10, 0.20, 0.30]


def rank_score(rankings, qs, n):
    """score_of(q) for a precomputed ranking: minus the rank position."""
    lookup = {}
    for r, q in zip(rankings, qs):
        s = np.zeros(n)
        s[r] = -np.arange(len(r), dtype=np.float64)
        lookup[id(q)] = s
    return lambda q: lookup[id(q)]


def risk_budget(score_of, qs, risk, tb, seed):
    """score_of(q) -> full score vector. Returns per-target calibration results
    and the full beta curve on the evaluation half."""
    hit = np.zeros((len(BUDGET_BETAS), len(qs)))
    rsk = np.zeros((len(BUDGET_BETAS), len(qs)))
    for j, q in enumerate(qs):
        s = np.asarray(score_of(q), dtype=np.float64)
        x = s[q.candidates]
        sd = x.std()
        z = np.zeros_like(s)
        z[q.candidates] = (x - x.mean()) / (sd if sd > 0 else 1.0)
        for i, b in enumerate(BUDGET_BETAS):
            out = evaluate_substitution([rank_candidates(z - b * risk, q, tb)], [q], risk, K)
            hit[i, j], rsk[i, j] = out["hit"][0], out["risk"][0]
    custs = sorted({q.customer for q in qs})
    flag = dict(zip(custs, np.random.default_rng(seed + 1000).random(len(custs)) < 0.5))
    cal = np.array([flag[q.customer] for q in qs])
    ev = ~cal
    red_cal = 1 - rsk[:, cal].mean(1) / rsk[0, cal].mean()
    red_ev = 1 - rsk[:, ev].mean(1) / rsk[0, ev].mean()
    hit_ev = hit[:, ev].mean(1)
    res = {"curve_eval": {"beta": BUDGET_BETAS, "risk_reduction": red_ev.tolist(),
                          "hit@10": hit_ev.tolist()}}
    for t in TARGETS:
        ok = np.flatnonzero(red_cal >= t)
        i = int(ok[0]) if len(ok) else len(BUDGET_BETAS) - 1
        res[f"target{int(t * 100)}"] = {"beta": BUDGET_BETAS[i], "reachable_on_calibration": bool(len(ok)),
                                        "achieved_reduction_eval": float(red_ev[i]),
                                        "hit_lost_eval": float(hit_ev[0] - hit_ev[i])}
    return res
CONF = ROOT / "configs" / "model_v4"
GNN_KEYS = ("embedding_dim", "hidden_channels", "out_channels", "num_layers", "lr", "weight_decay")


def load_conf(name):
    d = json.loads((CONF / f"anonnet_{name}.json").read_text())
    best = d["best_params"]
    epochs = next((t["epochs_run"] for t in d["trials"] if t["params"] == best), None)
    return best, epochs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--convs", default="sage,gat")
    ap.add_argument("--kges", default="TransE,DistMult,ComplEx,RotatE")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in a.seeds.split(",")]
    convs = [c for c in a.convs.split(",") if c]
    kges = [m for m in a.kges.split(",") if m]

    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=False)
    data, nm = built.data, built.id_maps["node"]
    n = data["node"].num_nodes
    nodes = g.nodes.set_index("id").loc[sorted(nm, key=nm.get)]
    X = idf_product_matrix(nodes["products_list"].apply(product_items).tolist())
    E = data[ET].edge_index.numpy()
    cats = classify_products_column(nodes["products"].reset_index(drop=True))
    comp = compute_composite_score(nodes["location"].reset_index(drop=True), cats)
    risk_norm = normalize_risk(torch.tensor(comp["composite_score"].to_numpy(), dtype=torch.float32).to(a.device))
    risk = risk_norm.cpu().numpy()

    confs = {}
    for c in convs:
        p, _ = load_conf(c)
        confs[c] = {k: p[k] for k in GNN_KEYS}
    for m in kges:
        p, ep = load_conf(m)
        confs[m] = {"params": p, "epochs": ep}

    per_seed = defaultdict(lambda: defaultdict(list))

    def record(method, ranked, qs):
        out = evaluate_substitution(ranked, qs, risk, K)
        for key in ("hit", "recall", "risk"):
            per_seed[method][f"{key}@{K}"].append(float(np.nanmean(out[key])))
        per_seed[method]["rr"].append(float(np.nanmean(out["rr"])))

    all_budgets = []
    for seed in seeds:
        t0 = time.perf_counter()
        sc = build_substitution_scenario(E, X, tau=a.tau, seed=seed)
        qs = sc.queries
        rng = np.random.default_rng(seed)
        tb = rng.random(n)
        tr = sc.train_edges
        A = sp.csr_matrix((np.ones(tr.shape[1]), (tr[0], tr[1])), shape=(n, n))
        A.data[:] = 1.0
        pop = np.asarray(A.sum(axis=1)).ravel()
        co = (A @ A.T).tocsr()
        r_co = [rank_candidates(co[q.disrupted].toarray().ravel(), q, pop + tb) for q in qs]
        record("co_supply", r_co, qs)
        budgets = {}
        # rank-based scores (position in the ranking, tie-breaks included)
        # give rule-based and fused rankers a smooth score to shift
        budgets["co_supply"] = risk_budget(rank_score(r_co, qs, n), qs, risk, tb, seed)
        custs = sorted({q.customer for q in qs})
        cidx = {c: i for i, c in enumerate(custs)}
        cust_t = torch.tensor(custs)

        sup = torch.from_numpy(tr[[1, 0]].copy())
        mp = torch.from_numpy(np.concatenate([tr, tr[[1, 0]]], axis=1).copy())
        for conv in convs:
            for lam in (0.0, 1.0):
                zc, zs = train_scenario_model(data, ET, None, sup, risk_norm, lam, seed, device=a.device,
                                              conv_kind=conv, mp_edge_index=mp, hparams=confs[conv])
                S = (zc[cust_t.to(zc.device)] @ zs.T).cpu().numpy()
                r_gnn = [rank_candidates(S[cidx[q.customer]], q, tb) for q in qs]
                record(f"gnn_{conv}_tuned_lam{lam:g}", r_gnn, qs)
                if lam == 0.0:
                    zs_np = zs.cpu().numpy()
                    r_sup = [rank_candidates(zs_np @ zs_np[q.disrupted], q, tb) for q in qs]
                    r_f = [rrf_order([x, y, z], q) for x, y, z, q in zip(r_gnn, r_sup, r_co, qs)]
                    record(f"fused_{conv}_tuned", r_f, qs)
                    budgets[f"fused_{conv}_tuned"] = risk_budget(rank_score(r_f, qs, n), qs, risk, tb, seed)
                    budgets[f"gnn_{conv}_tuned"] = risk_budget(lambda q: S[cidx[q.customer]], qs, risk, tb, seed)
                    for B in BETAS:
                        record(f"gnn_{conv}_tuned_shift_beta{B:g}",
                               [rank_candidates(S[cidx[q.customer]] - B * risk, q, tb) for q in qs], qs)

        td = HeteroData()
        td["node"].num_nodes = n
        td["node"].node_id = torch.arange(n)
        td[ET].edge_index = torch.from_numpy(tr.copy())
        for m in kges:
            cfg = KGEConfig(model=m, seed=seed, device=a.device, **confs[m]["params"])
            model, off = train_kge_on_graph(td, ET, cfg, confs[m]["epochs"])
            with torch.no_grad():
                allv = torch.arange(n, device=a.device)
                h = cust_t.to(a.device)
                # reciprocal relation 1: (customer, supplies^-1, ?) ranks suppliers
                S = torch.cat([model.score(h[i:i + 512], torch.ones_like(h[i:i + 512]), allv)
                               for i in range(0, len(h), 512)]).cpu().numpy()
                Z = model.ent.weight.detach().cpu().numpy()
            r_k = [rank_candidates(S[cidx[q.customer]], q, tb) for q in qs]
            record(f"kge_{m}", r_k, qs)
            budgets[f"kge_{m}"] = risk_budget(lambda q: S[cidx[q.customer]], qs, risk, tb, seed)
            r_ks = [rank_candidates(Z @ Z[q.disrupted], q, tb) for q in qs]
            record(f"fused_kge_{m}", [rrf_order([x, y, z], q) for x, y, z, q in zip(r_k, r_ks, r_co, qs)], qs)
            for B in BETAS:
                record(f"kge_{m}_shift_beta{B:g}",
                       [rank_candidates(S[cidx[q.customer]] - B * risk, q, tb) for q in qs], qs)
        all_budgets.append(budgets)
        line = " | ".join(f"{mm} {per_seed[mm]['hit@10'][-1]:.3f}" for mm in per_seed
                          if "shift" not in mm)
        print(f"seed {seed} ({time.perf_counter() - t0:.0f}s): {line}", flush=True)

    summary = {m: {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                   for k, v in d.items()} for m, d in per_seed.items()}
    tests = {m: {k: exact_wilcoxon(per_seed[m][k], per_seed["co_supply"][k]) for k in ("hit@10", "rr")}
             for m in per_seed if m != "co_supply" and len(seeds) > 1}
    out = {"seeds": seeds, "tau": a.tau, "k": K, "configs": confs,
           "note": "learned rankers tuned on link-prediction validation MRR (scripts/41); "
                   "all tests exploratory, vs co_supply",
           "per_seed": {m: dict(d) for m, d in per_seed.items()}, "summary": summary,
           "wilcoxon_vs_co_supply": tests,
           "risk_budget": {"betas": BUDGET_BETAS, "targets": TARGETS, "per_seed": all_budgets,
                           "summary": {m: {f"target{int(t * 100)}": {
                               k: float(np.mean([b[m][f"target{int(t * 100)}"][k] for b in all_budgets]))
                               for k in ("beta", "achieved_reduction_eval", "hit_lost_eval")}
                               for t in TARGETS} for m in all_budgets[0]}}}
    p = ROOT / "results" / "metrics" / f"v4_substitution_tuned_tau{a.tau:g}{a.tag}.json"
    p.write_text(json.dumps(out, indent=2))
    for m, d in summary.items():
        print(f"{m:32} hit@10 {d['hit@10']['mean']:.4f} rr {d['rr']['mean']:.4f} risk@10 {d['risk@10']['mean']:.4f}")
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
