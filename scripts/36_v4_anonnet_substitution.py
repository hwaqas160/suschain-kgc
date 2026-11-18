"""v4 work packages 3 and 4: supplier substitution on AnonNet with held-out
ground truth, and the sustainability mechanisms on that task.

Task (src/suschain/decision_eval/substitutes.py): a customer loses a visible
supplier s; methods rank the firms capable of replacing s; a recommendation is
correct if it is a real supplier of that customer which was hidden from every
method and is product-similar to s.

Methods
  random, popularity              controls
  product_sim                     similarity of each candidate's products to s's
  co_supply                       customers shared with s in the visible graph
  gnn_{conv}_lam{L}               GNN link score for (customer, candidate); penalty weight L
  gnn_{conv}_shift_beta{B}        plain GNN score - B * risk; no retraining
  gnn_{conv}_filter_pool{P}       plain GNN's top P re-ranked by risk
  gnn_{conv}_supsim_lam{L}        similarity of candidate and failed-supplier embeddings
  fused_{conv}_lam{L}             reciprocal-rank fusion of GNN link score, embedding
                                  similarity and co-supply; conditions on customer AND
                                  failed supplier
  fused_heuristic                 the same fusion over popularity, product_sim and
                                  co_supply only (the non-learned comparison)
  fused_{conv}_filter_pool{P}     plain fused ranking, top P re-ranked by risk

Fusion is reciprocal-rank fusion with the conventional constant 60
(Cormack et al., 2009), so no weight is tuned on the test queries.

Risk is the country-and-sector carbon-intensity proxy (enrichment/
anonnet_sustainability.py), min-max normalised. Firms without a location have
no score; they are ranked at the mean score (the policy the original code used),
and the share of recommended firms without a score is reported for every method.

Outputs: results/metrics/v4_anonnet_substitution_tau{tau}.json and .npz

Run:   python -u scripts/36_v4_anonnet_substitution.py [--tau 0.1]
Smoke: python -u scripts/36_v4_anonnet_substitution.py --smoke --device cpu
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata
from suschain.decision_eval.scenario_trainer import train_scenario_model
from suschain.decision_eval.substitutes import (
    build_substitution_scenario,
    evaluate_substitution,
    idf_product_matrix,
    product_items,
    rank_candidates,
)
from suschain.enrichment.anonnet_sustainability import compute_composite_score
from suschain.enrichment.product_taxonomy import classify_products_column
from suschain.eval.significance import exact_wilcoxon
from suschain.training.multi_objective_loss import normalize_risk

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
ET = ("node", "supplies", "node")
KS = (5, 10)
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
BETAS = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
POOLS = [20, 50, 100]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--convs", default="sage,gat")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lambdas", default=",".join(f"{x:g}" for x in LAMBDAS))
    ap.add_argument("--tag", default="", help="suffix for the output file names")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    a.seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in a.seeds.split(",")]
    a.convs = a.convs.split(",")
    a.lambdas = [float(x) for x in a.lambdas.split(",")]
    return a


def filtered_order(scores, q, risk, pool, tb):
    """Plain ranking with its top `pool` re-ordered by ascending risk."""
    ranked = rank_candidates(scores, q, tb)
    head, tail = ranked[:pool], ranked[pool:]
    head = head[np.argsort(risk[head], kind="stable")]
    return np.concatenate([head, tail])


RRF_K = 60


def rrf_order(rankings, q):
    """Reciprocal-rank fusion of several full rankings of q.candidates."""
    score = np.zeros(len(q.candidates))
    pos_of = {v: i for i, v in enumerate(q.candidates)}
    for r in rankings:
        idx = np.fromiter((pos_of[v] for v in r), dtype=np.int64, count=len(r))
        score[idx] += 1.0 / (RRF_K + np.arange(1, len(r) + 1))
    return q.candidates[np.lexsort((q.candidates, -score))]


def main():
    a = parse_args()
    lambdas, seeds, convs, tag = a.lambdas, a.seeds, a.convs, a.tag
    if a.smoke:
        lambdas, seeds, convs, tag = [0.0, 1.0], [0], ["sage"], "_smoke"

    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=False)
    data = built.data
    nm = built.id_maps["node"]
    n = data["node"].num_nodes
    order = sorted(nm, key=nm.get)
    nodes = g.nodes.set_index("id").loc[order]
    X = idf_product_matrix(nodes["products_list"].apply(product_items).tolist())
    E = data[ET].edge_index.numpy()

    cats = classify_products_column(nodes["products"].reset_index(drop=True))
    comp = compute_composite_score(nodes["location"].reset_index(drop=True), cats)
    risk_raw = torch.tensor(comp["composite_score"].to_numpy(), dtype=torch.float32)
    scored = ~torch.isnan(risk_raw).numpy()
    risk_norm = normalize_risk(risk_raw.to(a.device))
    risk = risk_norm.cpu().numpy()
    coverage = {
        "firms": int(n),
        "location_present": int(nodes["location"].notna().sum()),
        "risk_scored": int(scored.sum()),
        "risk_scored_pct": float(scored.mean() * 100),
        "product_category_other_pct": float((cats == "other").mean() * 100),
        "missing_risk_policy": "ranked at the mean of the normalised scores",
    }
    print("risk coverage:", coverage, flush=True)

    per_query, per_seed = {}, defaultdict(lambda: defaultdict(list))
    timings, scen_stats = [], []

    def record(seed, method, ranked, queries):
        for k in KS:
            out = evaluate_substitution(ranked, queries, risk, k)
            unscored = np.array([(~scored[r[:k]]).mean() for r in ranked])
            for m, v in {**out, "unscored": unscored}.items():
                if m == "rr" and k != KS[-1]:
                    continue
                key = m if m == "rr" else f"{m}@{k}"
                per_query[f"{method}/{seed}/{key}"] = v.astype(np.float32)
                per_seed[method][key].append(float(np.nanmean(v)))

    for seed in seeds:
        sc = build_substitution_scenario(E, X, tau=a.tau, seed=seed)
        qs = sc.queries
        scen_stats.append({"seed": seed, "queries": len(qs),
                           "customers": len({q.customer for q in qs}),
                           "mean_truth": float(np.mean([len(q.truth) for q in qs])),
                           "median_candidates": float(np.median([len(q.candidates) for q in qs]))})
        rng = np.random.default_rng(seed)
        tb = rng.random(n)                                   # random tie-break
        tr = sc.train_edges
        A = sp.csr_matrix((np.ones(tr.shape[1]), (tr[0], tr[1])), shape=(n, n))  # supplier x customer
        A.data[:] = 1.0
        pop = np.asarray(A.sum(axis=1)).ravel()
        co = (A @ A.T).tocsr()

        record(seed, "random", [rank_candidates(rng.random(n), q) for q in qs], qs)
        r_pop = [rank_candidates(pop, q, tb) for q in qs]
        r_sim = [rank_candidates((X @ X[q.disrupted].T).toarray().ravel(), q, tb) for q in qs]
        r_co = [rank_candidates(co[q.disrupted].toarray().ravel(), q, pop + tb) for q in qs]
        record(seed, "popularity", r_pop, qs)
        record(seed, "product_sim", r_sim, qs)
        record(seed, "co_supply", r_co, qs)
        record(seed, "fused_heuristic",
               [rrf_order([a_, b_, c_], q) for a_, b_, c_, q in zip(r_pop, r_sim, r_co, qs)], qs)

        # supervision: customer -> supplier, so the ranked side is the destination
        sup = torch.from_numpy(tr[[1, 0]].copy())
        mp = torch.from_numpy(np.concatenate([tr, tr[[1, 0]]], axis=1).copy())
        custs = sorted({q.customer for q in qs})
        cidx = {c: i for i, c in enumerate(custs)}
        for conv in convs:
            for lam in lambdas:
                t0 = time.perf_counter()
                zc, zs = train_scenario_model(data, ET, None, sup, risk_norm, lam, seed,
                                              device=a.device, conv_kind=conv, mp_edge_index=mp)
                S = (zc[torch.tensor(custs, device=zc.device)] @ zs.T).cpu().numpy()
                zs_np = zs.cpu().numpy()
                timings.append(time.perf_counter() - t0)
                r_gnn = [rank_candidates(S[cidx[q.customer]], q, tb) for q in qs]
                r_sup = [rank_candidates(zs_np @ zs_np[q.disrupted], q, tb) for q in qs]
                r_fus = [rrf_order([a_, b_, c_], q) for a_, b_, c_, q in zip(r_gnn, r_sup, r_co, qs)]
                record(seed, f"gnn_{conv}_lam{lam:g}", r_gnn, qs)
                record(seed, f"gnn_{conv}_supsim_lam{lam:g}", r_sup, qs)
                record(seed, f"fused_{conv}_lam{lam:g}", r_fus, qs)
                if lam == 0.0:
                    for B in BETAS:
                        record(seed, f"gnn_{conv}_shift_beta{B:g}",
                               [rank_candidates(S[cidx[q.customer]] - B * risk, q, tb) for q in qs], qs)
                    for P in POOLS:
                        record(seed, f"gnn_{conv}_filter_pool{P}",
                               [filtered_order(S[cidx[q.customer]], q, risk, P, tb) for q in qs], qs)
                        record(seed, f"fused_{conv}_filter_pool{P}",
                               [np.concatenate([r[:P][np.argsort(risk[r[:P]], kind="stable")], r[P:]])
                                for r in r_fus], qs)
        line = " | ".join(f"{m} hit@10={per_seed[m]['hit@10'][-1]:.3f} risk@10={per_seed[m]['risk@10'][-1]:.3f}"
                          for m in ["random", "popularity", "product_sim", "co_supply",
                                    "fused_heuristic", f"gnn_{convs[0]}_lam0",
                                    f"fused_{convs[0]}_lam0"])
        print(f"seed {seed} ({len(qs)} queries): {line} | train {np.mean(timings):.1f}s", flush=True)

    tests = {}
    for conv in convs:
        base = per_seed[f"gnn_{conv}_lam0"]
        for m, v in per_seed.items():
            if m != f"gnn_{conv}_lam0":
                tests[f"{m}_vs_gnn_{conv}_lam0"] = {
                    key: exact_wilcoxon(v[key], base[key]) for key in ("hit@10", "risk@10", "rr")}
    summary = {m: {key: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                   for key, v in mv.items()} for m, mv in per_seed.items()}
    out = {"seeds": seeds, "tau": a.tau, "ks": KS, "convs": convs, "lambdas": lambdas,
           "betas": BETAS, "pools": POOLS, "edge_direction": "source supplies target (assumed; see docs/v4)",
           "risk_coverage": coverage, "scenario_stats": scen_stats,
           "per_seed": {m: dict(v) for m, v in per_seed.items()}, "summary": summary,
           "wilcoxon_exploratory": tests,
           "mean_train_seconds": float(np.mean(timings)) if timings else None}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"v4_anonnet_substitution_tau{a.tau:g}{tag}"
    (RESULTS_DIR / f"{stem}.json").write_text(json.dumps(out, indent=2))
    np.savez_compressed(RESULTS_DIR / f"{stem}.npz", **per_query)
    print(f"Saved: {RESULTS_DIR / (stem + '.json')}")


if __name__ == "__main__":
    main()
