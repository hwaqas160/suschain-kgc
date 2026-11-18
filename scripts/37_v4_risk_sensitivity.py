"""v4 work package 4: how much do the AnonNet substitution conclusions depend
on how the risk proxy is built?

Risk variants
  baseline        country CO2 per capita x hand-set sector multiplier; firms
                  without a location ranked at the mean (the policy used elsewhere)
  missing_max     firms without a location treated as the riskiest
  missing_min     firms without a location treated as the safest
  country_only    sector multiplier removed
  sector_only     country term removed (every firm gets its sector multiplier)
  perturb_{i}     every sector multiplier scaled by an independent U(0.5, 1.5)
                  factor (20 draws)

For every variant and seed, with one plain GraphSAGE model per seed (risk does
not enter its training):
  - risk@10 of the fixed rankings (random, popularity, product_sim, co_supply,
    plain GNN, fused GNN), and whether their order by risk changes;
  - the inference-time shift at beta in {1, 3}, and the pool-20 filter: hit@10
    lost and risk@10 removed relative to the plain model.
The learned penalty (lambda = 1) is retrained only for the five structural
variants, not for the 20 perturbations.

Output: results/metrics/v4_risk_sensitivity_anonnet.json
Run:    python -u scripts/37_v4_risk_sensitivity.py [--device cpu]
"""
import argparse
import importlib.util
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata
from suschain.decision_eval.scenario_trainer import train_scenario_model
from suschain.decision_eval.substitutes import (build_substitution_scenario, evaluate_substitution,
                                                idf_product_matrix, product_items, rank_candidates)
from suschain.enrichment.anonnet_sustainability import (CATEGORY_INTENSITY_MULTIPLIER,
                                                        compute_country_component)
from suschain.enrichment.product_taxonomy import classify_products_column
from suschain.training.multi_objective_loss import normalize_risk

# reuse the fusion helper from script 36 rather than copying it
_spec = importlib.util.spec_from_file_location("s36", ROOT / "scripts" / "36_v4_anonnet_substitution.py")
_s36 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s36)
rrf_order, filtered_order = _s36.rrf_order, _s36.filtered_order

ET = ("node", "supplies", "node")
K = 10
FIXED = ["random", "popularity", "product_sim", "co_supply", "gnn_plain", "fused_plain"]
STRUCTURAL = ["baseline", "missing_max", "missing_min", "country_only", "sector_only"]
N_PERTURB = 20


def build_variants(country, cats, rng):
    """country: normalised country term (NaN where missing); cats: sector per firm."""
    mult = cats.map(CATEGORY_INTENSITY_MULTIPLIER).fillna(CATEGORY_INTENSITY_MULTIPLIER["other"]).to_numpy()
    c = country.to_numpy(dtype=float)
    def norm(raw, fill):
        t = torch.tensor(raw, dtype=torch.float32)
        valid = ~torch.isnan(t)
        lo, hi = t[valid].min(), t[valid].max()
        t = (t - lo) / (hi - lo)
        fillv = {"mean": float(t[valid].mean()), "max": 1.0, "min": 0.0}[fill]
        return torch.nan_to_num(t, nan=fillv).numpy()
    v = {
        "baseline": norm(c * mult, "mean"),
        "missing_max": norm(c * mult, "max"),
        "missing_min": norm(c * mult, "min"),
        "country_only": norm(c, "mean"),
        "sector_only": norm(mult.astype(float), "mean"),
    }
    cat_names = sorted(CATEGORY_INTENSITY_MULTIPLIER)
    for i in range(N_PERTURB):
        f = dict(zip(cat_names, rng.uniform(0.5, 1.5, len(cat_names))))
        pm = np.array([CATEGORY_INTENSITY_MULTIPLIER.get(x, 1.0) * f.get(x, f["other"]) for x in cats])
        v[f"perturb_{i}"] = norm(c * pm, "mean")
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in a.seeds.split(",")]

    g = load_anonnet()
    b = build_anonnet_heterodata(g, make_undirected=False)
    data, nm = b.data, b.id_maps["node"]
    n = data["node"].num_nodes
    nodes = g.nodes.set_index("id").loc[sorted(nm, key=nm.get)]
    X = idf_product_matrix(nodes["products_list"].apply(product_items).tolist())
    E = data[ET].edge_index.numpy()
    cats = classify_products_column(nodes["products"].reset_index(drop=True))
    country = compute_country_component(nodes["location"].reset_index(drop=True))
    variants = build_variants(country, cats, np.random.default_rng(12345))

    res = defaultdict(lambda: defaultdict(list))   # variant -> metric -> [per seed]
    for seed in seeds:
        sc = build_substitution_scenario(E, X, tau=a.tau, seed=seed)
        qs = sc.queries
        rng = np.random.default_rng(seed)
        tb = rng.random(n)
        tr = sc.train_edges
        A = sp.csr_matrix((np.ones(tr.shape[1]), (tr[0], tr[1])), shape=(n, n))
        A.data[:] = 1.0
        pop = np.asarray(A.sum(axis=1)).ravel()
        co = (A @ A.T).tocsr()
        sup = torch.from_numpy(tr[[1, 0]].copy())
        mp = torch.from_numpy(np.concatenate([tr, tr[[1, 0]]], axis=1).copy())
        custs = sorted({q.customer for q in qs})
        cidx = {c: i for i, c in enumerate(custs)}
        cidx_t = torch.tensor(custs)

        def gnn_scores(risk_np, lam):
            rn = torch.tensor(risk_np, dtype=torch.float32, device=a.device)
            zc, zs = train_scenario_model(data, ET, None, sup, rn, lam, seed, device=a.device,
                                          conv_kind="sage", mp_edge_index=mp)
            return (zc[cidx_t.to(zc.device)] @ zs.T).cpu().numpy(), zs.cpu().numpy()

        S0, Z0 = gnn_scores(variants["baseline"], 0.0)       # risk does not affect lambda = 0
        r_co = [rank_candidates(co[q.disrupted].toarray().ravel(), q, pop + tb) for q in qs]
        fixed = {
            "random": [rank_candidates(rng.random(n), q) for q in qs],
            "popularity": [rank_candidates(pop, q, tb) for q in qs],
            "product_sim": [rank_candidates((X @ X[q.disrupted].T).toarray().ravel(), q, tb) for q in qs],
            "co_supply": r_co,
            "gnn_plain": [rank_candidates(S0[cidx[q.customer]], q, tb) for q in qs],
        }
        fixed["fused_plain"] = [rrf_order([x, rank_candidates(Z0 @ Z0[q.disrupted], q, tb), y], q)
                                for x, y, q in zip(fixed["gnn_plain"], r_co, qs)]
        plain_hit = evaluate_substitution(fixed["gnn_plain"], qs, variants["baseline"], K)["hit"].mean()

        for vname, risk in variants.items():
            rs = {m: evaluate_substitution(r, qs, risk, K) for m, r in fixed.items()}
            for m in FIXED:
                res[vname][f"risk@10/{m}"].append(float(rs[m]["risk"].mean()))
            base_risk = rs["gnn_plain"]["risk"].mean()
            for B in (1.0, 3.0):
                o = evaluate_substitution([rank_candidates(S0[cidx[q.customer]] - B * risk, q, tb) for q in qs],
                                          qs, risk, K)
                res[vname][f"shift_beta{B:g}/hit_lost"].append(float(plain_hit - o["hit"].mean()))
                res[vname][f"shift_beta{B:g}/risk_removed_pct"].append(float(100 * (1 - o["risk"].mean() / base_risk)))
            o = evaluate_substitution([filtered_order(S0[cidx[q.customer]], q, risk, 20, tb) for q in qs], qs, risk, K)
            res[vname]["filter20/hit_lost"].append(float(plain_hit - o["hit"].mean()))
            res[vname]["filter20/risk_removed_pct"].append(float(100 * (1 - o["risk"].mean() / base_risk)))
            if vname in STRUCTURAL:
                S1, _ = gnn_scores(risk, 1.0)
                o = evaluate_substitution([rank_candidates(S1[cidx[q.customer]], q, tb) for q in qs], qs, risk, K)
                res[vname]["learned_lam1/hit_lost"].append(float(plain_hit - o["hit"].mean()))
                res[vname]["learned_lam1/risk_removed_pct"].append(float(100 * (1 - o["risk"].mean() / base_risk)))
        print(f"seed {seed} done: baseline shift_beta1 risk removed "
              f"{res['baseline']['shift_beta1/risk_removed_pct'][-1]:.1f}% for "
              f"{res['baseline']['shift_beta1/hit_lost'][-1]:.4f} hit; learned lam1 "
              f"{res['baseline']['learned_lam1/risk_removed_pct'][-1]:.1f}% for "
              f"{res['baseline']['learned_lam1/hit_lost'][-1]:.4f} hit", flush=True)

    summary = {v: {m: {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0}
                   for m, x in d.items()} for v, d in res.items()}
    base_order = [summary["baseline"][f"risk@10/{m}"]["mean"] for m in FIXED]
    stability = {v: float(kendalltau(base_order, [summary[v][f"risk@10/{m}"]["mean"] for m in FIXED]).statistic)
                 for v in summary}
    out = {"seeds": seeds, "tau": a.tau, "k": K, "variants": list(variants),
           "kendall_tau_of_method_order_by_risk_vs_baseline": stability,
           "summary": summary, "per_seed": {v: dict(d) for v, d in res.items()}}
    p = ROOT / "results" / "metrics" / "v4_risk_sensitivity_anonnet.json"
    p.write_text(json.dumps(out, indent=2))
    print("Kendall tau vs baseline (min over variants):", min(stability.values()))
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
