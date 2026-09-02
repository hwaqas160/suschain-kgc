"""Multi-seed version of the lambda sweep (scripts/10 and 11).

The single-seed sweep was the last remaining methodological gap in this
paper: every other headline comparison runs at seven paired seeds with a
Wilcoxon test, but the multi-objective sweep did not, so the small effect
observed on LODHalal could not be separated from seed variance. This
script reruns both sweeps at the same seven seeds used throughout
(scripts/07) and reports mean +/- std per lambda, plus a paired Wilcoxon
test of lambda=0 against the largest lambda on both utility and risk.

Run: python scripts/18_pareto_multiseed.py
"""
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.enrichment.anonnet_sustainability import compute_composite_score
from suschain.enrichment.lodhalal_certification import compute_manufacturer_cert_coverage
from suschain.enrichment.product_taxonomy import classify_products_column
from suschain.training.train_linkpred import TrainConfig
from suschain.training.train_linkpred_multiobjective import train_link_prediction_multiobjective

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
SEEDS = [0, 1, 2, 3, 4, 5, 6]          # same seven seeds as scripts/07


def lodhalal_risk(g, built):
    cov_df = compute_manufacturer_cert_coverage(g)
    man_map = built.id_maps["manufacturer"]
    risk = torch.full((built.data["manufacturer"].num_nodes,), float("nan"))
    idx = cov_df["manufacturer"].map(man_map).to_numpy()
    risk[idx] = torch.tensor((1.0 - cov_df["cert_coverage"]).to_numpy(), dtype=torch.float32)
    return risk


def anonnet_risk(g, built):
    categories = classify_products_column(g.nodes["products"])
    scores_df = compute_composite_score(g.nodes["location"], categories)
    node_map = built.id_maps["node"]
    risk = torch.full((built.data["node"].num_nodes,), float("nan"))
    idx = g.nodes["id"].map(node_map).to_numpy()
    risk[idx] = torch.tensor(scores_df["composite_score"].to_numpy(), dtype=torch.float32)
    return risk


def wilcoxon(a, b):
    """Exact two-sided Wilcoxon signed-rank test on paired samples."""
    from itertools import product
    d = [x - y for x, y in zip(a, b) if x != y]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    w_plus = sum(r for r, dv in zip(ranks, d) if dv > 0)
    # exact null distribution over all sign assignments
    count = 0
    total = 0
    for signs in product([0, 1], repeat=n):
        s = sum(r for r, sg in zip(ranks, signs) if sg)
        total += 1
        if s >= max(w_plus, sum(ranks) - w_plus) or s <= min(w_plus, sum(ranks) - w_plus):
            count += 1
    return min(1.0, count / total)


def sweep(name, loader, builder, edge_type, rev_edge_type, conv_kind, risk_fn, outfile):
    print(f"\n{'=' * 70}\n{name}: {len(LAMBDAS)} lambdas x {len(SEEDS)} seeds\n{'=' * 70}")
    g = loader()
    built = builder(g, make_undirected=True)
    risk = risk_fn(g, built)

    per_lambda = {}
    for lam in LAMBDAS:
        mrrs, risks = [], []
        for seed in SEEDS:
            cfg = TrainConfig(conv_kind=conv_kind, epochs=100, patience=15, seed=seed)
            r = train_link_prediction_multiobjective(
                built.data, edge_type, rev_edge_type, cfg, risk, sustainability_lambda=lam
            )
            mrrs.append(r.test_ranking["mrr"])
            risks.append(r.mean_risk_at_10)
        per_lambda[lam] = {"mrr": mrrs, "mean_risk_at_10": risks}
        print(f"  lambda={lam:<4} MRR={statistics.mean(mrrs):.4f}+/-{statistics.stdev(mrrs):.4f}  "
              f"risk={statistics.mean(risks):.5f}+/-{statistics.stdev(risks):.5f}")

    lo, hi = LAMBDAS[0], LAMBDAS[-1]
    p_mrr = wilcoxon(per_lambda[lo]["mrr"], per_lambda[hi]["mrr"])
    p_risk = wilcoxon(per_lambda[lo]["mean_risk_at_10"], per_lambda[hi]["mean_risk_at_10"])
    print(f"  Wilcoxon lambda={lo} vs lambda={hi}:  p(MRR)={p_mrr:.4f}  p(risk)={p_risk:.4f}")

    out = {
        "seeds": SEEDS,
        "lambdas": LAMBDAS,
        "conv_kind": conv_kind,
        "per_lambda": {str(k): v for k, v in per_lambda.items()},
        "summary": {
            str(k): {
                "mrr_mean": statistics.mean(v["mrr"]),
                "mrr_std": statistics.stdev(v["mrr"]),
                "risk_mean": statistics.mean(v["mean_risk_at_10"]),
                "risk_std": statistics.stdev(v["mean_risk_at_10"]),
            } for k, v in per_lambda.items()
        },
        "wilcoxon_lambda0_vs_lambda_max": {"p_mrr": p_mrr, "p_risk": p_risk},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / outfile).write_text(json.dumps(out, indent=2))
    print(f"  saved {outfile}")
    return out


def main():
    sweep("LODHalal (RGCN)", load_lodhalal, build_lodhalal_heterodata,
          ("food", "has_manufacturer", "manufacturer"),
          ("manufacturer", "rev_has_manufacturer", "food"),
          "rgcn", lodhalal_risk, "phase3_pareto_multiseed_lodhalal_rgcn.json")

    sweep("AnonNet (GAT)", load_anonnet, build_anonnet_heterodata,
          ("node", "supplies", "node"), None,
          "gat", anonnet_risk, "phase3_pareto_multiseed_anonnet_gat.json")

    print("\nDone.")


if __name__ == "__main__":
    main()
