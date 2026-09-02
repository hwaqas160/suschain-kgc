"""Phase 3 (C3): sweeps the sustainability-loss weight lambda and produces
the Pareto frontier (link-prediction quality vs. mean risk of top-10
recommended candidates) -- the paper's proposed centerpiece figure,
directly answering the base paper's own stated future work on
constraint-aware/multi-objective training (plan §6, C3).

lambda=0 reproduces the Phase-1 unconstrained baseline (the control).
Run on LODHalal with RGCN (this project's strongest LODHalal link
predictor, so any sustainability/utility tradeoff is visible against a
meaningful baseline, not noise).

Run: python scripts/10_pareto_frontier.py
"""
import json
from dataclasses import asdict
from pathlib import Path

import torch

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.enrichment.lodhalal_certification import compute_manufacturer_cert_coverage
from suschain.training.train_linkpred import TrainConfig
from suschain.training.train_linkpred_multiobjective import train_link_prediction_multiobjective

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]


def build_risk_tensor(g, built):
    cov_df = compute_manufacturer_cert_coverage(g)
    man_map = built.id_maps["manufacturer"]
    num_man = built.data["manufacturer"].num_nodes
    risk = torch.full((num_man,), float("nan"))
    local_idx = cov_df["manufacturer"].map(man_map).to_numpy()
    risk[local_idx] = torch.tensor((1.0 - cov_df["cert_coverage"]).to_numpy(), dtype=torch.float32)
    return risk


def main():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    edge_type = ("food", "has_manufacturer", "manufacturer")
    rev_edge_type = ("manufacturer", "rev_has_manufacturer", "food")
    risk = build_risk_tensor(g, built)

    results = []
    for lam in LAMBDAS:
        print(f"\n--- lambda={lam} ---")
        cfg = TrainConfig(conv_kind="rgcn", epochs=100, patience=15, seed=0)
        result = train_link_prediction_multiobjective(
            built.data, edge_type, rev_edge_type, cfg, risk, sustainability_lambda=lam
        )
        print(f"mrr={result.test_ranking['mrr']:.4f} hits@10={result.test_ranking['hits@10']:.4f} "
              f"mean_risk_at_10={result.mean_risk_at_10:.4f} wall_clock={result.wall_clock_seconds:.1f}s")
        results.append(result)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for r in results:
        d = asdict(r)
        d["config"]["device"] = str(d["config"]["device"])
        out.append(d)
    path = RESULTS_DIR / "phase3_pareto_frontier_lodhalal_rgcn.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {path}")

    print("\nSummary (lambda, MRR, Hits@10, mean_risk_at_10):")
    for r in results:
        print(f"  {r.sustainability_lambda:>4} {r.test_ranking['mrr']:.4f} "
              f"{r.test_ranking['hits@10']:.4f} {r.mean_risk_at_10:.4f}")


if __name__ == "__main__":
    main()
