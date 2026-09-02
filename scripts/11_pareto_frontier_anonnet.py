"""Phase 3 (C3) follow-up: Pareto frontier on AnonNet instead of LODHalal.

LODHalal's sweep (scripts/10) produced a compressed, uninformative
mean_risk_at_10 range (0.0003-0.0011) across ALL lambda values -- because
96.2% of manufacturers are already at risk=0 (see PHASE2_SUSTAINABILITY.md),
and RGCN's structural bias toward popular/well-established manufacturers
already correlates with compliance, leaving little headroom for the
sustainability loss to demonstrate a tradeoff. AnonNet's composite
carbon-intensity risk score has genuine continuous spread (roughly
balanced tertile bands), making it a better substrate for actually
observing the mechanism at work. Uses GAT (AnonNet's strongest tested
link predictor among the GNNs -- see PHASE1_FINAL_BENCHMARK.md).

Run: python scripts/11_pareto_frontier_anonnet.py
"""
import json
from dataclasses import asdict
from pathlib import Path

import torch

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata
from suschain.enrichment.anonnet_sustainability import compute_composite_score
from suschain.enrichment.product_taxonomy import classify_products_column
from suschain.training.train_linkpred import TrainConfig
from suschain.training.train_linkpred_multiobjective import train_link_prediction_multiobjective

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]


def build_risk_tensor(g, built):
    categories = classify_products_column(g.nodes["products"])
    scores_df = compute_composite_score(g.nodes["location"], categories)
    node_map = built.id_maps["node"]
    num_nodes = built.data["node"].num_nodes
    risk = torch.full((num_nodes,), float("nan"))
    local_idx = g.nodes["id"].map(node_map).to_numpy()
    risk[local_idx] = torch.tensor(scores_df["composite_score"].to_numpy(), dtype=torch.float32)
    return risk


def main():
    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=True)
    edge_type = ("node", "supplies", "node")
    risk = build_risk_tensor(g, built)
    print(f"risk coverage: {(~torch.isnan(risk)).sum().item()}/{len(risk)}")

    results = []
    for lam in LAMBDAS:
        print(f"\n--- lambda={lam} ---")
        cfg = TrainConfig(conv_kind="gat", epochs=100, patience=15, seed=0)
        result = train_link_prediction_multiobjective(
            built.data, edge_type, None, cfg, risk, sustainability_lambda=lam
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
    path = RESULTS_DIR / "phase3_pareto_frontier_anonnet_gat.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {path}")

    print("\nSummary (lambda, MRR, Hits@10, mean_risk_at_10):")
    for r in results:
        print(f"  {r.sustainability_lambda:>4} {r.test_ranking['mrr']:.4f} "
              f"{r.test_ranking['hits@10']:.4f} {r.mean_risk_at_10:.4f}")


if __name__ == "__main__":
    main()
