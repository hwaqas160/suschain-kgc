"""Phase 1: decoder ablation, specifically testing whether GAT's ranking
anomaly (docs/PHASE1_BASELINES.md -- strong classification metrics, weak
ranking metrics, on BOTH datasets) is a decoder-fit issue. Compares
dot-product (baseline), bilinear, and MLP decoders for GAT on LODHalal and
AnonNet, holding the encoder fixed.

Run: python scripts/05_decoder_ablation.py
"""
import json
from dataclasses import asdict
from pathlib import Path

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.training.train_linkpred import TrainConfig, train_link_prediction

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
DECODERS = ["dot", "bilinear", "mlp"]


def run_lodhalal():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    edge_type = ("food", "has_manufacturer", "manufacturer")
    rev_edge_type = ("manufacturer", "rev_has_manufacturer", "food")

    results = {}
    for decoder_kind in DECODERS:
        print(f"\n--- LODHalal / GAT / decoder:{decoder_kind} ---")
        cfg = TrainConfig(conv_kind="gat", decoder_kind=decoder_kind, epochs=100, patience=15, seed=0)
        result = train_link_prediction(built.data, edge_type, rev_edge_type, cfg)
        results[decoder_kind] = result
        print(f"epochs_run={result.epochs_run} wall_clock={result.wall_clock_seconds:.1f}s")
        print(f"test_metrics={result.test_metrics}")
        print(f"test_ranking={result.test_ranking}")
    return results


def run_anonnet():
    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=True)
    edge_type = ("node", "supplies", "node")

    results = {}
    for decoder_kind in DECODERS:
        print(f"\n--- AnonNet / GAT / decoder:{decoder_kind} ---")
        cfg = TrainConfig(conv_kind="gat", decoder_kind=decoder_kind, epochs=100, patience=15, seed=0)
        result = train_link_prediction(built.data, edge_type, None, cfg)
        results[decoder_kind] = result
        print(f"epochs_run={result.epochs_run} wall_clock={result.wall_clock_seconds:.1f}s")
        print(f"test_metrics={result.test_metrics}")
        print(f"test_ranking={result.test_ranking}")
    return results


def save(name: str, results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, r in results.items():
        d = asdict(r)
        d["config"]["device"] = str(d["config"]["device"])
        out[k] = d
    path = RESULTS_DIR / f"phase1_decoder_ablation_{name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {path}")


def main():
    lod_results = run_lodhalal()
    save("lodhalal_gat", lod_results)
    anon_results = run_anonnet()
    save("anonnet_gat", anon_results)


if __name__ == "__main__":
    main()
