"""Phase 1 (C2): KG-embedding baselines (TransE, DistMult, ComplEx, RotatE)
via PyKEEN, using the same split as the GNN/heuristic evaluations. This is
the direct fix for plan limitation B3: "For a paper titled 'Knowledge Graph
Completion,' there is no TransE, DistMult, ComplEx, RotatE, or ConvE."

Run: python scripts/04_kge_baselines.py
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_link_prediction
from suschain.models.kge import train_kge_baseline

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
KGE_MODELS = ["TransE", "DistMult", "ComplEx", "RotatE"]
EPOCHS = 100


def run_anonnet_kge():
    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=True)
    edge_type = ("node", "supplies", "node")
    split = split_link_prediction(built.data, edge_type, rev_edge_type=None, seed=0)
    num_nodes = built.data["node"].num_nodes

    results = {}
    for name in KGE_MODELS:
        print(f"\n--- AnonNet / KGE:{name} ---")
        t0 = time.perf_counter()
        r = train_kge_baseline(split, edge_type, num_nodes, num_nodes, same_node_type=True, model_name=name, epochs=EPOCHS, seed=0)
        print(f"wall_clock={time.perf_counter() - t0:.1f}s")
        print(f"test_metrics={r.test_metrics}")
        print(f"ranking_metrics={r.ranking_metrics}")
        results[name] = r
    return results


def run_lodhalal_kge():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    edge_type = ("food", "has_manufacturer", "manufacturer")
    rev_edge_type = ("manufacturer", "rev_has_manufacturer", "food")
    split = split_link_prediction(built.data, edge_type, rev_edge_type, seed=0)
    num_food = built.data["food"].num_nodes
    num_man = built.data["manufacturer"].num_nodes

    results = {}
    for name in KGE_MODELS:
        print(f"\n--- LODHalal / KGE:{name} ---")
        t0 = time.perf_counter()
        r = train_kge_baseline(split, edge_type, num_food, num_man, same_node_type=False, model_name=name, epochs=EPOCHS, seed=0)
        print(f"wall_clock={time.perf_counter() - t0:.1f}s")
        print(f"test_metrics={r.test_metrics}")
        print(f"ranking_metrics={r.ranking_metrics}")
        results[name] = r
    return results


def save(name: str, results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {k: asdict(v) for k, v in results.items()}
    path = RESULTS_DIR / f"phase1_kge_{name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {path}")


def main():
    anon_results = run_anonnet_kge()
    save("anonnet", anon_results)
    lod_results = run_lodhalal_kge()
    save("lodhalal", lod_results)


if __name__ == "__main__":
    main()
