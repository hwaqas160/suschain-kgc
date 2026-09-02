"""Phase 2: real (non-synthetic) node classification on both datasets --
AnonNet's country x category carbon-intensity bands, LODHalal's
certification-coverage bands -- with the full metric suite the base paper
never reported for its own (synthetic, single-dataset, single-model)
node classification task (plan B10): macro-F1, per-class F1, confusion
matrix, majority-class baseline, across all applicable architectures.

Run: python scripts/09_node_classification.py
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_node_classification
from suschain.enrichment.anonnet_sustainability import band_scores, compute_composite_score
from suschain.enrichment.lodhalal_certification import band_cert_coverage_exact, compute_manufacturer_cert_coverage
from suschain.enrichment.product_taxonomy import classify_products_column
from suschain.training.train_nodeclf import NodeClfConfig, train_node_classification

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"


def run_anonnet():
    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=True)

    categories = classify_products_column(g.nodes["products"])
    scores_df = compute_composite_score(g.nodes["location"], categories)
    bands = band_scores(scores_df["composite_score"])

    valid_mask = bands.notna()
    node_map = built.id_maps["node"]  # raw AnonNet id ("n0" etc) -> local idx
    raw_ids = g.nodes.loc[valid_mask, "id"]
    local_ids = raw_ids.map(node_map).to_numpy()
    string_labels = bands[valid_mask].tolist()

    split = split_node_classification(local_ids, string_labels, num_nodes_total=built.data["node"].num_nodes, seed=0)
    print(f"AnonNet node classification: {len(local_ids)} labeled nodes, classes={split.class_names}")

    results = {}
    for conv_kind in ["sage", "gat"]:  # rgcn skipped: single edge type
        cfg = NodeClfConfig(conv_kind=conv_kind, epochs=150, patience=20, seed=0)
        result = train_node_classification(built.data, "node", split, cfg)
        results[conv_kind] = result
        print(f"  {conv_kind}: majority_baseline={result.majority_baseline:.4f} "
              f"test_accuracy={result.test_metrics['accuracy']:.4f} "
              f"test_macro_f1={result.test_metrics['macro_f1']:.4f}")
        print(f"    confusion_matrix={result.test_metrics['confusion_matrix']}")
    return results, split.class_names


def run_lodhalal():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)

    cov_df = compute_manufacturer_cert_coverage(g)
    bands = band_cert_coverage_exact(cov_df["cert_coverage"])

    man_map = built.id_maps["manufacturer"]
    local_ids = cov_df["manufacturer"].map(man_map).to_numpy()
    string_labels = bands.tolist()

    split = split_node_classification(local_ids, string_labels, num_nodes_total=built.data["manufacturer"].num_nodes, seed=0)
    print(f"LODHalal node classification: {len(local_ids)} labeled nodes, classes={split.class_names}")

    results = {}
    for conv_kind in ["sage", "gat", "rgcn"]:
        cfg = NodeClfConfig(conv_kind=conv_kind, epochs=150, patience=20, seed=0)
        result = train_node_classification(built.data, "manufacturer", split, cfg)
        results[conv_kind] = result
        print(f"  {conv_kind}: majority_baseline={result.majority_baseline:.4f} "
              f"test_accuracy={result.test_metrics['accuracy']:.4f} "
              f"test_macro_f1={result.test_metrics['macro_f1']:.4f}")
        print(f"    confusion_matrix={result.test_metrics['confusion_matrix']}")
    return results, split.class_names


def save(name: str, results: dict, class_names: list[str]):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"class_names": class_names, "results": {}}
    for kind, r in results.items():
        d = asdict(r)
        d["config"]["device"] = str(d["config"]["device"])
        out["results"][kind] = d
    path = RESULTS_DIR / f"phase2_nodeclf_{name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {path}")


def main():
    anon_results, anon_classes = run_anonnet()
    save("anonnet", anon_results, anon_classes)
    lod_results, lod_classes = run_lodhalal()
    save("lodhalal", lod_results, lod_classes)


if __name__ == "__main__":
    main()
