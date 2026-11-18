"""Multi-seed version of node classification (scripts/09).

The single-run version (Table 10 in the manuscript) reports point
estimates with no variance, which is the exact failure mode the paper
argues against everywhere else. Here the label split AND model init both
vary per seed (matching the established pattern in
07_multiseed_final.py's split_link_prediction(..., seed=config.seed)),
so seven independent (split, model) draws are compared with mean +/- std
and exact pairwise Wilcoxon signed-rank tests, same as every other
multi-seed table in this project.

Run: python -u scripts/28_nodeclf_multiseed.py
"""
import json
from dataclasses import asdict
from pathlib import Path

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_node_classification
from suschain.enrichment.anonnet_sustainability import band_scores, compute_composite_score
from suschain.enrichment.lodhalal_certification import band_cert_coverage_exact, compute_manufacturer_cert_coverage
from suschain.enrichment.product_taxonomy import classify_products_column
from suschain.eval.significance import compare_models
from suschain.training.train_nodeclf import NodeClfConfig, train_node_classification

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
SEEDS = list(range(7))
METRICS = ["accuracy", "macro_f1"]


def run_anonnet():
    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=True)
    categories = classify_products_column(g.nodes["products"])
    scores_df = compute_composite_score(g.nodes["location"], categories)
    bands = band_scores(scores_df["composite_score"])
    valid_mask = bands.notna()
    node_map = built.id_maps["node"]
    raw_ids = g.nodes.loc[valid_mask, "id"]
    local_ids = raw_ids.map(node_map).to_numpy()
    string_labels = bands[valid_mask].tolist()

    per_seed = {k: {m: [] for m in METRICS} for k in ["sage", "gat"]}
    class_names = None
    for seed in SEEDS:
        split = split_node_classification(local_ids, string_labels,
                                           num_nodes_total=built.data["node"].num_nodes, seed=seed)
        class_names = split.class_names
        for conv_kind in ["sage", "gat"]:
            cfg = NodeClfConfig(conv_kind=conv_kind, epochs=150, patience=20, seed=seed)
            r = train_node_classification(built.data, "node", split, cfg)
            per_seed[conv_kind]["accuracy"].append(r.test_metrics["accuracy"])
            per_seed[conv_kind]["macro_f1"].append(r.test_metrics["macro_f1"])
            print(f"  anonnet/{conv_kind}/seed={seed}: acc={r.test_metrics['accuracy']:.4f} "
                  f"macro_f1={r.test_metrics['macro_f1']:.4f} majority={r.majority_baseline:.4f}", flush=True)
    return per_seed, class_names


def run_lodhalal():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    cov_df = compute_manufacturer_cert_coverage(g)
    bands = band_cert_coverage_exact(cov_df["cert_coverage"])
    man_map = built.id_maps["manufacturer"]
    local_ids = cov_df["manufacturer"].map(man_map).to_numpy()
    string_labels = bands.tolist()

    per_seed = {k: {m: [] for m in METRICS} for k in ["sage", "gat", "rgcn"]}
    class_names = None
    for seed in SEEDS:
        split = split_node_classification(local_ids, string_labels,
                                           num_nodes_total=built.data["manufacturer"].num_nodes, seed=seed)
        class_names = split.class_names
        for conv_kind in ["sage", "gat", "rgcn"]:
            cfg = NodeClfConfig(conv_kind=conv_kind, epochs=150, patience=20, seed=seed)
            r = train_node_classification(built.data, "manufacturer", split, cfg)
            per_seed[conv_kind]["accuracy"].append(r.test_metrics["accuracy"])
            per_seed[conv_kind]["macro_f1"].append(r.test_metrics["macro_f1"])
            print(f"  lodhalal/{conv_kind}/seed={seed}: acc={r.test_metrics['accuracy']:.4f} "
                  f"macro_f1={r.test_metrics['macro_f1']:.4f} majority={r.majority_baseline:.4f}", flush=True)
    return per_seed, class_names


def summarize_and_save(name: str, per_seed: dict, class_names: list[str]):
    out = {"class_names": class_names, "seeds": SEEDS, "per_seed": per_seed, "comparisons": {}}
    for metric in METRICS:
        model_values = {model: vals[metric] for model, vals in per_seed.items()}
        out["comparisons"][metric] = compare_models(model_values)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / f"phase2_nodeclf_multiseed_{name}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\n=== {name} summary (7 seeds) ===")
    for metric in METRICS:
        print(f"-- {metric} --")
        for model, s in out["comparisons"][metric]["summary"].items():
            print(f"  {model:<8} {s['mean']:.4f} +/- {s['std']:.4f}")
        for pair, res in out["comparisons"][metric]["pairwise"].items():
            sig = "SIG" if res["significant_at_0.05"] else "n.s."
            print(f"    {pair}: p={res['p_value']:.4f} {sig}")
    print(f"Saved: {p}")


def main():
    anon_per_seed, anon_classes = run_anonnet()
    summarize_and_save("anonnet", anon_per_seed, anon_classes)
    lod_per_seed, lod_classes = run_lodhalal()
    summarize_and_save("lodhalal", lod_per_seed, lod_classes)


if __name__ == "__main__":
    main()
