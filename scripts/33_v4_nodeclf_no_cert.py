"""v4 work package 1: LODHalal node classification with and without the
certificate relation.

The class label (Fully / Partially / Not certified) is computed from
certificate links. With those links in the graph, a model can read the label
off its neighbourhood, so the earlier result (GAT macro-F1 0.62, 16 of 16
minority-class recall on one seed) may be label leakage. This script trains the
same three encoders on the "full" and "no_cert" graphs with identical label
splits and seeds, so the two are paired.

Output: results/metrics/v4_nodeclf_lodhalal_no_cert.json
Run:    python -u scripts/33_v4_nodeclf_no_cert.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_node_classification
from suschain.enrichment.lodhalal_certification import (band_cert_coverage_exact,
                                                        compute_manufacturer_cert_coverage)
from suschain.eval.significance import exact_wilcoxon, mean_std
from suschain.training.train_nodeclf import NodeClfConfig, train_node_classification

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
SEEDS = list(range(10))
CONVS = ["sage", "gat", "rgcn"]
GRAPHS = ["full", "no_cert"]
METRICS = ["accuracy", "macro_f1"]


def main():
    g = load_lodhalal()
    cov = compute_manufacturer_cert_coverage(g)
    labels = band_cert_coverage_exact(cov["cert_coverage"]).tolist()
    built = {gr: build_lodhalal_heterodata(g, make_undirected=True, relations=gr) for gr in GRAPHS}
    local_ids = cov["manufacturer"].map(built["full"].id_maps["manufacturer"]).to_numpy()
    assert (cov["manufacturer"].map(built["no_cert"].id_maps["manufacturer"]).to_numpy() == local_ids).all()
    n_man = built["full"].data["manufacturer"].num_nodes

    per_seed = {gr: {c: {m: [] for m in METRICS} for c in CONVS} for gr in GRAPHS}
    majority, class_names = None, None
    for seed in SEEDS:
        split = split_node_classification(local_ids, labels, num_nodes_total=n_man, seed=seed)
        class_names = split.class_names
        for gr in GRAPHS:
            for conv in CONVS:
                cfg = NodeClfConfig(conv_kind=conv, epochs=150, patience=20, seed=seed)
                r = train_node_classification(built[gr].data, "manufacturer", split, cfg)
                majority = r.majority_baseline
                for m in METRICS:
                    per_seed[gr][conv][m].append(r.test_metrics[m])
                print(f"seed {seed} {gr:8} {conv:5} acc={r.test_metrics['accuracy']:.4f} "
                      f"macro_f1={r.test_metrics['macro_f1']:.4f}", flush=True)

    summary = {gr: {c: {m: dict(zip(("mean", "std"), mean_std(v[m]))) for m in METRICS}
                    for c, v in d.items()} for gr, d in per_seed.items()}
    tests = {c: {m: exact_wilcoxon(per_seed["no_cert"][c][m], per_seed["full"][c][m]) for m in METRICS}
             for c in CONVS}
    out = {"seeds": SEEDS, "class_names": class_names, "majority_baseline_last_seed": majority,
           "per_seed": per_seed, "summary": summary,
           "wilcoxon_no_cert_vs_full": tests}
    p = RESULTS_DIR / "v4_nodeclf_lodhalal_no_cert.json"
    p.write_text(json.dumps(out, indent=2))
    for c in CONVS:
        print(f"{c:5} macro_f1 full={summary['full'][c]['macro_f1']['mean']:.4f} "
              f"no_cert={summary['no_cert'][c]['macro_f1']['mean']:.4f} p={tests[c]['macro_f1']:.4f}")
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
