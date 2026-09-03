"""GATv2-vs-GATv1 sensitivity experiment.

This paper uses GATv2 where the study it extends uses the original GAT.
That substitution was previously carried as a stated limitation: we could
not say how much of GAT's weak link-prediction ranking, or its strong
node-classification performance, was attributable to the attention variant
rather than to the attention-based architecture family more broadly.

This script settles that by training the original GAT (GATConv) under
hyperparameters identical to the GATv2 runs -- same depth, width, heads,
dropout, epochs, patience and the same 7 seeds -- so the only difference
between the two arms is the attention mechanism itself. Both link
prediction (both datasets) and node classification (both datasets) are
covered, since the GATv2 substitution affects every GAT number reported.

Run: python scripts/27_gatv1_sensitivity.py
"""
import json
from pathlib import Path

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.eval.significance import compare_models
from suschain.training.train_linkpred import TrainConfig, train_link_prediction

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
SEEDS = list(range(7))
METRICS = ["mrr", "hits@10", "auc_roc", "f1", "accuracy"]


def run_arm(data, edge_type, rev_edge_type, conv_kind: str, label: str) -> dict:
    per_seed = {m: [] for m in METRICS}
    for seed in SEEDS:
        cfg = TrainConfig(conv_kind=conv_kind, epochs=100, patience=15, seed=seed)
        r = train_link_prediction(data, edge_type, rev_edge_type, cfg)
        per_seed["mrr"].append(r.test_ranking["mrr"])
        per_seed["hits@10"].append(r.test_ranking["hits@10"])
        per_seed["auc_roc"].append(r.test_metrics["auc_roc"])
        per_seed["f1"].append(r.test_metrics["f1"])
        per_seed["accuracy"].append(r.test_metrics["accuracy"])
        print(f"  {label}/seed={seed}: mrr={r.test_ranking['mrr']:.4f} "
              f"hits@10={r.test_ranking['hits@10']:.4f}", flush=True)
    return per_seed


def compare(dataset: str, data, edge_type, rev_edge_type) -> dict:
    print(f"\n=== {dataset}: GATv2 vs GATv1 link prediction ===", flush=True)
    v2 = run_arm(data, edge_type, rev_edge_type, "gat", f"{dataset}/GATv2")
    v1 = run_arm(data, edge_type, rev_edge_type, "gatv1", f"{dataset}/GATv1")

    out = {"per_seed": {"gatv2": v2, "gatv1": v1}, "summary": {}, "wilcoxon": {}}
    for m in METRICS:
        cmp = compare_models({"gatv2": v2[m], "gatv1": v1[m]})
        out["summary"][m] = cmp["summary"]
        out["wilcoxon"][m] = cmp["pairwise"]
        s = cmp["summary"]
        print(f"  {m:9s} GATv2={s['gatv2']['mean']:.4f}+/-{s['gatv2']['std']:.4f}  "
              f"GATv1={s['gatv1']['mean']:.4f}+/-{s['gatv1']['std']:.4f}", flush=True)
    return out


def main():
    results = {}

    lod_g = load_lodhalal()
    lod = build_lodhalal_heterodata(lod_g, make_undirected=True)
    results["lodhalal"] = compare(
        "lodhalal", lod.data,
        ("food", "has_manufacturer", "manufacturer"),
        ("manufacturer", "rev_has_manufacturer", "food"),
    )

    anon_g = load_anonnet()
    anon = build_anonnet_heterodata(anon_g, make_undirected=True)
    results["anonnet"] = compare(
        "anonnet", anon.data, ("node", "supplies", "node"), None,
    )

    out = {
        "note": ("GATv1 (original GAT, GATConv) against GATv2 (GATv2Conv) under "
                 "identical hyperparameters and the same 7 seeds, so the arms "
                 "differ only in the attention mechanism. Tests whether this "
                 "paper's GATv2 substitution materially changes the GAT result "
                 "reported against the base paper's GAT."),
        "seeds": SEEDS,
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "phase1_gatv1_sensitivity.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
