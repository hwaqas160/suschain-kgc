"""Phase 0 deliverable: attempt to reproduce the base paper's GraphSAGE/GAT/
RGCN link-prediction results on LODHalal and AnonNet. Per the plan, exact
reproduction is not expected (the paper's protocol is underspecified — no
disclosed negative-sampling ratio, split method, or filtered-vs-raw ranking
convention) — this run uses this project's own frozen, documented protocol
(see splitters.py / eval/ranking.py) and reports results honestly next to
the paper's numbers rather than silently forcing a match.

Run: python scripts/02_reproduce_baseline.py
"""
import json
from dataclasses import asdict
from pathlib import Path

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.training.train_linkpred import TrainConfig, train_link_prediction

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"

PAPER_LODHALAL = {
    "GraphSAGE": {"accuracy": 0.6658, "auc_roc": 0.8789, "f1": 0.1379, "mrr": 0.0419, "hits@10": 0.3462},
    "GAT": {"accuracy": 0.6887, "auc_roc": 0.8958, "f1": 0.1618, "mrr": 0.0455, "hits@10": 0.3884},
    "RGCN": {"accuracy": 0.7124, "auc_roc": 0.9186, "f1": 0.1987, "mrr": 0.0574, "hits@10": 0.4315},
}
PAPER_ANONNET = {
    "GraphSAGE": {"accuracy": 0.6740, "auc_roc": 0.8587, "f1": 0.4349, "mrr": 0.0436, "hits@10": 0.4000},
    "GAT": {"accuracy": 0.8610, "auc_roc": 0.9112, "f1": 0.7254, "mrr": 0.0477, "hits@10": 0.4200},
}


def run_lodhalal():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    edge_type = ("food", "has_manufacturer", "manufacturer")
    rev_edge_type = ("manufacturer", "rev_has_manufacturer", "food")

    results = {}
    for kind in ["sage", "gat", "rgcn"]:
        print(f"\n--- LODHalal / {kind} ---")
        cfg = TrainConfig(conv_kind=kind, epochs=100, patience=15, seed=0)
        result = train_link_prediction(built.data, edge_type, rev_edge_type, cfg)
        results[kind] = result
        print(f"epochs_run={result.epochs_run} wall_clock={result.wall_clock_seconds:.1f}s "
              f"peak_vram={result.peak_vram_mb:.0f}MB")
        print(f"test_metrics(tuned_thr={result.tuned_threshold:.3f})={result.test_metrics}")
        print(f"test_ranking={result.test_ranking}")
    return results


def run_anonnet():
    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=True)
    edge_type = ("node", "supplies", "node")
    # ToUndirected() on a same-node-type relation symmetrizes edge_index
    # in place rather than creating a separate "rev_supplies" type (verified:
    # data.metadata() shows only a single ('node','supplies','node') edge
    # type after ToUndirected) -- so there is no reverse relation to pass.
    rev_edge_type = None

    results = {}
    for kind in ["sage", "gat"]:  # RGCN skipped: single relation type, matches paper's own rationale
        print(f"\n--- AnonNet / {kind} ---")
        cfg = TrainConfig(conv_kind=kind, epochs=100, patience=15, seed=0)
        result = train_link_prediction(built.data, edge_type, rev_edge_type, cfg)
        results[kind] = result
        print(f"epochs_run={result.epochs_run} wall_clock={result.wall_clock_seconds:.1f}s "
              f"peak_vram={result.peak_vram_mb:.0f}MB")
        print(f"test_metrics(tuned_thr={result.tuned_threshold:.3f})={result.test_metrics}")
        print(f"test_ranking={result.test_ranking}")
    return results


def print_comparison(dataset_name: str, results: dict, paper: dict):
    print(f"\n{'=' * 90}\n{dataset_name}: ours (tuned threshold, filtered ranking) vs. paper (Table 4/5)\n{'=' * 90}")
    header = f"{'model':<10}{'acc(ours)':>12}{'acc(paper)':>12}{'auc(ours)':>12}{'auc(paper)':>12}{'f1(ours)':>10}{'f1(paper)':>10}{'mrr(ours)':>11}{'mrr(paper)':>11}{'h10(ours)':>11}{'h10(paper)':>11}"
    print(header)
    for kind_key, model_name in [("sage", "GraphSAGE"), ("gat", "GAT"), ("rgcn", "RGCN")]:
        if kind_key not in results or model_name not in paper:
            continue
        r = results[kind_key]
        p = paper[model_name]
        row = (
            f"{model_name:<10}"
            f"{r.test_metrics['accuracy']:>12.4f}{p['accuracy']:>12.4f}"
            f"{r.test_metrics['auc_roc']:>12.4f}{p['auc_roc']:>12.4f}"
            f"{r.test_metrics['f1']:>10.4f}{p['f1']:>10.4f}"
            f"{r.test_ranking['mrr']:>11.4f}{p['mrr']:>11.4f}"
            f"{r.test_ranking['hits@10']:>11.4f}{p['hits@10']:>11.4f}"
        )
        print(row)


def save_results(name: str, results: dict):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for kind, r in results.items():
        d = asdict(r)
        d["config"]["device"] = str(d["config"]["device"])
        out[kind] = d
    path = RESULTS_DIR / f"phase0_baseline_{name}.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"Saved: {path}")


def main():
    lod_results = run_lodhalal()
    anon_results = run_anonnet()
    print_comparison("LODHalal", lod_results, PAPER_LODHALAL)
    print_comparison("AnonNet", anon_results, PAPER_ANONNET)
    save_results("lodhalal", lod_results)
    save_results("anonnet", anon_results)


if __name__ == "__main__":
    main()
