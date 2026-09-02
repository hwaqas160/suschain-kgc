"""Does the certificate relation explain the gap against Peeris et al.?

Our LODHalal RGCN reports MRR ~0.50 where the base paper reports 0.057. This
script tests the most likely mechanism directly: RGCN trained on the identical
split, with and without the has_cert relation present in the graph.

If removing certificates collapses MRR toward the base paper's value, then the
gap is a graph-construction difference (which relations were included), not a
modelling advantage, and the paper must say so.

Run: python -u scripts/23_cert_relation_ablation.py
"""
import json
import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.training.train_linkpred import TrainConfig, train_link_prediction

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
ET = ("food", "has_manufacturer", "manufacturer")
REV = ("manufacturer", "rev_has_manufacturer", "food")
SEEDS = [0, 1, 2, 3, 4, 5, 6]

DROP = [("food", "has_cert", "cert"), ("cert", "rev_has_cert", "food")]


def strip_cert(data):
    """Return a copy of the graph with the certificate relation removed."""
    out = data.clone()
    for et in DROP:
        if et in out.edge_types:
            del out[et]
    if "cert" in out.node_types:
        del out["cert"]
    return out


def run(data, label):
    mrrs, hits = [], []
    for seed in SEEDS:
        cfg = TrainConfig(conv_kind="rgcn", epochs=100, patience=15, seed=seed)
        r = train_link_prediction(data, ET, REV, cfg)
        mrrs.append(r.test_ranking["mrr"])
        hits.append(r.test_ranking["hits@10"])
        print(f"  {label} seed {seed}: MRR={mrrs[-1]:.4f} Hits@10={hits[-1]:.4f}", flush=True)
    return {
        "mrr_mean": statistics.mean(mrrs), "mrr_std": statistics.stdev(mrrs),
        "hits@10_mean": statistics.mean(hits), "hits@10_std": statistics.stdev(hits),
        "mrr_per_seed": mrrs, "hits@10_per_seed": hits,
    }


def main():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    full = built.data
    nocert = strip_cert(full)

    print("relations WITH certificates:")
    for et in full.edge_types:
        print(f"   {et}")
    print("relations WITHOUT certificates:")
    for et in nocert.edge_types:
        print(f"   {et}")
    print()

    with_cert = run(full, "with-cert   ")
    print()
    without_cert = run(nocert, "without-cert")

    print("\n=== RGCN on LODHalal, certificate-relation ablation (7 seeds) ===")
    print(f"  with certificates    : MRR {with_cert['mrr_mean']:.4f} +/- {with_cert['mrr_std']:.4f}"
          f"   Hits@10 {with_cert['hits@10_mean']:.4f} +/- {with_cert['hits@10_std']:.4f}")
    print(f"  without certificates : MRR {without_cert['mrr_mean']:.4f} +/- {without_cert['mrr_std']:.4f}"
          f"   Hits@10 {without_cert['hits@10_mean']:.4f} +/- {without_cert['hits@10_std']:.4f}")
    print(f"  base paper reports   : MRR 0.0574              Hits@10 0.4315")

    out = {"seeds": SEEDS, "with_cert": with_cert, "without_cert": without_cert,
           "base_paper_reported": {"mrr": 0.0574, "hits@10": 0.4315}}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "phase1_cert_relation_ablation_lodhalal.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
