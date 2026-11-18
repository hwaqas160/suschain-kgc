"""v4 work package 9: replication of Peeris et al. (2026) in their own setting.

Setting, from their paper and their first author's answers (8 October 2026):
  - two negatives per positive (training, validation and test);
  - reported metrics computed on the validation split;
  - LODHalal: food and manufacturer files only, i.e. the single relation
    food -has_manufacturer-> manufacturer (relations preset
    "food_manufacturer_only");
  - AnonNet: the 3,813-node product-location graph (unique location values
    and unique product strings). The edges are not documented; both readings
    are run (graph_builder.build_anonnet_product_location).
Their hyperparameters are not reported, so the fixed default configuration
(TrainConfig) is used for every encoder.

Every metric is reported on validation and on test, accuracy and F1 at the
fixed threshold 0.5 and at the F1-optimal validation threshold, and ranking
over the full catalogue and over 99 sampled negatives.

Output: results/metrics/v4_peeris_replication.json
Run:    python -u scripts/44_v4_peeris_replication.py [--seeds 0-9]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_product_location, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_link_prediction
from suschain.training.replication import train_replication
from suschain.training.train_linkpred import TrainConfig

# Their published numbers (Peeris et al., 2026), as tabulated in the v4 manuscript.
REPORTED = {
    "lodhalal": {
        "rgcn": {"accuracy": 0.7124, "auc_roc": 0.9186, "f1": 0.1987, "mrr": 0.0574, "hits@10": 0.4315},
        "gat": {"accuracy": 0.6887, "auc_roc": 0.8958, "f1": 0.1618, "mrr": 0.0455, "hits@10": 0.3884},
        "sage": {"accuracy": 0.6658, "auc_roc": 0.8789, "f1": 0.1379, "mrr": 0.0419, "hits@10": 0.3462},
    },
    "anonnet": {
        "gat": {"accuracy": 0.8610, "auc_roc": 0.9112, "f1": 0.7254, "mrr": 0.0477, "hits@10": 0.4200},
        "sage": {"accuracy": 0.6740, "auc_roc": 0.8587, "f1": 0.4349, "mrr": 0.0436, "hits@10": 0.4000},
    },
}


def datasets():
    lod = build_lodhalal_heterodata(load_lodhalal(), make_undirected=True, relations="food_manufacturer_only").data
    yield ("lodhalal_food_manufacturer", lod, ("food", "has_manufacturer", "manufacturer"),
           ("manufacturer", "rev_has_manufacturer", "food"), ["sage", "gat", "rgcn"], False)
    g = load_anonnet()
    for v in ("firm", "supply"):
        d = build_anonnet_product_location(g, v).data
        yield (f"anonnet_product_location_{v}", d, ("product", "at", "location"),
               ("location", "rev_at", "product"), ["sage", "gat", "rgcn"], True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in a.seeds.split(",")]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {"seeds": seeds, "neg_sampling_ratio": 2.0, "reported": REPORTED, "results": {}}
    for name, data, et, rv, convs, both in datasets():
        res = {}
        for seed in seeds:
            split = split_link_prediction(data, et, rv, seed=seed, neg_sampling_ratio=2.0)
            for conv in convs:
                cfg = TrainConfig(conv_kind=conv, epochs=100, patience=15, seed=seed, device=dev)
                r = train_replication(data, et, rv, split, cfg, both_directions=both)
                for part in ("val", "test"):
                    for k, v in r[part].items():
                        res.setdefault(conv, {}).setdefault(part, {}).setdefault(k, []).append(float(v))
                v = r["val"]
                print(f"{name} seed {seed} {conv:5} val acc0.5 {v['accuracy_0.5']:.4f} f1_0.5 {v['f1_0.5']:.4f} "
                      f"AUC {v['auc_roc']:.4f} MRR full {v['tail_full_mrr']:.4f} sampled {v['tail_sampled99_mrr']:.4f} "
                      f"H@10 full {v['tail_full_hits@10']:.4f} sampled {v['tail_sampled99_hits@10']:.4f}", flush=True)
        out["results"][name] = {
            conv: {part: {k: {"mean": float(np.mean(x)), "std": float(np.std(x, ddof=1)) if len(x) > 1 else 0.0,
                              "values": x} for k, x in d.items()} for part, d in pd.items()}
            for conv, pd in res.items()}
    p = ROOT / "results" / "metrics" / f"v4_peeris_replication{a.tag}.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
