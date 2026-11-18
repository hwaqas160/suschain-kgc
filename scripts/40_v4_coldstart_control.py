"""v4 work package 8: where does the embedding-table model's cold-start
signal come from?

A held-out food never receives a gradient, so its embedding row is random
initialisation, and with every edge removed it has no neighbours. Yet the
manuscript reports its cold-start AUC (0.8745) above the popularity control
(0.8635). Candidate explanation: the encoder's bias terms turn any input into
roughly the same vector, so scores reduce to a learned destination prior that
ranks manufacturers a little better than raw training degree, with no
information about the query food.

Controls, on the same held-out edges and negatives:
  mean_query_vector  every held-out food gets the average of their vectors
  inputs_zeroed      held-out embedding rows set to zero before encoding
  inputs_redrawn     held-out embedding rows re-drawn from N(0, 1)
If these match the original AUC, the margin over popularity is a learned
prior, not query information.

Output: results/metrics/v4_coldstart_control.json
Run:    python -u scripts/40_v4_coldstart_control.py
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.inductive_splitters import inductive_entity_holdout
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.eval.significance import exact_wilcoxon
from suschain.training.train_inductive import InductiveTrainConfig, train_and_eval_inductive

_spec = importlib.util.spec_from_file_location("s29", ROOT / "scripts" / "29_inductive_multiseed.py")
_s29 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_s29)
EDGE_TYPE, popularity_coldstart_auc = _s29.EDGE_TYPE, _s29.popularity_coldstart_auc

SEEDS = list(range(10))
KEYS = ["original", "mean_query_vector", "inputs_zeroed", "inputs_redrawn", "popularity"]


def main():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    num_dst = built.data["manufacturer"].num_nodes
    nnd = {nt: built.data[nt].num_nodes for nt in built.data.node_types}
    per_seed = {k: [] for k in KEYS}
    for seed in SEEDS:
        split = inductive_entity_holdout(built.data, EDGE_TYPE, holdout_node_type="food",
                                         holdout_fraction=0.1, seed=seed)
        cfg = InductiveTrainConfig(conv_kind="sage", epochs=100, patience=15, seed=seed)
        r = train_and_eval_inductive(split, EDGE_TYPE, built.data.metadata(), nnd, cfg, None,
                                     "embedding", coldstart_controls=True)
        per_seed["original"].append(r.coldstart_metrics["auc_roc"])
        for k in ("mean_query_vector", "inputs_zeroed", "inputs_redrawn"):
            per_seed[k].append(r.coldstart_controls[k])
        per_seed["popularity"].append(popularity_coldstart_auc(split, seed, num_dst)["auc_roc"])
        print("seed", seed, {k: round(v[-1], 4) for k, v in per_seed.items()}, flush=True)

    summary = {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))} for k, v in per_seed.items()}
    tests = {f"{k}_vs_original": exact_wilcoxon(per_seed[k], per_seed["original"])
             for k in ("mean_query_vector", "inputs_zeroed", "inputs_redrawn", "popularity")}
    out = {"seeds": SEEDS, "per_seed": per_seed, "summary": summary, "wilcoxon": tests}
    p = ROOT / "results" / "metrics" / "v4_coldstart_control.json"
    p.write_text(json.dumps(out, indent=2))
    for k, s in summary.items():
        print(f"{k:18} {s['mean']:.4f} +/- {s['std']:.4f}")
    print(tests)
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
