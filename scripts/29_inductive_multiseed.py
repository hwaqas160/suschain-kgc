"""Multi-seed version of the inductive/cold-start evaluation (scripts/13).

The single-run version (Table 12) reports embedding-table AUC 0.872,
text-feature AUC 0.863 and (from a separate one-off diagnostic,
docs/PHASE4_INDUCTIVE_RESULTS.md) a popularity-only control at 0.865, then
the manuscript calls the three "statistically indistinguishable" with no
test behind that claim. This runs all three at the same seven seeds --
varying both which entities are held out (inductive_entity_holdout(...,
seed=seed)) and model init per seed, matching the established multi-seed
pattern elsewhere in this project -- and adds a genuine paired Wilcoxon
test between them.

The popularity control is scored on the IDENTICAL cold-start positive/
negative edges the trained models see each seed (same seed_offset=300
negative-sampling formula as train_inductive.py), so the three-way
comparison is a fair, paired one, not just three numbers that happen to
be close.

Run: python -u scripts/29_inductive_multiseed.py
"""
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.inductive_splitters import inductive_entity_holdout
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.enrichment.text_features import encode_series
from suschain.eval.metrics import binary_link_metrics
from suschain.eval.significance import compare_models
from suschain.models.heuristics import popularity_scores
from suschain.training.train_inductive import InductiveTrainConfig, train_and_eval_inductive

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
SEEDS = list(range(7))
EDGE_TYPE = ("food", "has_manufacturer", "manufacturer")


def popularity_coldstart_auc(split, seed: int, num_dst: int) -> dict:
    """Scores the popularity-only control on the SAME held-out cold-start
    edges and the SAME negative sample the trained models use that seed
    (seed_offset=300, matching train_inductive.py exactly), so the
    comparison is paired rather than incidental."""
    heldout_pos = split.heldout_test_edges
    known_dst_by_src: dict[int, set[int]] = {}
    all_edges = split.train_data[EDGE_TYPE].edge_index
    for s, d in zip(all_edges[0].tolist(), all_edges[1].tolist()):
        known_dst_by_src.setdefault(s, set()).add(d)

    rng = np.random.default_rng(seed + 300)
    negs = []
    for s in heldout_pos[0].tolist():
        known = known_dst_by_src.get(s, set())
        while True:
            cand = int(rng.integers(0, num_dst))
            if cand not in known:
                negs.append(cand)
                break
    neg_dst = torch.tensor(negs, dtype=torch.long)

    pop = popularity_scores(split.train_data[EDGE_TYPE].edge_index, num_dst)
    pos_scores = pop[heldout_pos[1].numpy()]
    neg_scores = pop[neg_dst.numpy()]
    scores = np.concatenate([pos_scores, neg_scores])
    labels = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
    # normalize to [0,1] so binary_link_metrics' 0.5 threshold is meaningful
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    return binary_link_metrics(labels, scores, threshold=0.5)


def main():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    num_dst = built.data["manufacturer"].num_nodes

    print("Encoding food names with sentence-transformers (cached after first run)...", flush=True)
    food_text_emb = encode_series(g.food["nama"], cache_key="lodhalal_food_nama_v1")
    text_features_base = {"food": torch.from_numpy(food_text_emb)}

    per_seed = {"embedding": [], "text": [], "popularity": []}
    for seed in SEEDS:
        split = inductive_entity_holdout(built.data, EDGE_TYPE, holdout_node_type="food",
                                          holdout_fraction=0.1, seed=seed)
        num_nodes_dict = {nt: built.data[nt].num_nodes for nt in built.data.node_types}

        for mode, tf in [("embedding", None), ("text", text_features_base)]:
            cfg = InductiveTrainConfig(conv_kind="sage", epochs=100, patience=15, seed=seed)
            r = train_and_eval_inductive(split, EDGE_TYPE, built.data.metadata(), num_nodes_dict, cfg, tf, mode)
            auc = r.coldstart_metrics.get("auc_roc", float("nan"))
            per_seed[mode].append(auc)
            print(f"  seed={seed} mode={mode}: coldstart_auc={auc:.4f}", flush=True)

        pop_metrics = popularity_coldstart_auc(split, seed, num_dst)
        per_seed["popularity"].append(pop_metrics["auc_roc"])
        print(f"  seed={seed} mode=popularity: coldstart_auc={pop_metrics['auc_roc']:.4f}", flush=True)

    comparison = compare_models(per_seed)
    out = {"seeds": SEEDS, "per_seed_coldstart_auc": per_seed, "comparison": comparison}

    print("\n=== cold-start AUC, 7 seeds ===")
    for model, s in comparison["summary"].items():
        print(f"  {model:<12} {s['mean']:.4f} +/- {s['std']:.4f}")
    print("pairwise Wilcoxon:")
    for pair, res in comparison["pairwise"].items():
        sig = "SIG" if res["significant_at_0.05"] else "n.s."
        print(f"  {pair}: p={res['p_value']:.4f} {sig}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "phase4_inductive_multiseed_lodhalal.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
