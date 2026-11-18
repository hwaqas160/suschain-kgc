"""v4 work package 8: the non-learned baselines re-evaluated with
  - ranks under the optimistic, realistic and pessimistic tie policies, and
  - accuracy at the accuracy-optimal validation threshold, beside accuracy at
    the F1-optimal threshold used so far,
over 10 split seeds (heuristics are deterministic given the split).

Why. eval/heuristic_eval.py and eval/ranking.py rank the true item as
1 + (#candidates scoring strictly higher), which counts every tie in its
favour. GNN scores are continuous and almost never tie, so this matters only
for the heuristics: Common Neighbors and Adamic-Adar score most pairs 0, and
popularity ties every manufacturer with the same degree. The F1-optimal
threshold can select "everything positive" for such scorers (accuracy 0.5,
F1 0.667), which explains the 0.5000 / 0.6667 rows in the manuscript.

Output: results/metrics/v4_heuristics_ties.json
Run:    python -u scripts/39_v4_heuristics_ties.py
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_link_prediction
from suschain.eval.heuristic_eval import PopularityScorer, _score_pairs
from suschain.eval.metrics import (best_accuracy_threshold, best_f1_threshold, binary_link_metrics,
                                   ranks_with_ties)
from suschain.eval.ranking import build_known_positives
from suschain.models.heuristics import HeuristicScorer, build_adjacency, popularity_scores

RESULTS = Path(__file__).resolve().parents[1] / "results" / "metrics"
SEEDS = list(range(10))
POLICIES = ("optimistic", "realistic", "pessimistic")


def evaluate(scorer, method, split, et, num_nodes):
    val, test = split.val[et], split.test[et]
    vs, _ = _score_pairs(scorer, method, val.edge_label_index)
    ts, rows = _score_pairs(scorer, method, test.edge_label_index)
    vl, tl = val.edge_label.numpy(), test.edge_label.numpy()
    m_f1 = binary_link_metrics(tl, ts, threshold=best_f1_threshold(vl, vs))
    m_acc = binary_link_metrics(tl, ts, threshold=best_accuracy_threshold(vl, vs))
    known = build_known_positives(split.train[et].edge_index,
                                  val.edge_label_index[:, val.edge_label == 1.0],
                                  test.edge_label_index[:, test.edge_label == 1.0])
    pos = test.edge_label_index[:, test.edge_label == 1.0].numpy()
    ranks = {p: [] for p in POLICIES}
    for s, d in pos.T:
        row = rows.get(int(s))
        row = scorer.score_row(int(s), method) if row is None else row
        row = row.astype(np.float64).copy()
        others = [k for k in known.get(int(s), ()) if k != d]
        if others:
            row[others] = -np.inf
        for p, r in ranks_with_ties(row, row[d]).items():
            ranks[p].append(r)
    out = {"auc_roc": m_f1["auc_roc"],
           "accuracy_f1_threshold": m_f1["accuracy"], "f1_f1_threshold": m_f1["f1"],
           "accuracy_acc_threshold": m_acc["accuracy"], "f1_acc_threshold": m_acc["f1"]}
    for p, r in ranks.items():
        r = np.array(r)
        out[f"mrr_{p}"] = float(np.mean(1 / r))
        out[f"hits@10_{p}"] = float(np.mean(r <= 10))
    return out


def main():
    results = {"anonnet": {}, "lodhalal": {}}
    ga = load_anonnet()
    ba = build_anonnet_heterodata(ga, make_undirected=True)
    eta = ("node", "supplies", "node")
    na = ba.data["node"].num_nodes
    gl = load_lodhalal()
    bl = build_lodhalal_heterodata(gl, make_undirected=True)
    etl = ("food", "has_manufacturer", "manufacturer")
    rvl = ("manufacturer", "rev_has_manufacturer", "food")
    nm = bl.data["manufacturer"].num_nodes

    for seed in SEEDS:
        sp_a = split_link_prediction(ba.data, eta, rev_edge_type=None, seed=seed)
        sc = HeuristicScorer(build_adjacency(sp_a.train[eta].edge_index, na))
        for method in ("common_neighbors", "adamic_adar", "preferential_attachment"):
            r = evaluate(sc, method, sp_a, eta, na)
            for k, v in r.items():
                results["anonnet"].setdefault(method, {}).setdefault(k, []).append(v)
        sp_l = split_link_prediction(bl.data, etl, rvl, seed=seed)
        pop = PopularityScorer(popularity_scores(sp_l.train[etl].edge_index, nm))
        r = evaluate(pop, "popularity", sp_l, etl, nm)
        for k, v in r.items():
            results["lodhalal"].setdefault("popularity", {}).setdefault(k, []).append(v)
        print(f"seed {seed}: CN mrr opt/real/pess = "
              f"{results['anonnet']['common_neighbors']['mrr_optimistic'][-1]:.4f}/"
              f"{results['anonnet']['common_neighbors']['mrr_realistic'][-1]:.4f}/"
              f"{results['anonnet']['common_neighbors']['mrr_pessimistic'][-1]:.4f}  "
              f"AA acc F1-thr/acc-thr = {results['anonnet']['adamic_adar']['accuracy_f1_threshold'][-1]:.4f}/"
              f"{results['anonnet']['adamic_adar']['accuracy_acc_threshold'][-1]:.4f}", flush=True)

    summary = {ds: {m: {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1))}
                        for k, v in d.items()} for m, d in dd.items()} for ds, dd in results.items()}
    p = RESULTS / "v4_heuristics_ties.json"
    p.write_text(json.dumps({"seeds": SEEDS, "per_seed": results, "summary": summary}, indent=2))
    for ds, dd in summary.items():
        for m, d in dd.items():
            print(f"{ds:8} {m:24} MRR opt {d['mrr_optimistic']['mean']:.4f} real {d['mrr_realistic']['mean']:.4f} "
                  f"pess {d['mrr_pessimistic']['mean']:.4f} | H@10 real {d['hits@10_realistic']['mean']:.4f} | "
                  f"acc F1-thr {d['accuracy_f1_threshold']['mean']:.4f} acc-thr {d['accuracy_acc_threshold']['mean']:.4f}")
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
