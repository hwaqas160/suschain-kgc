"""Certificate co-occurrence baseline, evaluated through the SAME harness as
every other heuristic in Table 2 (validation-tuned threshold, full-catalog
filtered ranking, 99-candidate sampled-pool ranking) so it produces a
directly comparable Accuracy/AUC-ROC/F1/MRR/Hits@10 row rather than the
ranking-only numbers from scripts/22.

Why this exists. scripts/22 and 23 showed that 99% of LODHalal certificates
map to exactly one manufacturer, and that a zero-parameter rule voting by
co-certified foods reaches MRR ~0.96 on the same split every GNN in Table 2 is
evaluated on -- beating RGCN (0.50) by 2x. That is strong enough that it
belongs in Table 2 itself as a named baseline, not a footnote, using the exact
evaluation code path (suschain.eval.heuristic_eval.evaluate_heuristic) already
used for Popularity so every number in the table is apples-to-apples.

Run: python scripts/25_cert_shortcut_full_metrics.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_link_prediction
from suschain.eval.heuristic_eval import evaluate_heuristic
from suschain.eval.ranking import build_known_positives
from suschain.models.heuristics import popularity_scores

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
ET = ("food", "has_manufacturer", "manufacturer")
REV = ("manufacturer", "rev_has_manufacturer", "food")
CERT = ("food", "has_cert", "cert")
SEEDS = [0, 1, 2, 3, 4, 5, 6]


class CertCooccurrenceScorer:
    """score_row(src=food_id) -> score per manufacturer, using ONLY the
    certificates food `src` holds (input-graph information, not the target
    relation) and the TRAINING-split has_manufacturer edges of every other
    food sharing at least one of those certificates. A negligible popularity
    prior breaks ties and covers foods no certificate resolves."""

    def __init__(self, foods_of_cert, certs_of_food, train_man_of_food, popularity_prior):
        self.foods_of_cert = foods_of_cert
        self.certs_of_food = certs_of_food
        self.train_man_of_food = train_man_of_food
        self.prior = popularity_prior

    def score_row(self, src: int, method: str = "cert_cooccurrence") -> np.ndarray:
        scores = self.prior.copy()
        for c in self.certs_of_food.get(src, ()):
            for f2 in self.foods_of_cert[c]:
                if f2 == src:
                    continue
                m2 = self.train_man_of_food.get(f2)
                if m2 is not None:
                    scores[m2] += 1.0
        return scores


def run_seed(built, seed):
    data = built.data
    split = split_link_prediction(data, ET, REV, seed=seed)
    num_man = data["manufacturer"].num_nodes

    cert_ei = data[CERT].edge_index
    foods_of_cert = defaultdict(set)
    certs_of_food = defaultdict(set)
    for f, c in zip(cert_ei[0].tolist(), cert_ei[1].tolist()):
        foods_of_cert[c].add(f)
        certs_of_food[f].add(c)

    mp = split.train[ET].edge_index
    train_man_of_food = dict(zip(mp[0].tolist(), mp[1].tolist()))

    pop = popularity_scores(mp, num_man)
    prior = pop / (pop.max() + 1e-9) * 1e-6  # negligible tie-break only

    scorer = CertCooccurrenceScorer(foods_of_cert, certs_of_food, train_man_of_food, prior)

    known_pos = build_known_positives(
        split.train[ET].edge_index,
        split.val[ET].edge_label_index[:, split.val[ET].edge_label == 1.0],
        split.test[ET].edge_label_index[:, split.test[ET].edge_label == 1.0],
    )

    return evaluate_heuristic(
        scorer, "cert_cooccurrence",
        split.val[ET].edge_label_index, split.val[ET].edge_label,
        split.test[ET].edge_label_index, split.test[ET].edge_label,
        known_pos, num_man, seed=seed,
    )


def main():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)

    per_seed = []
    for seed in SEEDS:
        r = run_seed(built, seed)
        per_seed.append(r)
        print(f"  seed {seed}: acc={r['accuracy']:.4f} auc={r['auc_roc']:.4f} f1={r['f1']:.4f} "
              f"mrr={r['mrr']:.4f} hits@10={r['hits@10']:.4f}", flush=True)

    import statistics
    keys = ["accuracy", "auc_roc", "f1", "mrr", "hits@10"]
    summary = {k: {"mean": statistics.mean(r[k] for r in per_seed),
                   "std": statistics.stdev(r[k] for r in per_seed)} for k in keys}
    print("\n=== cert_cooccurrence heuristic, full metric suite, 7 seeds ===")
    for k, v in summary.items():
        print(f"  {k:10} {v['mean']:.4f} +/- {v['std']:.4f}")

    out = {"seeds": SEEDS, "per_seed": per_seed, "summary": summary}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "phase1_cert_cooccurrence_full_lodhalal.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
