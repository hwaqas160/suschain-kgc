"""Certificate co-occurrence baseline for LODHalal has_manufacturer.

Motivation. Under the disclosed split, 100% of test food nodes are isolated in
the has_manufacturer message-passing graph (every food has exactly one
manufacturer edge, and the test edge is held out). The only route to the
answer is food -> certificate -> other foods -> their manufacturer. Because
99% of certificates in LODHalal are attached to exactly one manufacturer, that
route is close to a lookup table.

This script implements that route as an explicit non-learned baseline on the
same splits and seeds as the GNN benchmark, so the paper can report what a
trivial relational rule achieves before crediting any learned model.

It uses strictly the information the encoder also sees: certificate edges from
the input graph (has_cert is not the prediction target) and manufacturer edges
from the TRAINING message-passing partition only.

Run: python -u scripts/22_cert_shortcut_baseline.py
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_link_prediction

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
ET = ("food", "has_manufacturer", "manufacturer")
REV = ("manufacturer", "rev_has_manufacturer", "food")
CERT = ("food", "has_cert", "cert")
SEEDS = [0, 1, 2, 3, 4, 5, 6]
KS = (1, 3, 10)


def main():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    data = built.data
    num_man = data["manufacturer"].num_nodes

    # certificate structure comes from the input graph, not the target relation
    cert_ei = data[CERT].edge_index
    foods_of_cert = defaultdict(set)
    certs_of_food = defaultdict(set)
    for f, c in zip(cert_ei[0].tolist(), cert_ei[1].tolist()):
        foods_of_cert[c].add(f)
        certs_of_food[f].add(c)

    # descriptive: how nearly 1:1 is certificate -> manufacturer?
    man_of_food_full = dict(zip(*[x.tolist() for x in data[ET].edge_index]))
    per_cert = []
    for c, fs in foods_of_cert.items():
        ms = {man_of_food_full[f] for f in fs if f in man_of_food_full}
        if ms:
            per_cert.append(len(ms))
    per_cert = np.array(per_cert)
    structure = {
        "certificates_with_linked_food": int(len(per_cert)),
        "mean_manufacturers_per_certificate": float(per_cert.mean()),
        "median_manufacturers_per_certificate": float(np.median(per_cert)),
        "max_manufacturers_per_certificate": int(per_cert.max()),
        "pct_certificates_mapping_to_one_manufacturer": float((per_cert == 1).mean() * 100),
    }
    print("certificate -> manufacturer structure:")
    for k, v in structure.items():
        print(f"  {k}: {v}")

    per_seed = {"mrr": [], "coverage": [], **{f"hits@{k}": [] for k in KS}}

    for seed in SEEDS:
        split = split_link_prediction(data, ET, REV, seed=seed)
        mp = split.train[ET].edge_index
        train_man = dict(zip(*[x.tolist() for x in mp]))

        pop = np.zeros(num_man)
        for m in mp[1].tolist():
            pop[m] += 1
        prior = pop / (pop.max() + 1e-9) * 1e-6   # negligible tie-break only

        s = split.test[ET]
        mask = s.edge_label == 1.0
        test_edges = s.edge_label_index[:, mask].t().tolist()

        rr, hits = [], {k: [] for k in KS}
        covered = 0
        for f, true_m in test_edges:
            votes = defaultdict(float)
            for c in certs_of_food.get(f, ()):
                for f2 in foods_of_cert[c]:
                    if f2 != f and f2 in train_man:
                        votes[train_man[f2]] += 1.0
            scores = prior.copy()
            if votes:
                covered += 1
                for m, v in votes.items():
                    scores[m] += v
            order = np.argsort(-scores)
            rank = int(np.where(order == true_m)[0][0]) + 1
            rr.append(1.0 / rank)
            for k in KS:
                hits[k].append(1.0 if rank <= k else 0.0)

        per_seed["mrr"].append(float(np.mean(rr)))
        per_seed["coverage"].append(covered / len(test_edges))
        for k in KS:
            per_seed[f"hits@{k}"].append(float(np.mean(hits[k])))
        print(f"  seed {seed}: MRR={np.mean(rr):.4f} Hits@10={np.mean(hits[10]):.4f} "
              f"coverage={covered/len(test_edges)*100:.1f}%", flush=True)

    summary = {k: {"mean": statistics.mean(v), "std": statistics.stdev(v)}
               for k, v in per_seed.items()}
    print("\n=== certificate co-occurrence baseline, 7 seeds ===")
    for k, v in summary.items():
        print(f"  {k:10} {v['mean']:.4f} +/- {v['std']:.4f}")

    out = {"seeds": SEEDS, "structure": structure,
           "per_seed": per_seed, "summary": summary}
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "phase1_cert_shortcut_lodhalal.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
