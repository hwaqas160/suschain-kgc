"""v5 robustness check (exploratory, added after the confirmatory run): reverse-link exposure.

The substitution scenario hides directed links independently. AnonNet has 5,722 reciprocal links,
so for some hidden supplier links v->c the opposite link c->v stays visible. A query is
"reverse-exposed" when at least one of its correct answers v has c->v visible: learned models,
which pass messages or use reciprocal triples over visible links in both directions, can see a
direct link between c and v; co-supply cannot.

This script splits the stored confirmatory per-query outcomes (no retraining) into clean and
reverse-exposed queries and repeats the main comparisons on the clean queries.

Output: results/metrics/v5_reverse_exposure.json
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from suschain.decision_eval.v5_common import load_context, scenario_for  # noqa: E402
from suschain.eval.significance import exact_wilcoxon  # noqa: E402

MET = ROOT / "results" / "metrics"
SEEDS = list(range(10, 20))


def main():
    ctx = load_context("cpu")
    cv = np.load(MET / "v5_confirmatory_curves.npz")
    methods = sorted({k.split("/")[1] for k in cv.files if k.startswith("s10/") and "/queries/" not in k})
    per = {m: {"all": [], "clean": [], "exposed": []} for m in methods}
    share = []
    for seed in SEEDS:
        sc = scenario_for(ctx, seed)
        visible = set(map(tuple, sc.tr.T.tolist()))  # (supplier, customer) pairs
        cust = np.array([q.customer for q in sc.qs])
        assert (cv[f"s{seed}/queries/customer"] == cust).all(), "query order differs from the stored run"
        exposed = np.array([any((q.customer, int(v)) in visible for v in q.truth) for q in sc.qs])
        share.append(float(exposed.mean()))
        for m in methods:
            h = cv[f"s{seed}/{m}/hit"][0]  # beta = 0, the plain ranking
            per[m]["all"].append(float(h.mean()))
            per[m]["clean"].append(float(h[~exposed].mean()))
            per[m]["exposed"].append(float(h[exposed].mean()))
        print(f"seed {seed}: {exposed.mean():.3f} of queries reverse-exposed", flush=True)

    summary = {m: {k: float(np.mean(v)) for k, v in d.items()} for m, d in per.items()}
    tests = {}
    pairs = [("kge_RotatE_softmax", "co_supply"), ("nbfnet", "co_supply"), ("gnn_gat_softmax", "co_supply"),
             ("gnn_sage_softmax", "co_supply"), ("gnn_gat_softmax", "gnn_gat_bce"), ("nbfnet", "kge_RotatE_softmax")]
    for a, b in pairs:
        d = np.array(per[a]["clean"]) - np.array(per[b]["clean"])
        tests[f"{a} - {b} (clean queries)"] = {"mean_diff": float(d.mean()), "seeds_ahead": int((d > 0).sum()),
                                                "p_exact_wilcoxon": float(exact_wilcoxon(per[a]["clean"], per[b]["clean"]))}
    out = {"seeds": SEEDS, "share_exposed_per_seed": share, "share_exposed_mean": float(np.mean(share)),
           "summary_hit10": summary, "per_seed_hit10": per, "tests": tests,
           "note": "exploratory; computed after the confirmatory run from stored per-query outcomes"}
    (MET / "v5_reverse_exposure.json").write_text(json.dumps(out, indent=2))
    for m in ["co_supply", "kge_RotatE_softmax", "nbfnet", "gnn_gat_softmax", "gnn_sage_softmax", "gnn_gat_bce", "gnn_sage_bce"]:
        print(m, {k: round(v, 4) for k, v in summary[m].items()})
    for k, v in tests.items():
        print(k, v)


if __name__ == "__main__":
    main()
