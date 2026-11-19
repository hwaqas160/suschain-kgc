"""v5 confirmatory run: frozen settings, fresh seeds.

Settings come from configs/model_v5/substitution_*.json (scripts/46, tuned on
scenario seed 100). Evaluation seeds are 10-19, which no earlier run used. The
hypotheses and the rules that pick "the best GNN" and "the best KGE" were
written down in docs/v5/PREREGISTRATION.md before this script was run.

Per seed and method it records plain hit@10, MRR, risk@10 and the training
wall-clock; for each method it also stores per-query hit@10 and risk@10 under
every shift weight (BETAS), so the risk-budget analysis (scripts/48) needs no
retraining. For the best GNN configuration it also trains the learned penalty
(lambda 1 and 5). Rule-based rankers get rank-position scores for the shift.

Output: results/metrics/v5_confirmatory.json and v5_confirmatory_curves.npz
Run:    python -u scripts/47_v5_confirmatory.py --seeds 10-19
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suschain.decision_eval.substitutes import evaluate_substitution, rank_candidates
from suschain.decision_eval.v5_common import (BETAS, K, co_supply_scores, evaluate, load_context,
                                              risk_curves, scenario_for, train_scorer)

CONF = ROOT / "configs" / "model_v5"


def load_methods(conf=CONF):
    cfgs = {}
    for p in sorted(Path(conf).glob("substitution_*.json")):
        d = json.loads(p.read_text())
        cfgs[d["key"]] = d
    return cfgs


def pick(cfgs, prefix, suffix=""):
    """Pre-registered rule: the configuration with the highest tuning-scenario hit@10."""
    c = {k: v for k, v in cfgs.items() if k.startswith(prefix) and k.endswith(suffix)}
    return max(c, key=lambda k: c[k]["best_val_hit10"])


def rank_scores(rankings, qs, n):
    out = {}
    for r, q in zip(rankings, qs):
        s = np.zeros(n)
        s[r] = -np.arange(len(r), dtype=np.float64)
        out[id(q)] = s
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="10-19")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    ap.add_argument("--conf-dir", default=str(CONF))
    ap.add_argument("--only", default="", help="comma-separated method keys (default: all tuned)")
    a = ap.parse_args()
    lo, _, hi = a.seeds.partition("-")
    seeds = list(range(int(lo), int(hi) + 1)) if hi else [int(s) for s in a.seeds.split(",")]

    cfgs = load_methods(a.conf_dir)
    keys = [k for k in a.only.split(",") if k] or sorted(cfgs)
    best_gnn = pick(cfgs, "gnn_")
    best_kge = pick(cfgs, "kge_", "_softmax")
    print(f"methods {keys}\nbest GNN by tuning hit@10: {best_gnn}; best KGE (softmax): {best_kge}", flush=True)

    ctx = load_context(a.device)
    res = defaultdict(lambda: defaultdict(list))      # method -> metric -> per-seed list
    curves = {}

    def record(name, sc, ev, seconds=None):
        res[name]["hit@10"].append(float(ev["hit"].mean()))
        res[name]["recall@10"].append(float(ev["recall"].mean()))
        res[name]["rr"].append(float(ev["rr"].mean()))
        res[name]["risk@10"].append(float(ev["risk"].mean()))
        if seconds is not None:
            res[name]["train_seconds"].append(float(seconds))

    for seed in seeds:
        t0 = time.perf_counter()
        sc = scenario_for(ctx, seed)
        qs, n = sc.qs, ctx.n
        res["_n_queries"]["n"].append(len(qs))
        res["_n_queries"]["customers"].append(len(sc.customers))
        # rules
        r_co = [rank_candidates(co_supply_scores(sc, q), q, sc.pop + sc.tb) for q in qs]
        ev = evaluate_substitution(r_co, qs, ctx.risk, K)
        record("co_supply", sc, ev)
        r_pop = [rank_candidates(sc.pop, q, sc.tb) for q in qs]
        record("popularity", sc, evaluate_substitution(r_pop, qs, ctx.risk, K))
        look = rank_scores(r_co, qs, n)
        hit, rsk = _curves_rows(ctx, qs, look, sc.tb)
        curves[f"s{seed}/co_supply/hit"], curves[f"s{seed}/co_supply/risk"] = hit, rsk
        curves[f"s{seed}/queries/customer"] = np.array([q.customer for q in qs])

        for key in keys:
            m = cfgs[key]["best"]
            S, sec = train_scorer(ctx, sc, m, lam=0.0)
            ev = evaluate(ctx, sc, S)
            record(key, sc, ev, sec)
            hit, rsk = risk_curves(ctx, sc, S)
            curves[f"s{seed}/{key}/hit"], curves[f"s{seed}/{key}/risk"] = hit, rsk
            # time to re-rank one query under a shift (z-score, shift, sort)
            t1 = time.perf_counter()
            for q in qs[:500]:
                s = S[sc.cidx[q.customer]]
                x = s[q.candidates]
                rank_candidates(s - 1.0 * ctx.risk, q, sc.tb)
            res[key]["rerank_ms_per_query"].append((time.perf_counter() - t1) / 500 * 1e3)
            if key == best_gnn:
                for lam in (1.0, 5.0):
                    S2, sec2 = train_scorer(ctx, sc, m, lam=lam)
                    ev2 = evaluate(ctx, sc, S2)
                    record(f"{key}_lam{lam:g}", sc, ev2, sec2)
                    hit2, rsk2 = risk_curves(ctx, sc, S2)
                    curves[f"s{seed}/{key}_lam{lam:g}/hit"] = hit2
                    curves[f"s{seed}/{key}_lam{lam:g}/risk"] = rsk2
            print(f"seed {seed} {key:22} hit@10 {res[key]['hit@10'][-1]:.3f} "
                  f"risk {res[key]['risk@10'][-1]:.3f} train {sec:.0f}s", flush=True)
        print(f"seed {seed} done in {time.perf_counter() - t0:.0f}s; co_supply {res['co_supply']['hit@10'][-1]:.3f}",
              flush=True)
        out = {"seeds_done": seeds[:seeds.index(seed) + 1], "betas": BETAS, "best_gnn": best_gnn,
               "best_kge": best_kge, "configs": {k: cfgs[k]["best"] for k in keys},
               "per_seed": {m: {k: v for k, v in d.items()} for m, d in res.items()}}
        (ROOT / "results" / "metrics" / f"v5_confirmatory{a.tag}.json").write_text(json.dumps(out, indent=2))
        np.savez_compressed(ROOT / "results" / "metrics" / f"v5_confirmatory_curves{a.tag}.npz", **curves)
    print("done")


def _curves_rows(ctx, qs, look, tb):
    """risk_curves for rankers whose score vector is stored per query (rule rankers)."""
    hit = np.zeros((len(BETAS), len(qs)))
    rsk = np.zeros_like(hit)
    for j, q in enumerate(qs):
        s = look[id(q)]
        x = s[q.candidates]
        sd = x.std()
        z = np.zeros_like(s)
        z[q.candidates] = (x - x.mean()) / (sd if sd > 0 else 1.0)
        for i, b in enumerate(BETAS):
            out = evaluate_substitution([rank_candidates(z - b * ctx.risk, q, tb)], [q], ctx.risk, K)
            hit[i, j], rsk[i, j] = out["hit"][0], out["risk"][0]
    return hit, rsk


if __name__ == "__main__":
    main()
