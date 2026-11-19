"""v5 analysis of the confirmatory run (scripts/47).

1. Monotonicity lemma check: for every query, risk@10 is non-increasing in the
   shift weight beta (it must be, for exact top-k under score - beta * risk).
2. Risk-budget guarantee. Unit of exchangeability: the customer. For a ranker
   and a risk level alpha, customers are split at random into calibration and
   test halves (200 splits per seed). Two rules choose beta on the calibration
   half:
     grid      smallest beta whose calibration mean risk is <= alpha   (v4 rule)
     crc       smallest beta with (n/(n+1)) * mean_cal(L) + 1/(n+1) <= alpha
               (conformal risk control; loss bounded by 1, non-increasing in beta)
   For crc, E[test loss] <= alpha holds in finite samples under exchangeability.
   alpha_t = (1 - t) * R0, where R0 is the ranker's plain risk@10 on the TUNING
   scenario (seed 100), so alpha is fixed before any evaluation seed is seen.
   Reported per ranker and target: mean test risk minus alpha, share of splits
   with test mean <= alpha, mean hit@10 lost against beta = 0 on the test half,
   mean beta, and the share of splits where alpha was unreachable.
3. The pre-declared tests (docs/v5/PREREGISTRATION.md) with exact Wilcoxon and Holm.

Output: results/metrics/v5_analysis.json
Run:    python -u scripts/48_v5_analysis.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suschain.decision_eval.v5_common import BETAS, K, load_context, scenario_for, train_scorer, evaluate
from suschain.eval.significance import exact_wilcoxon

TARGETS = [0.05, 0.10, 0.20, 0.30]
SPLITS = 200


def customer_matrix(arr, cust_of_query, customers):
    """[len(BETAS), n_customers]: mean over each customer's queries."""
    out = np.zeros((arr.shape[0], len(customers)))
    idx = {c: i for i, c in enumerate(customers)}
    cnt = np.zeros(len(customers))
    for j, c in enumerate(cust_of_query):
        out[:, idx[c]] += arr[:, j]
        cnt[idx[c]] += 1
    return out / cnt


def beta_rule(Lcal, alpha, rule):
    n = Lcal.shape[1]
    m = Lcal.mean(1)
    adj = (n / (n + 1.0)) * m + 1.0 / (n + 1.0) if rule == "crc" else m
    ok = np.flatnonzero(adj <= alpha)
    return (int(ok[0]), True) if len(ok) else (len(BETAS) - 1, False)


CAL_SIZES = (25, 50, 100, "half")


def budget_study(Lc, Hc, alpha, rng):
    """Per calibration size and rule: how the chosen beta behaves on held-out customers."""
    out = {}
    n = Lc.shape[1]
    # CRC needs L_i(beta_max) <= alpha for every customer: report how many customers violate it
    infeasible = float((Lc[-1] > alpha).mean())
    for size in CAL_SIZES:
        out[str(size)] = {}
        for rule in ("grid", "crc"):
            gap, within, lost, betas, reach = [], [], [], [], []
            r2 = np.random.default_rng(int(rng.integers(1 << 31)))
            for _ in range(SPLITS):
                if size == "half":
                    cal = r2.random(n) < 0.5
                else:
                    if size >= n - 5:
                        continue
                    cal = np.zeros(n, dtype=bool)
                    cal[r2.choice(n, size, replace=False)] = True
                if cal.sum() < 5 or (~cal).sum() < 5:
                    continue
                i, ok = beta_rule(Lc[:, cal], alpha, rule)
                test_risk = Lc[i, ~cal].mean()
                gap.append(test_risk - alpha)
                within.append(float(test_risk <= alpha))
                lost.append(Hc[0, ~cal].mean() - Hc[i, ~cal].mean())
                betas.append(BETAS[i])
                reach.append(float(ok))
            out[str(size)][rule] = {
                "mean_test_risk_minus_alpha": float(np.mean(gap)), "share_within": float(np.mean(within)),
                "hit_lost": float(np.mean(lost)), "mean_beta": float(np.mean(betas)),
                "share_reachable": float(np.mean(reach))}
    out["customers_above_alpha_at_beta_max"] = infeasible
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--tuning-tag", default="_tuning")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    d = json.loads((ROOT / "results" / "metrics" / f"v5_confirmatory{a.tag}.json").read_text())
    cv = np.load(ROOT / "results" / "metrics" / f"v5_confirmatory_curves{a.tag}.npz")
    seeds = d["seeds_done"]
    per = d["per_seed"]
    methods = [m for m in per if not m.startswith("_")]
    out = {"seeds": seeds, "best_gnn": d["best_gnn"], "best_kge": d["best_kge"], "betas": BETAS}

    # tuning-scenario plain risk R0 per ranker: alpha is fixed from a run on seed 100,
    # which no evaluation seed touches (scripts/47 --seeds 100 --tag _tuning)
    tp = ROOT / "results" / "metrics" / f"v5_confirmatory{a.tuning_tag}.json"
    R0 = None
    if tp.exists():
        tj = json.loads(tp.read_text())["per_seed"]
        R0 = {m: float(np.mean(v["risk@10"])) for m, v in tj.items() if not m.startswith("_")}

    # 1. monotonicity
    viol, tot, worst = 0, 0, 0.0
    for s in seeds:
        for m in methods:
            key = f"s{s}/{m}/risk"
            if key not in cv.files:
                continue
            r = cv[key]
            dif = np.diff(r, axis=0)
            viol += int((dif > 1e-6).sum())
            tot += dif.size
            worst = max(worst, float(dif.max()))
    out["monotonicity"] = {"violations": viol, "comparisons": tot, "largest_increase": worst}

    # 2. risk budget
    bud = {}
    for m in methods:
        if f"s{seeds[0]}/{m}/risk" not in cv.files or m in ("popularity",) or "_lam" in m:
            continue
        r0 = (R0 or {}).get(m)
        bud[m] = {}
        for t in TARGETS:
            rows = []
            for s in seeds:
                risk = cv[f"s{s}/{m}/risk"]
                hit = cv[f"s{s}/{m}/hit"]
                cust = cv[f"s{s}/queries/customer"]
                customers = sorted(set(cust.tolist()))
                Lc, Hc = customer_matrix(risk, cust, customers), customer_matrix(hit, cust, customers)
                base = r0 if r0 is not None else float(Lc[0].mean())
                alpha = (1 - t) * base
                rows.append(budget_study(Lc, Hc, alpha, np.random.default_rng(1000 + s)))
            bud[m][f"target{int(t * 100)}"] = {
                size: {rule: {k: float(np.mean([r[size][rule][k] for r in rows])) for k in rows[0][size][rule]}
                       for rule in ("grid", "crc")} for size in map(str, CAL_SIZES)}
            bud[m][f"target{int(t * 100)}"]["customers_above_alpha_at_beta_max"] = float(
                np.mean([r["customers_above_alpha_at_beta_max"] for r in rows]))
            bud[m][f"target{int(t * 100)}"]["alpha_from"] = "tuning scenario" if r0 is not None else "eval seed (fallback)"
    out["risk_budget"] = bud

    # 3. declared tests
    def v(m, k="hit@10"):
        return np.array(per[m][k])
    tests = {}
    bk, bg = d["best_kge"], d["best_gnn"]
    tests["H1 best KGE vs co-supply, hit@10"] = exact_wilcoxon(v(bk), v("co_supply"))
    if "nbfnet" in per:
        tests["H2 NBFNet vs co-supply, hit@10"] = exact_wilcoxon(v("nbfnet"), v("co_supply"))
    sib = bg.rsplit("_", 1)[0] + "_" + ("bce" if bg.endswith("softmax") else "softmax")
    if sib in per:
        tests[f"H3 {bg} vs {sib}, hit@10"] = exact_wilcoxon(v(bg), v(sib))
    # H4: shift vs learned penalty at equal risk, best GNN, lambda = 1
    if f"{bg}_lam1" in per:
        diffs = []
        for s in seeds:
            plain_h, plain_r = cv[f"s{s}/{bg}/hit"].mean(1), cv[f"s{s}/{bg}/risk"].mean(1)
            lh, lr = cv[f"s{s}/{bg}_lam1/hit"].mean(1)[0], cv[f"s{s}/{bg}_lam1/risk"].mean(1)[0]
            order = np.argsort(plain_r)
            shift_h = float(np.interp(lr, plain_r[order], plain_h[order]))
            diffs.append(shift_h - lh)
        out["H4_shift_minus_penalty_hit"] = {"per_seed": diffs, "mean": float(np.mean(diffs))}
        tests["H4 shift vs learned penalty at equal risk (lambda 1), hit@10"] = exact_wilcoxon(
            diffs, [0.0] * len(diffs))
    names = list(tests)
    p = np.array([tests[k] for k in names])
    order = np.argsort(p)
    holm, running = {}, 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        holm[names[i]] = {"p": float(p[i]), "p_holm": float(running)}
    out["tests"] = holm
    out["summary"] = {m: {k: {"mean": float(np.mean(v_)), "std": float(np.std(v_, ddof=1)) if len(v_) > 1 else 0.0}
                          for k, v_ in per[m].items()} for m in methods}
    (ROOT / "results" / "metrics" / f"v5_analysis{a.tag}.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"monotonicity": out["monotonicity"], "tests": holm}, indent=2))


if __name__ == "__main__":
    main()
