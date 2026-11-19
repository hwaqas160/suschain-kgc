"""v5: pre-registered H5 (per-seed guarantee check) and exploratory penalty comparisons.

scripts/48 reports the risk budget averaged over seeds; H5 in docs/v5/PREREGISTRATION.md is
stated per seed: for the best KGE and targets of 10% and 20%, the conformal rule must have
mean test risk <= alpha + 0.001 (averaged over 200 calibration/test splits of the
customers) in at least 9 of the 10 seeds. The calibration share is half of the customers,
the setting of the manuscript; the other calibration sizes are reported alongside. The
splits use the same functions and random streams as scripts/48.

Exploratory (not pre-registered): the learned penalty at lambda = 5 against the shift at
equal risk, and the shift at the penalty's hit@10 cost.

Output: results/metrics/v5_h5_penalty.json
"""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from suschain.eval.significance import exact_wilcoxon  # noqa: E402

_s = importlib.util.spec_from_file_location("s48", ROOT / "scripts" / "48_v5_analysis.py")
s48 = importlib.util.module_from_spec(_s)
_s.loader.exec_module(s48)

MET = ROOT / "results" / "metrics"


def main():
    d = json.loads((MET / "v5_confirmatory.json").read_text())
    cv = np.load(MET / "v5_confirmatory_curves.npz")
    tj = json.loads((MET / "v5_confirmatory_tuning.json").read_text())["per_seed"]
    seeds, bk, bg = d["seeds_done"], d["best_kge"], d["best_gnn"]
    out = {"best_kge": bk, "best_gnn": bg, "seeds": seeds}

    # H5
    r0 = float(np.mean(tj[bk]["risk@10"]))
    h5 = {"R0_tuning": r0}
    for t in (0.10, 0.20):
        alpha = (1 - t) * r0
        rows = []
        for s in seeds:
            cust = cv[f"s{s}/queries/customer"]
            customers = sorted(set(cust.tolist()))
            Lc = s48.customer_matrix(cv[f"s{s}/{bk}/risk"], cust, customers)
            Hc = s48.customer_matrix(cv[f"s{s}/{bk}/hit"], cust, customers)
            res = s48.budget_study(Lc, Hc, alpha, np.random.default_rng(1000 + s))
            rows.append({size: {rule: res[size][rule] for rule in ("grid", "crc")} for size in map(str, s48.CAL_SIZES)})
        per = {}
        for size in map(str, s48.CAL_SIZES):
            per[size] = {}
            for rule in ("grid", "crc"):
                gaps = [r[size][rule]["mean_test_risk_minus_alpha"] for r in rows]
                per[size][rule] = {"per_seed_gap": gaps, "seeds_within_alpha_plus_0.001": int(sum(g <= 0.001 for g in gaps)),
                                   "hit_lost_mean": float(np.mean([r[size][rule]["hit_lost"] for r in rows]))}
        h5[f"target{int(t * 100)}"] = {"alpha": alpha, **per}
        h5[f"target{int(t * 100)}"]["pass"] = per["half"]["crc"]["seeds_within_alpha_plus_0.001"] >= 9
    out["H5"] = h5

    # exploratory: penalty vs shift
    pen = {}
    for lam in ("lam1", "lam5"):
        eq, cost, dr_pen, dr_shift = [], [], [], []
        for s in seeds:
            ph, pr = cv[f"s{s}/{bg}/hit"].mean(1), cv[f"s{s}/{bg}/risk"].mean(1)
            lh, lr = cv[f"s{s}/{bg}_{lam}/hit"].mean(1)[0], cv[f"s{s}/{bg}_{lam}/risk"].mean(1)[0]
            o = np.argsort(pr)
            eq.append(float(np.interp(lr, pr[o], ph[o])) - lh)
            # shift with the same hit@10 cost as the penalty: risk reached
            loss = ph[0] - lh
            # lowest risk on the beta grid whose hit@10 is at least the penalty's
            r_at = float(pr[ph >= lh].min()) if (ph >= lh).any() else float(pr[0])
            dr_pen.append((pr[0] - lr) / pr[0])
            dr_shift.append((pr[0] - r_at) / pr[0])
            cost.append(loss)
        pen[lam] = {"shift_minus_penalty_hit_at_equal_risk": {"per_seed": eq, "mean": float(np.mean(eq)),
                                                               "p": exact_wilcoxon(eq, [0.0] * len(eq)),
                                                               "seeds_shift_better": int(sum(e > 0 for e in eq))},
                    "penalty_hit_cost_mean": float(np.mean(cost)),
                    "penalty_risk_reduction_mean": float(np.mean(dr_pen)),
                    "shift_risk_reduction_at_same_cost_mean": float(np.mean(dr_shift))}
    out["penalty_exploratory"] = pen
    (MET / "v5_h5_penalty.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
