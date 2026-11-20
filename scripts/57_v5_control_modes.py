"""v5 post-run checks (exploratory), computed from stored per-query curves; no retraining.

1. Ways to meet a risk target (AnonNet, confirmatory seeds 10-19).
   The risk of a shortlist depends only on the scores and the risk values, so the shift
   weight need not be fixed in advance. Besides the two calibrated rules of scripts/48
   (one weight for every customer, chosen on calibration customers), three label-free
   operating modes are evaluated on every customer of a seed:
     per_list      each query gets the smallest grid weight whose own risk@10 <= alpha
     per_customer  each customer gets the smallest grid weight whose mean risk@10 <= alpha
     pooled        one weight for all customers of the seed, chosen on those same customers
                   (in-sample: the best a single weight can do without a finite-sample margin)
   alpha = (1 - t) * R0 with R0 from the tuning scenario, as in scripts/48.
   Reported: mean customer risk minus alpha, share of customers within alpha,
   hit@10 lost (customer-weighted, as in scripts/48) and the share of lists or customers for
   which no grid weight reaches alpha.

2. Trase importer-level re-analysis (model seed 0 curves).
   For a learned ranker the candidates, the answers and the scores of an importer's queries
   do not depend on the dropped exporter, so an importer's queries are copies of one outcome.
   hit@10 is recomputed with every importer weighted equally, per year, and tested over years.
   Also: per-list control on Trase and the split into years before and after the 2019 break.

Output: results/metrics/v5_control_modes.json
Run:    py -u scripts/57_v5_control_modes.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from suschain.eval.significance import exact_wilcoxon  # noqa: E402

BETAS = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0]
TARGETS = [0.10, 0.20, 0.30]
M = ROOT / "results" / "metrics"


def per_customer(arr, cust):
    cs = np.unique(cust)
    idx = np.searchsorted(cs, cust)
    out = np.zeros((arr.shape[0], len(cs)))
    np.add.at(out.T, idx, arr.T)
    return out / np.bincount(idx)[None, :], idx, cs


def first_ok(L, alpha):
    """L: [B, n]. Index of the smallest weight with L <= alpha per column (last if none)."""
    ok = L <= alpha + 1e-12
    any_ok = ok.any(0)
    return np.where(any_ok, ok.argmax(0), L.shape[0] - 1), ~any_ok


def modes(risk, hit, cust, alpha):
    Lc, idx, cs = per_customer(risk, cust)
    Hc, _, _ = per_customer(hit, cust)
    n_c = len(cs)
    res = {}
    # per list
    j, inf = first_ok(risk, alpha)
    q = np.arange(risk.shape[1])
    r_sel, h_sel = risk[j, q], hit[j, q]
    Lsel = np.bincount(idx, r_sel, n_c) / np.bincount(idx)
    Hsel = np.bincount(idx, h_sel, n_c) / np.bincount(idx)
    res["per_list"] = dict(gap=float(Lsel.mean() - alpha), within=float((Lsel <= alpha + 1e-12).mean()),
                           lost=float(Hc[0].mean() - Hsel.mean()), infeasible=float(inf.mean()),
                           lists_within=float((r_sel <= alpha + 1e-12).mean()))
    # per customer
    j, inf = first_ok(Lc, alpha)
    c = np.arange(n_c)
    Lsel, Hsel = Lc[j, c], Hc[j, c]
    res["per_customer"] = dict(gap=float(Lsel.mean() - alpha), within=float((Lsel <= alpha + 1e-12).mean()),
                               lost=float(Hc[0].mean() - Hsel.mean()), infeasible=float(inf.mean()))
    # pooled in-sample single weight
    m = Lc.mean(1)
    ok = np.flatnonzero(m <= alpha + 1e-12)
    b = int(ok[0]) if len(ok) else len(BETAS) - 1
    res["pooled"] = dict(gap=float(Lc[b].mean() - alpha), within=float((Lc[b] <= alpha + 1e-12).mean()),
                         lost=float(Hc[0].mean() - Hc[b].mean()), beta=BETAS[b])
    return res


def anonnet():
    d = json.loads((M / "v5_confirmatory.json").read_text())
    tj = json.loads((M / "v5_confirmatory_tuning.json").read_text())["per_seed"]
    cv = np.load(M / "v5_confirmatory_curves.npz")
    an = json.loads((M / "v5_analysis.json").read_text())["risk_budget"]
    seeds = d["seeds_done"]
    rankers = [d["best_kge"], "nbfnet", d["best_gnn"], "co_supply"]
    out = {}
    for m in rankers:
        R0 = float(np.mean(tj[m]["risk@10"]))
        out[m] = {}
        for t in TARGETS:
            alpha = (1 - t) * R0
            rows = [modes(cv[f"s{s}/{m}/risk"], cv[f"s{s}/{m}/hit"], cv[f"s{s}/queries/customer"], alpha) for s in seeds]
            agg = {mode: {k: float(np.mean([r[mode][k] for r in rows])) for k in rows[0][mode]} for mode in rows[0]}
            tk = f"target{int(t * 100)}"
            for size in ("25", "100", "half"):
                for rule in ("grid", "crc"):
                    e = an[m][tk][size][rule]
                    agg[f"{rule}_{size}"] = dict(gap=e["mean_test_risk_minus_alpha"], within_splits=e["share_within"],
                                                 lost=e["hit_lost"])
            out[m][tk] = agg
    return out


def trase():
    cv = np.load(M / "v5_trase_curves.npz")
    years = sorted({k.split("/")[0] for k in cv.files})
    rankers = ["kge_RotatE_softmax", "nbfnet", "gnn_sage_bce", "gnn_sage_softmax", "kge_ComplEx_softmax", "co_supply"]
    per = {m: {} for m in rankers}
    conc = {}
    dup = {}
    for y in years:
        cust = cv[f"{y}/queries/customer"]
        cs, idx, cnt = np.unique(cust, return_inverse=True, return_counts=True)
        conc[y] = dict(queries=int(len(cust)), importers=int(len(cs)),
                       top5_share=float(np.sort(cnt)[::-1][:5].sum() / len(cust)))
        for m in rankers:
            h = cv[f"{y}/{m}/hit"][0]
            hi = np.bincount(idx, h) / cnt
            per[m][y] = dict(query_weighted=float(h.mean()), importer_weighted=float(hi.mean()))
            if m == "kge_RotatE_softmax":
                # are an importer's queries identical for a learned ranker?
                same = [np.ptp(h[idx == i]) == 0 for i in range(len(cs))]
                dup[y] = float(np.mean(same))
    tests = {}
    for m in rankers[:-1]:
        for w in ("query_weighted", "importer_weighted"):
            a = np.array([per[m][y][w] for y in years])
            b = np.array([per["co_supply"][y][w] for y in years])
            tests[f"{m}/{w}"] = dict(mean=float(a.mean()), co_supply=float(b.mean()), diff=float((a - b).mean()),
                                     years_ahead=int((a > b).sum()), p=float(exact_wilcoxon(a, b)))
    early = [y for y in years if int(y) <= 2018]
    late = [y for y in years if int(y) >= 2020]
    split = {m: {g: float(np.mean([per[m][y]["importer_weighted"] for y in ys])) for g, ys in
                 (("2014_2018", early), ("2020_2022", late))} for m in rankers}
    # per-list control on Trase for RotatE, 20% target from its 2016 risk (R0 = 0.070 in the paper)
    tr = json.loads((M / "v5_trase_analysis.json").read_text())
    R0 = tr["risk_budget"]["kge_RotatE_softmax"]["R0_tuning_year"]
    ctrl = {}
    if R0 is not None:
        for t in TARGETS:
            alpha = (1 - t) * float(R0)
            rows = [modes(cv[f"{y}/kge_RotatE_softmax/risk"], cv[f"{y}/kge_RotatE_softmax/hit"],
                          cv[f"{y}/queries/customer"], alpha) for y in years]
            ctrl[f"target{int(t * 100)}"] = {mode: {k: float(np.mean([r[mode][k] for r in rows])) for k in rows[0][mode]}
                                            for mode in ("per_list", "per_customer")}
    return dict(per_year=per, concentration=conc, learned_queries_identical_within_importer=dup,
                tests=tests, early_late=split, R0=R0, control=ctrl)


if __name__ == "__main__":
    res = dict(anonnet=anonnet(), trase=trase())
    (M / "v5_control_modes.json").write_text(json.dumps(res, indent=2))
    a = res["anonnet"]
    for m in a:
        for tk in a[m]:
            e = a[m][tk]
            print(f"{m:22s} {tk}  per_list lost {e['per_list']['lost']:.3f} gap {e['per_list']['gap']:+.4f} inf {e['per_list']['infeasible']:.3f} | "
                  f"per_cust lost {e['per_customer']['lost']:.3f} within {e['per_customer']['within']:.3f} | pooled lost {e['pooled']['lost']:.3f} | "
                  f"crc25 {e['crc_25']['lost']:.3f} crc100 {e['crc_100']['lost']:.3f} crcHalf {e['crc_half']['lost']:.3f} gridHalf {e['grid_half']['lost']:.3f}")
    t = res["trase"]
    print(json.dumps(t["tests"], indent=1))
    print(json.dumps(t["concentration"], indent=1))
    print("identical:", t["learned_queries_identical_within_importer"])
    print("early/late:", json.dumps(t["early_late"], indent=1))
    print("R0", t["R0"], json.dumps(t["control"], indent=1))
