"""v5: observed supplier substitution on Trase Brazil soy (importer <- exporter).

See docs/v5/PREREGISTRATION_TRASE.md for the task, the years and the hypotheses.

  python -u scripts/50_v5_trase.py --phase tune     # tuning year 2016
  python -u scripts/50_v5_trase.py --phase eval     # evaluation years, model seeds 0-2

Output: configs/model_v5/trase_*.json, results/metrics/v5_trase.json and
v5_trase_curves.npz (per-query hit/risk under every shift weight).
"""
import argparse
import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import scipy.sparse as sp
import torch
from torch_geometric.data import HeteroData

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from suschain.decision_eval.substitutes import SubstitutionQuery, evaluate_substitution, rank_candidates
from suschain.decision_eval.v5_common import ET, K, evaluate, risk_curves, train_scorer
from suschain.eval.significance import exact_wilcoxon

CSV = ROOT / "data" / "external" / "brazil_soy_v2_6_1_composite.csv"
CONF = ROOT / "configs" / "model_v5"
TUNE_YEAR = 2016
EVAL_YEARS = [2014, 2015, 2017, 2018, 2020, 2021, 2022]
MODEL_SEEDS = [0, 1, 2]
EXPO = "soy_deforestation_5_year_total_exposure"

_spec = importlib.util.spec_from_file_location("s46", ROOT / "scripts" / "46_v5_tune_substitution.py")
s46 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s46)
_spec48 = importlib.util.spec_from_file_location("s48", ROOT / "scripts" / "48_v5_analysis.py")
s48 = importlib.util.module_from_spec(_spec48)
_spec48.loader.exec_module(s48)

METHODS = ["gnn_sage_bce", "gnn_sage_softmax", "kge_RotatE_softmax", "kge_ComplEx_softmax", "nbfnet"]


class World:
    pass


def load_world(device):
    d = pd.read_csv(CSV, usecols=["year", "importer", "exporter", "volume", EXPO])
    d = d[~d.importer.str.contains("UNKNOWN", na=True) & ~d.exporter.str.contains("UNKNOWN", na=True)]
    d = d.groupby(["year", "importer", "exporter"], as_index=False).agg(volume=("volume", "sum"), expo=(EXPO, "sum"))
    w = World()
    imps = sorted(d.importer.unique())
    exps = sorted(d.exporter.unique())
    w.imp = {x: i for i, x in enumerate(imps)}
    w.exp = {x: len(imps) + i for i, x in enumerate(exps)}
    w.n = len(imps) + len(exps)
    d["c"], d["s"] = d.importer.map(w.imp), d.exporter.map(w.exp)
    w.d = d
    # exporter-year deforestation exposure per tonne
    g = d.groupby(["year", "s"], as_index=False).agg(volume=("volume", "sum"), expo=("expo", "sum"))
    g["intensity"] = g.expo / g.volume.clip(lower=1e-9)
    w.intensity = g[g.year >= 2013]
    data = HeteroData()
    data["node"].num_nodes = w.n
    data["node"].node_id = torch.arange(w.n)
    data[ET].edge_index = torch.zeros(2, 1, dtype=torch.long)
    w.data, w.device = data, device
    return w


def risk_before(w, year):
    g = w.intensity[w.intensity.year < year].sort_values("year").groupby("s").intensity.last()
    r = np.full(w.n, np.nan)
    r[g.index.to_numpy()] = g.to_numpy()
    cap = np.nanpercentile(r[~np.isnan(r)], 99)
    r = np.minimum(r, cap)
    lo, hi = np.nanmin(r), np.nanmax(r)
    r = (r - lo) / (hi - lo)
    r[np.isnan(r)] = np.nanmean(r)
    return r


def scenario(w, year, seed):
    d = w.d
    vis = d[d.year < year][["s", "c"]].drop_duplicates()
    prev = d[d.year == year - 1].groupby("c").s.apply(set).to_dict()
    cur = d[d.year == year].groupby("c").s.apply(set).to_dict()
    seen = d[d.year < year].groupby("c").s.apply(set).to_dict()
    active = np.array(sorted(d[d.year == year].s.unique()))
    tr = vis.to_numpy().T.copy()
    qs = []
    for c, S in prev.items():
        if c not in cur:
            continue
        new = cur[c] - seen[c]
        if not new:
            continue
        for s in sorted(S - cur[c]):
            cand = active[~np.isin(active, list(seen[c] | {s}))]
            truth = np.array(sorted(new))
            if np.isin(truth, cand).all():
                qs.append(SubstitutionQuery(int(c), int(s), truth, cand))
    sc = type("Scenario", (), {})()
    sc.qs, sc.tr, sc.seed, sc.year = qs, tr, seed, year
    sc.tb = np.random.default_rng(seed).random(w.n)
    A = sp.csr_matrix((np.ones(tr.shape[1]), (tr[0], tr[1])), shape=(w.n, w.n))
    A.data[:] = 1.0
    sc.pop = np.asarray(A.sum(axis=1)).ravel()
    sc.co = (A @ A.T).tocsr()
    sc.customers = sorted({q.customer for q in qs})
    sc.cidx = {c: i for i, c in enumerate(sc.customers)}
    return sc


def ctx_for(w, year):
    c = type("Ctx", (), {})()
    c.data, c.n, c.device = w.data, w.n, w.device
    c.risk = risk_before(w, year)
    c.risk_norm = torch.tensor(c.risk, dtype=torch.float32)
    return c


def tune(a):
    w = load_world(a.device)
    ctx = ctx_for(w, TUNE_YEAR)
    sc = scenario(w, TUNE_YEAR, 0)
    print(f"tuning year {TUNE_YEAR}: {len(sc.qs)} queries, {len(sc.customers)} customers, {w.n} nodes", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    for key in METHODS:
        t0 = time.perf_counter()
        if key == "nbfnet":
            trials = []
            for dim in (32, 64):
                for layers in (2, 3):
                    m = dict(family="nbfnet", dim=dim, layers=layers, seconds=90.0)
                    S, sec = train_scorer(ctx, sc, m)
                    v = float(evaluate(ctx, sc, S)["hit"].mean())
                    trials.append({"params": {"dim": dim, "layers": layers}, "val_hit10": v, "seconds": sec})
                    print(f"  nbfnet {dim}/{layers}: {v:.4f}", flush=True)
            b = max(trials, key=lambda t: t["val_hit10"])
            best, val = dict(family="nbfnet", seconds=300.0, **b["params"]), b["val_hit10"]
        else:
            log = []

            def obj(trial):
                m = s46.suggest(key, trial)
                S, sec = train_scorer(ctx, sc, m)
                v = float(evaluate(ctx, sc, S)["hit"].mean())
                log.append({"params": dict(trial.params), "val_hit10": v, "seconds": sec})
                print(f"  {key} trial {len(log)}: {v:.4f} ({sec:.0f}s)", flush=True)
                return v
            study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=0))
            study.optimize(obj, n_trials=12 if key.startswith("gnn") else 8)
            best, val, trials = s46.suggest(key, optuna.trial.FixedTrial(study.best_params)), study.best_value, log
        (CONF / f"trase_{key}.json").write_text(json.dumps(
            {"key": key, "tuning_year": TUNE_YEAR, "best": best, "best_val_hit10": float(val), "trials": trials,
             "total_seconds": time.perf_counter() - t0}, indent=2))
        print(f"== {key}: tuning-year hit@10 {val:.4f}", flush=True)


def evaluate_years(a):
    w = load_world(a.device)
    cfgs = {k: json.loads((CONF / f"trase_{k}.json").read_text()) for k in METHODS}
    best = max((k for k in METHODS), key=lambda k: cfgs[k]["best_val_hit10"])
    print("best learned ranker by tuning-year hit@10:", best, flush=True)
    res = defaultdict(lambda: defaultdict(dict))
    curves = {}
    for year in EVAL_YEARS:
        ctx = ctx_for(w, year)
        per = defaultdict(lambda: defaultdict(list))
        for seed in MODEL_SEEDS:
            sc = scenario(w, year, seed)
            qs = sc.qs
            if seed == MODEL_SEEDS[0]:
                res["_n"][year] = {"queries": len(qs), "customers": len(sc.customers)}
            r_co = [rank_candidates(np.asarray(sc.co[q.disrupted].todense()).ravel(), q, sc.pop + sc.tb) for q in qs]
            ev = evaluate_substitution(r_co, qs, ctx.risk, K)
            for k_, v_ in (("hit", ev["hit"]), ("rr", ev["rr"]), ("risk", ev["risk"])):
                per["co_supply"][k_].append(float(v_.mean()))
            ev = evaluate_substitution([rank_candidates(sc.pop, q, sc.tb) for q in qs], qs, ctx.risk, K)
            for k_, v_ in (("hit", ev["hit"]), ("rr", ev["rr"]), ("risk", ev["risk"])):
                per["popularity"][k_].append(float(v_.mean()))
            rnd = np.random.default_rng(seed + 77).random(w.n)
            ev = evaluate_substitution([rank_candidates(rnd, q, None) for q in qs], qs, ctx.risk, K)
            for k_, v_ in (("hit", ev["hit"]), ("rr", ev["rr"]), ("risk", ev["risk"])):
                per["random"][k_].append(float(v_.mean()))
            if seed == MODEL_SEEDS[0]:
                rows = {id(q): np.zeros(w.n) for q in qs}
                for r, q in zip(r_co, qs):
                    rows[id(q)][r] = -np.arange(len(r), dtype=np.float64)
                hit, rsk = _rule_curves(ctx, qs, rows, sc.tb)
                curves[f"{year}/co_supply/hit"], curves[f"{year}/co_supply/risk"] = hit, rsk
                curves[f"{year}/queries/customer"] = np.array([q.customer for q in qs])
            for key in METHODS:
                S, sec = train_scorer(ctx, sc, cfgs[key]["best"])
                ev = evaluate(ctx, sc, S)
                for k_, v_ in (("hit", ev["hit"]), ("rr", ev["rr"]), ("risk", ev["risk"])):
                    per[key][k_].append(float(v_.mean()))
                per[key]["seconds"].append(sec)
                if seed == MODEL_SEEDS[0]:
                    hit, rsk = risk_curves(ctx, sc, S)
                    curves[f"{year}/{key}/hit"], curves[f"{year}/{key}/risk"] = hit, rsk
        for m, d in per.items():
            res[m][year] = {k: float(np.mean(v)) for k, v in d.items()}
        print(f"year {year}: " + " ".join(f"{m} {per[m]['hit'] and np.mean(per[m]['hit']):.3f}" for m in per), flush=True)
        (ROOT / "results" / "metrics" / "v5_trase.json").write_text(json.dumps(
            {"best": best, "years": EVAL_YEARS, "model_seeds": MODEL_SEEDS, "per_year": res}, indent=2))
        np.savez_compressed(ROOT / "results" / "metrics" / "v5_trase_curves.npz", **curves)
    analyse(res, curves, best, cfgs, w)


def _rule_curves(ctx, qs, rows, tb):
    from suschain.decision_eval.v5_common import BETAS
    hit = np.zeros((len(BETAS), len(qs)))
    rsk = np.zeros_like(hit)
    for j, q in enumerate(qs):
        s = rows[id(q)]
        x = s[q.candidates]
        z = np.zeros_like(s)
        z[q.candidates] = (x - x.mean()) / (x.std() if x.std() > 0 else 1.0)
        for i, b in enumerate(BETAS):
            o = evaluate_substitution([rank_candidates(z - b * ctx.risk, q, tb)], [q], ctx.risk, K)
            hit[i, j], rsk[i, j] = o["hit"][0], o["risk"][0]
    return hit, rsk


def analyse(res, curves, best, cfgs, w):
    out = {"best": best}
    hv = lambda m: [res[m][y]["hit"] for y in EVAL_YEARS]
    t = {"T1 best learned vs co-supply": exact_wilcoxon(hv(best), hv("co_supply")),
         "T2 nbfnet vs co-supply": exact_wilcoxon(hv("nbfnet"), hv("co_supply"))}
    ps = np.array(list(t.values()))
    order = np.argsort(ps)
    holm, run = {}, 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - rank) * ps[i]))
        holm[list(t)[i]] = {"p": float(ps[i]), "p_holm": float(run)}
    out["tests"] = holm
    out["summary"] = {m: {k: float(np.mean([res[m][y][k] for y in EVAL_YEARS])) for k in res[m][EVAL_YEARS[0]]}
                      for m in res if not m.startswith("_")}
    # risk budget for the best ranker and co-supply, alpha from the tuning year's plain risk
    tune_ctx = ctx_for(w, TUNE_YEAR)
    tsc = scenario(w, TUNE_YEAR, 0)
    bud = {}
    for m in ("co_supply", best, "nbfnet"):
        if m == "co_supply":
            r0 = float(np.mean(evaluate_substitution(
                [rank_candidates(np.asarray(tsc.co[q.disrupted].todense()).ravel(), q, tsc.pop + tsc.tb) for q in tsc.qs],
                tsc.qs, tune_ctx.risk, K)["risk"]))
        else:
            S, _ = train_scorer(tune_ctx, tsc, cfgs[m]["best"])
            r0 = float(evaluate(tune_ctx, tsc, S)["risk"].mean())
        bud[m] = {"R0_tuning_year": r0}
        for tgt in (0.10, 0.20, 0.30):
            rows = []
            for y in EVAL_YEARS:
                risk, hit = curves[f"{y}/{m}/risk"], curves[f"{y}/{m}/hit"]
                cust = curves[f"{y}/queries/customer"]
                customers = sorted(set(cust.tolist()))
                Lc = s48.customer_matrix(risk, cust, customers)
                Hc = s48.customer_matrix(hit, cust, customers)
                rows.append(s48.budget_study(Lc, Hc, (1 - tgt) * r0, np.random.default_rng(y)))
            bud[m][f"target{int(tgt * 100)}"] = {
                "per_year_crc_gap": [r["half"]["crc"]["mean_test_risk_minus_alpha"] for r in rows],
                "half": {rule: {k: float(np.mean([r["half"][rule][k] for r in rows])) for k in rows[0]["half"][rule]}
                         for rule in ("grid", "crc")},
                "customers_above_alpha_at_beta_max": float(np.mean([r["customers_above_alpha_at_beta_max"] for r in rows]))}
    out["risk_budget"] = bud
    (ROOT / "results" / "metrics" / "v5_trase_analysis.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"tests": holm, "summary": {m: round(v["hit"], 3) for m, v in out["summary"].items()}}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["tune", "eval"], required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    (tune if a.phase == "tune" else evaluate_years)(a)
