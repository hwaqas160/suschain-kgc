"""v4 work packages 1 and 2: the LODHalal disruption simulation as full
recall-risk curves, with and without the certificate relation.

On each seed's disruption scenario (identical to scripts/20), it evaluates:
  learned_lam{L}        RGCN trained with the penalty of Eq. 12 at weight L
  learned_neg_lam{L}    the same, penalty on sampled negatives only (ablation)
  shift_beta{B}         plain model (L=0) ranked by logit - B * risk; no retraining
  filter_pool{P}        plain model's top P re-ranked by risk (P=5 is the plain model)
  cert_rule             certificate co-occurrence vote (no learning)
  cert_filter_pool{P}   certificate rule's top P re-ranked by risk
  random, popularity    controls

Graph variants (--graphs): "full" includes certificate links; "no_cert" removes
them, so the model cannot read the certification-compliance risk off the graph.
The cert_rule rows use certificate data directly and are graph-independent;
they are stored once, under "rules".

It also records, for every learned_lam{L} with L > 0, the least-squares slope of
(s_L - s_0) against risk(v) over sampled candidate pairs. The derivation in
docs/v4/penalty_derivation.md predicts a slope near -L.

Outputs (never overwriting earlier files):
  results/metrics/v4_disruption_frontier_lodhalal.json   per-seed means, tests
  results/metrics/v4_disruption_frontier_lodhalal.npz    per-query outcomes

Run:   python -u scripts/34_v4_disruption_frontier.py
Smoke: python -u scripts/34_v4_disruption_frontier.py --smoke
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.decision_eval.disruption_simulation import (
    cert_cooccurrence_scores,
    per_query_outcomes,
    recommend_by_scores,
    recommend_posthoc_filter,
    recommend_random,
    recommend_score_penalty,
    simulate_disruption,
)
from suschain.decision_eval.scenario_trainer import train_scenario_model
from suschain.enrichment.lodhalal_certification import compute_manufacturer_cert_coverage
from suschain.eval.significance import exact_wilcoxon
from suschain.models.heuristics import popularity_scores
from suschain.training.multi_objective_loss import normalize_risk

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
ET = ("food", "has_manufacturer", "manufacturer")
REV = ("manufacturer", "rev_has_manufacturer", "food")
CERT = ("food", "has_cert", "cert")
K = 5
FRACTION = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LAMBDAS = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
NEG_ONLY_LAMBDAS = [1.0, 5.0]
BETAS = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
POOLS = [5, 10, 20, 50, 100]
METRICS = ["hit", "risk", "certified", "delta_risk"]
SLOPE_SAMPLE = 200_000


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0-9")
    ap.add_argument("--graphs", default="full,no_cert")
    ap.add_argument("--smoke", action="store_true",
                    help="seed 0, full graph, lambdas 0 and 1 only; writes *_smoke files")
    a = ap.parse_args()
    if "-" in a.seeds:
        lo, hi = a.seeds.split("-")
        a.seeds = list(range(int(lo), int(hi) + 1))
    else:
        a.seeds = [int(s) for s in a.seeds.split(",")]
    a.graphs = a.graphs.split(",")
    return a


def risk_slope(s_lam, s_0, risk, rng):
    """OLS slope and R^2 of (s_lam - s_0) on risk(v) over sampled pairs."""
    n_q, n_d = s_lam.shape
    qi = rng.integers(0, n_q, SLOPE_SAMPLE)
    di = rng.integers(0, n_d, SLOPE_SAMPLE)
    y = (s_lam[qi, di] - s_0[qi, di]).astype(np.float64)
    x = risk[di].astype(np.float64)
    xc, yc = x - x.mean(), y - y.mean()
    slope = float((xc * yc).sum() / (xc * xc).sum())
    r2 = float(1 - ((yc - slope * xc) ** 2).sum() / (yc * yc).sum())
    return {"slope": slope, "r2": r2, "intercept": float(y.mean() - slope * x.mean())}


def main():
    args = parse_args()
    lambdas, graphs, seeds, tag = LAMBDAS, args.graphs, args.seeds, ""
    neg_lambdas = NEG_ONLY_LAMBDAS
    if args.smoke:
        lambdas, graphs, seeds, neg_lambdas, tag = [0.0, 1.0], ["full"], [0], [], "_smoke"

    g = load_lodhalal()
    built = {gr: build_lodhalal_heterodata(g, make_undirected=True, relations=gr) for gr in graphs}
    ref = build_lodhalal_heterodata(g, make_undirected=True, relations="full")
    num_man = ref.data["manufacturer"].num_nodes
    orig_edges = ref.data[ET].edge_index
    for gr in graphs:   # node indexing must agree across variants
        assert torch.equal(built[gr].data[ET].edge_index, orig_edges)

    cov = compute_manufacturer_cert_coverage(g)
    idx = cov["manufacturer"].map(ref.id_maps["manufacturer"]).to_numpy()
    cert_cov = cov["cert_coverage"].to_numpy()
    risk_raw = torch.full((num_man,), float("nan"))
    risk_raw[idx] = torch.tensor(1.0 - cert_cov, dtype=torch.float32)
    risk_np = risk_raw.numpy()
    fully_certified = np.zeros(num_man, dtype=bool)
    fully_certified[idx] = cert_cov >= 1.0
    risk_norm = normalize_risk(risk_raw.to(DEVICE))
    risk_rank = np.nan_to_num(risk_np, nan=float(np.nanmean(risk_np)))

    certs_of_food, foods_of_cert = defaultdict(set), defaultdict(set)
    ce = ref.data[CERT].edge_index
    for f, c in zip(ce[0].tolist(), ce[1].tolist()):
        certs_of_food[f].add(c)
        foods_of_cert[c].add(f)

    per_query = {}                 # "graph/method/seed/metric" -> array
    per_seed = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # graph->method->metric->[seed means]
    slopes = defaultdict(lambda: defaultdict(list))                         # graph->lambda->[dict]
    timings = []

    def record(graph, method, seed, recs, disrupted):
        pq = per_query_outcomes(recs, disrupted, risk_rank, fully_certified)
        for m in METRICS:
            per_query[f"{graph}/{method}/{seed}/{m}"] = pq[m].astype(np.float32)
            per_seed[graph][method][m].append(float(pq[m].mean()))

    for seed in seeds:
        sc = simulate_disruption(orig_edges, fraction=FRACTION, seed=seed)
        dis = sc.disrupted_edges
        n_dis = dis.shape[1]
        rem = sc.remaining_edge_index
        pop = popularity_scores(rem, num_man)

        # graph-independent rows
        record("rules", "random", seed, recommend_random(n_dis, num_man, K, seed=seed), dis)
        record("rules", "popularity", seed,
               recommend_by_scores(torch.from_numpy(np.tile(pop, (n_dis, 1))), K), dis)
        train_man = dict(zip(rem[0].tolist(), rem[1].tolist()))
        cert_scores = cert_cooccurrence_scores(dis[0].tolist(), certs_of_food, foods_of_cert,
                                               train_man, pop)
        record("rules", "cert_rule", seed, recommend_by_scores(cert_scores, K), dis)
        for P in POOLS[1:]:
            record("rules", f"cert_filter_pool{P}", seed,
                   recommend_posthoc_filter(cert_scores, risk_rank, K, pool_size=P), dis)

        src = dis[0].to(DEVICE)
        for gr in graphs:
            data = built[gr].data
            s0 = None
            runs = [(lam, "all") for lam in lambdas] + [(lam, "negatives") for lam in neg_lambdas]
            for lam, target in runs:
                t0 = time.perf_counter()
                zf, zm = train_scenario_model(data, ET, REV, rem, risk_norm, lam, seed,
                                              device=DEVICE, penalty_on=target)
                scores = (zf[src] @ zm.T).cpu()
                timings.append(time.perf_counter() - t0)
                name = f"learned_lam{lam:g}" if target == "all" else f"learned_neg_lam{lam:g}"
                record(gr, name, seed, recommend_by_scores(scores, K), dis)

                if lam == 0.0 and target == "all":
                    s0 = scores
                    for B in BETAS:
                        record(gr, f"shift_beta{B:g}", seed,
                               recommend_score_penalty(scores, risk_rank, K, B), dis)
                    for P in POOLS[1:]:
                        record(gr, f"filter_pool{P}", seed,
                               recommend_posthoc_filter(scores, risk_rank, K, pool_size=P), dis)
                elif target == "all" and s0 is not None:
                    sl = risk_slope(scores.numpy(), s0.numpy(), risk_rank,
                                    np.random.default_rng(seed))
                    slopes[gr][f"{lam:g}"].append(sl)

            ps = per_seed[gr]
            print(f"seed {seed} [{gr}] plain hit={ps['learned_lam0']['hit'][-1]:.4f} "
                  f"risk={ps['learned_lam0']['risk'][-1]:.5f} | "
                  f"lam1 hit={ps['learned_lam1']['hit'][-1]:.4f} "
                  f"risk={ps['learned_lam1']['risk'][-1]:.5f} | "
                  f"mean train {np.mean(timings):.1f}s", flush=True)
        r = per_seed["rules"]
        print(f"seed {seed} [rules] cert_rule hit={r['cert_rule']['hit'][-1]:.4f} "
              f"risk={r['cert_rule']['risk'][-1]:.5f} | popularity hit="
              f"{r['popularity']['hit'][-1]:.4f}", flush=True)

    # Paired exact Wilcoxon tests against each graph's plain model (exploratory;
    # the declared primary family is tested in scripts/35).
    tests = {}
    for gr in graphs:
        base = per_seed[gr]["learned_lam0"]
        for method, vals in per_seed[gr].items():
            if method == "learned_lam0":
                continue
            tests[f"{gr}/{method}"] = {m: exact_wilcoxon(vals[m], base[m]) for m in METRICS}

    summary = {gr: {meth: {m: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0}
                           for m, v in mv.items()} for meth, mv in d.items()}
               for gr, d in per_seed.items()}
    out = {
        "seeds": seeds, "k": K, "disruption_fraction": FRACTION, "graphs": graphs,
        "lambdas": lambdas, "neg_only_lambdas": neg_lambdas, "betas": BETAS, "pools": POOLS,
        "risk_definition": "1 - certification coverage of the manufacturer (certification-compliance risk)",
        "missing_risk_policy": "unscored manufacturers ranked at the mean risk; none are unscored in LODHalal",
        "per_seed": {gr: {m: dict(v) for m, v in d.items()} for gr, d in per_seed.items()},
        "summary": summary,
        "wilcoxon_vs_plain_exploratory": tests,
        "logit_shift_slopes": {gr: dict(d) for gr, d in slopes.items()},
        "mean_train_seconds": float(np.mean(timings)) if timings else None,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jp = RESULTS_DIR / f"v4_disruption_frontier_lodhalal{tag}.json"
    jp.write_text(json.dumps(out, indent=2))
    np.savez_compressed(RESULTS_DIR / f"v4_disruption_frontier_lodhalal{tag}.npz", **per_query)
    print(f"Saved: {jp}")


if __name__ == "__main__":
    main()
