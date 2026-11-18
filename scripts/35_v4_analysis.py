"""v4 analysis: decision gate A.

Reads the outputs of scripts 34 (LODHalal frontier) and 36 (AnonNet
substitution) and produces
  - the declared primary tests (docs/v4/primary_tests.md), Holm-corrected,
    each with a customer-clustered bootstrap interval;
  - the learned-vs-shift comparison at equal risk, for every lambda > 0;
  - recall-against-risk figures for both datasets.

Outputs: results/metrics/v4_analysis.json, results/figures/v4_frontier_*.pdf

Run: python -u scripts/35_v4_analysis.py [--conv sage] [--tau 0.1]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from suschain.eval.significance import exact_wilcoxon, holm_correction, paired_bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
MET, FIG = ROOT / "results" / "metrics", ROOT / "results" / "figures"


def anonnet_customers(seeds, tau):
    """Customer of every query, in the order script 36 evaluated them."""
    from suschain.data.anonnet_loader import load_anonnet
    from suschain.data.graph_builder import build_anonnet_heterodata
    from suschain.decision_eval.substitutes import (build_substitution_scenario,
                                                    idf_product_matrix, product_items)
    g = load_anonnet()
    b = build_anonnet_heterodata(g, make_undirected=False)
    nm = b.id_maps["node"]
    order = sorted(nm, key=nm.get)
    X = idf_product_matrix(g.nodes.set_index("id").loc[order, "products_list"].apply(product_items).tolist())
    E = b.data["node", "supplies", "node"].edge_index.numpy()
    return {s: np.array([q.customer for q in build_substitution_scenario(E, X, tau=tau, seed=s).queries])
            for s in seeds}


def interp_at(risk_pts, hit_pts, r):
    o = np.argsort(risk_pts)
    rp, hp = np.asarray(risk_pts)[o], np.asarray(hit_pts)[o]
    if r < rp[0] or r > rp[-1]:
        return None
    return float(np.interp(r, rp, hp))


def shift_curve(ps, conv, betas, metric_h, metric_r, seed_i):
    names = [f"gnn_{conv}_lam0"] + [f"gnn_{conv}_shift_beta{b:g}" for b in betas]
    return ([ps[n][metric_r][seed_i] for n in names], [ps[n][metric_h][seed_i] for n in names])


def boot(npz, a, b, seeds, metric, clusters):
    xa = np.concatenate([npz[f"{a}/{s}/{metric}"] for s in seeds])
    xb = np.concatenate([npz[f"{b}/{s}/{metric}"] for s in seeds])
    cl = np.concatenate([[f"{s}:{c}" for c in clusters[s]] for s in seeds])
    return paired_bootstrap_ci(xa, xb, clusters=cl)


def anonnet(conv, tau, suffix=""):
    stem = MET / f"v4_anonnet_substitution_tau{tau:g}{suffix}"
    d = json.loads(Path(f"{stem}.json").read_text())
    npz = np.load(f"{stem}.npz")
    seeds, ps, betas = d["seeds"], d["per_seed"], d["betas"]
    clusters = anonnet_customers(seeds, tau)
    for s in seeds:   # alignment check against stored per-query arrays
        assert len(clusters[s]) == len(npz[f"random/{s}/hit@10"])

    plain, learned = f"gnn_{conv}_lam0", f"gnn_{conv}_lam1"
    # P3: hit of the shift curve at the learned model's risk, per seed
    shift_hit, learned_hit, out_of_range = [], [], []
    for i, s in enumerate(seeds):
        rr, hh = shift_curve(ps, conv, betas, "hit@10", "risk@10", i)
        h = interp_at(rr, hh, ps[learned]["risk@10"][i])
        if h is None:
            out_of_range.append(s)
            continue
        shift_hit.append(h)
        learned_hit.append(ps[learned]["hit@10"][i])

    raw = {
        "P1_learned_vs_plain_risk": exact_wilcoxon(ps[learned]["risk@10"], ps[plain]["risk@10"]),
        "P2_learned_vs_plain_hit": exact_wilcoxon(ps[learned]["hit@10"], ps[plain]["hit@10"]),
        "P3_learned_vs_shift_at_equal_risk_hit": exact_wilcoxon(learned_hit, shift_hit),
        "P4_filter20_vs_plain_hit": exact_wilcoxon(ps[f"gnn_{conv}_filter_pool20"]["hit@10"], ps[plain]["hit@10"]),
        "P5_fused_vs_co_supply_hit": exact_wilcoxon(ps[f"fused_{conv}_lam0"]["hit@10"], ps["co_supply"]["hit@10"]),
    }
    effects = {
        "P1_learned_vs_plain_risk": boot(npz, learned, plain, seeds, "risk@10", clusters),
        "P2_learned_vs_plain_hit": boot(npz, learned, plain, seeds, "hit@10", clusters),
        "P3_learned_vs_shift_at_equal_risk_hit": {
            "mean_diff": float(np.mean(np.array(learned_hit) - np.array(shift_hit))),
            "n_seeds": len(learned_hit), "seeds_out_of_range": out_of_range,
            "note": "per-seed interpolation; no per-query bootstrap is defined for an interpolated curve"},
        "P4_filter20_vs_plain_hit": boot(npz, f"gnn_{conv}_filter_pool20", plain, seeds, "hit@10", clusters),
        "P5_fused_vs_co_supply_hit": boot(npz, f"fused_{conv}_lam0", "co_supply", seeds, "hit@10", clusters),
    }
    primary = holm_correction(raw)
    for k in primary:
        primary[k]["effect"] = effects[k]

    # exploratory: learned vs shift at equal risk for every lambda > 0
    equal_risk = {}
    for lam in [x for x in d["lambdas"] if x > 0]:
        name = f"gnn_{conv}_lam{lam:g}"
        diffs = []
        for i, s in enumerate(seeds):
            rr, hh = shift_curve(ps, conv, betas, "hit@10", "risk@10", i)
            h = interp_at(rr, hh, ps[name]["risk@10"][i])
            if h is not None:
                diffs.append(ps[name]["hit@10"][i] - h)
        equal_risk[f"{lam:g}"] = {"mean_hit_diff_learned_minus_shift": float(np.mean(diffs)) if diffs else None,
                                  "n_seeds_in_range": len(diffs),
                                  "p_exact_wilcoxon": exact_wilcoxon(diffs, [0.0] * len(diffs)) if diffs else None}

    # figure
    S = d["summary"]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    def curve(names, label, style):
        xs = [S[n]["risk@10"]["mean"] for n in names]
        ys = [S[n]["hit@10"]["mean"] for n in names]
        ax.plot(xs, ys, style, label=label, ms=4)
    curve([f"gnn_{conv}_lam{x:g}" for x in d["lambdas"]], "learned penalty (λ)", "o-")
    curve([plain] + [f"gnn_{conv}_shift_beta{b:g}" for b in betas], "inference-time shift (β)", "s--")
    curve([plain] + [f"gnn_{conv}_filter_pool{p}" for p in d["pools"]], "post-hoc filter (pool)", "^:")
    for n, mk in [("co_supply", "D"), ("popularity", "v"), (f"fused_{conv}_lam0", "*"), ("random", "x")]:
        ax.plot(S[n]["risk@10"]["mean"], S[n]["hit@10"]["mean"], mk, ms=8, label=n.replace("_", " "))
    ax.set_xlabel("mean risk of the 10 recommended suppliers (normalised)")
    ax.set_ylabel("substitute hit@10")
    ax.set_title(f"AnonNet substitution ({conv}, τ={tau:g}, {len(seeds)} seeds)")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(alpha=0.3)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / f"v4_frontier_anonnet_{conv}.pdf")
    plt.close(fig)

    table = {m: {"hit@10": S[m]["hit@10"], "hit@5": S[m]["hit@5"], "rr": S[m]["rr"],
                 "risk@10": S[m]["risk@10"], "unscored@10": S[m]["unscored@10"]} for m in S}
    return {"primary_tests_holm": primary, "learned_vs_shift_equal_risk_exploratory": equal_risk,
            "summary_table": table, "risk_coverage": d["risk_coverage"], "scenario_stats": d["scenario_stats"]}


def lodhalal(suffix=""):
    p = MET / f"v4_disruption_frontier_lodhalal{suffix}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    S, ps = d["summary"], d["per_seed"]
    out = {}
    for gr in d["graphs"]:
        rows = {m: {k: S[gr][m][k] for k in ("hit", "risk", "certified")} for m in S[gr]}
        equal_risk = {}
        for lam in [x for x in d["lambdas"] if x > 0]:
            diffs = []
            for i in range(len(d["seeds"])):
                names = ["learned_lam0"] + [f"shift_beta{b:g}" for b in d["betas"]]
                h = interp_at([ps[gr][n]["risk"][i] for n in names], [ps[gr][n]["hit"][i] for n in names],
                              ps[gr][f"learned_lam{lam:g}"]["risk"][i])
                if h is not None:
                    diffs.append(ps[gr][f"learned_lam{lam:g}"]["hit"][i] - h)
            equal_risk[f"{lam:g}"] = {"mean_hit_diff_learned_minus_shift": float(np.mean(diffs)) if diffs else None,
                                      "n_seeds_in_range": len(diffs),
                                      "p_exact_wilcoxon": exact_wilcoxon(diffs, [0.0] * len(diffs)) if diffs else None}
        out[gr] = {"summary": rows, "learned_vs_shift_equal_risk_exploratory": equal_risk,
                   "logit_shift_slopes": d["logit_shift_slopes"].get(gr)}
    out["rules"] = {m: {k: S["rules"][m][k] for k in ("hit", "risk", "certified")} for m in S["rules"]}

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    full = S.get("full")
    if full:
        def curve(names, label, style):
            ax.plot([full[n]["risk"]["mean"] for n in names], [full[n]["hit"]["mean"] for n in names],
                    style, label=label, ms=4)
        curve([f"learned_lam{x:g}" for x in d["lambdas"]], "RGCN learned penalty (λ)", "o-")
        curve(["learned_lam0"] + [f"shift_beta{b:g}" for b in d["betas"]], "RGCN inference-time shift (β)", "s--")
        curve(["learned_lam0"] + [f"filter_pool{p}" for p in d["pools"][1:]], "RGCN post-hoc filter (pool)", "^:")
    R = S["rules"]
    ax.plot([R["cert_rule"]["risk"]["mean"]] + [R[f"cert_filter_pool{p}"]["risk"]["mean"] for p in d["pools"][1:]],
            [R["cert_rule"]["hit"]["mean"]] + [R[f"cert_filter_pool{p}"]["hit"]["mean"] for p in d["pools"][1:]],
            "D-", label="certificate rule + filter (pool)")
    ax.plot(R["popularity"]["risk"]["mean"], R["popularity"]["hit"]["mean"], "v", ms=8, label="popularity")
    ax.set_xlabel("mean certification-compliance risk of the 5 recommendations")
    ax.set_ylabel("recovery of the disrupted manufacturer (recall@5)")
    ax.set_title(f"LODHalal disruption, full graph ({len(d['seeds'])} seeds)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "v4_frontier_lodhalal.pdf")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conv", default="sage")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--suffix", default="", help="e.g. _smoke, to test on smoke outputs")
    a = ap.parse_args()
    res = {"declared_family": "docs/v4/primary_tests.md",
           "anonnet": anonnet(a.conv, a.tau, a.suffix), "lodhalal": lodhalal(a.suffix)}
    p = MET / f"v4_analysis{a.suffix}.json"
    p.write_text(json.dumps(res, indent=2))
    print(json.dumps(res["anonnet"]["primary_tests_holm"], indent=1))
    print(f"Saved: {p}")


if __name__ == "__main__":
    main()
