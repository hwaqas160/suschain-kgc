"""v4 paper figure 1: risk-accuracy trade-off of the three risk mechanisms.

(a) AnonNet substitution, tau 0.1, k 10: GraphSAGE and GATv2, learned penalty
    (lambda 0..5), inference-time shift (beta 0..30), filter (pool 10..100),
    with the non-learned rules as points.
(b) LODHalal disruption, k 5, full graph: the same three mechanisms on the
    RGCN-style model, plus the certificate rule with and without a filter.

Every point is a ten-seed mean read from
  results/metrics/v4_anonnet_substitution_tau0.1.json
  results/metrics/v4_disruption_frontier_lodhalal.json
Output: results/figures/v4_fig_tradeoff.pdf (and .png)
Run:    python scripts/42_v4_paper_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "results" / "metrics"
F = ROOT / "results" / "figures"

C = {"learned": "#1f77b4", "shift": "#ff7f0e", "filter": "#2ca02c", "rule": "#d62728", "other": "#7f7f7f"}


def mean(d, key):
    return d[key]["mean"]


def anonnet_panel(ax):
    s = json.load(open(M / "v4_anonnet_substitution_tau0.1.json"))["summary"]
    lams = [0.0, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0]
    betas = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
    pools = [20, 50, 100]
    for conv, ls, name in (("sage", "-", "GraphSAGE"), ("gat", "--", "GATv2")):
        def pt(k):
            return mean(s[k], "risk@10"), mean(s[k], "hit@10")
        plain = pt(f"gnn_{conv}_lam0")
        xs, ys = zip(*[pt(f"gnn_{conv}_lam{l:g}") for l in lams])
        ax.plot(xs, ys, ls, marker="o", ms=3.5, color=C["learned"], label=f"{name}: learned penalty")
        xs, ys = zip(*([plain] + [pt(f"gnn_{conv}_shift_beta{b:g}") for b in betas]))
        ax.plot(xs, ys, ls, marker="s", ms=3.5, color=C["shift"], label=f"{name}: score shift")
        xs, ys = zip(*([plain] + [pt(f"gnn_{conv}_filter_pool{p}") for p in pools]))
        ax.plot(xs, ys, ls, marker="^", ms=3.5, color=C["filter"], label=f"{name}: filter")
    for k, lab, mk in (("co_supply", "co-supply", "D"), ("popularity", "popularity", "v"),
                       ("fused_gat_lam0", "fusion with GATv2", "*"), ("random", "random", "x")):
        ax.plot(mean(s[k], "risk@10"), mean(s[k], "hit@10"), mk, ms=8 if mk == "*" else 6,
                color=C["rule"] if k in ("co_supply", "fused_gat_lam0") else C["other"],
                linestyle="none", label=lab)
    ax.set_xlabel("mean risk of the 10 recommended suppliers")
    ax.set_ylabel("hit@10 (a true substitute in the top 10)")
    ax.set_title("(a) AnonNet supplier substitution", loc="left", fontsize=10)
    ax.legend(fontsize=6.5, loc="upper left", ncol=2)
    ax.grid(alpha=0.3)


def lodhalal_panel(ax):
    d = json.load(open(M / "v4_disruption_frontier_lodhalal.json"))
    s, r = d["summary"]["full"], d["summary"]["rules"]
    def pt(dd, k):
        return mean(dd[k], "risk"), mean(dd[k], "hit")
    xs, ys = zip(*[pt(s, f"learned_lam{l:g}") for l in d["lambdas"]])
    ax.plot(xs, ys, "-o", ms=3.5, color=C["learned"], label="learned penalty")
    xs, ys = zip(*([pt(s, "learned_lam0")] + [pt(s, f"shift_beta{b:g}") for b in d["betas"]]))
    ax.plot(xs, ys, "-s", ms=3.5, color=C["shift"], label="score shift")
    xs, ys = zip(*([pt(s, "learned_lam0")] + [pt(s, f"filter_pool{p}") for p in d["pools"] if f"filter_pool{p}" in s]))
    ax.plot(xs, ys, "-^", ms=3.5, color=C["filter"], label="filter")
    cf = [k for k in r if k.startswith("cert_filter_pool")]
    xs, ys = zip(*([pt(r, "cert_rule")] + [pt(r, k) for k in sorted(cf, key=lambda k: int(k.split("pool")[1]))]))
    ax.plot(xs, ys, "-D", ms=4, color=C["rule"], label="certificate rule (+ filter)")
    ax.plot(*pt(r, "popularity"), "v", ms=6, color=C["other"], linestyle="none", label="popularity")
    ax.set_xlabel("mean compliance risk of the 5 recommended manufacturers")
    ax.set_ylabel("recall@5 of the lost manufacturer")
    ax.set_title("(b) LODHalal manufacturer recovery", loc="left", fontsize=10)
    ax.legend(fontsize=6.5, loc="center right")
    ax.grid(alpha=0.3)


def main():
    F.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    anonnet_panel(axes[0])
    lodhalal_panel(axes[1])
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(F / f"v4_fig_tradeoff.{ext}", dpi=200)
    print("Saved", F / "v4_fig_tradeoff.pdf")


if __name__ == "__main__":
    main()
