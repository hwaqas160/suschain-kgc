"""Pareto figure from the seven-seed lambda sweep (scripts/18).

Replaces the single-seed scatter: each lambda is now a mean over seven seeds
with standard-deviation error bars on both axes, so a reader can see directly
which movements along the frontier exceed seed variance and which do not.

Run: python scripts/19_plot_pareto_multiseed.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
from adjustText import adjust_text

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "paper" / "figures"

TEXT = "#0b0b0b"
GRID = "#d8d8d4"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#8a8a86", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
})

PANELS = [
    ("phase3_pareto_multiseed_lodhalal_rgcn.json",
     "LODHalal (RGCN)\nrisk = 1 - certification coverage"),
    ("phase3_pareto_multiseed_anonnet_gat.json",
     "AnonNet (GAT)\nrisk = country x category carbon intensity"),
]


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for ax, (fname, title) in zip(axes, PANELS):
        d = json.loads((RESULTS_DIR / fname).read_text())
        lams = d["lambdas"]
        s = d["summary"]
        risk = [s[str(l)]["risk_mean"] for l in lams]
        risk_sd = [s[str(l)]["risk_std"] for l in lams]
        mrr = [s[str(l)]["mrr_mean"] for l in lams]
        mrr_sd = [s[str(l)]["mrr_std"] for l in lams]

        ax.errorbar(risk, mrr, xerr=risk_sd, yerr=mrr_sd, fmt="none",
                    ecolor="#9a9a96", elinewidth=1.0, capsize=2.5, zorder=2)
        ax.plot(risk, mrr, color="gray", alpha=0.35, linestyle="--", zorder=1)
        sc = ax.scatter(risk, mrr, c=lams, cmap="viridis", s=85,
                        zorder=3, edgecolors="white", linewidths=0.6)
        # Points sit close together in some panels, so fixed offsets collide;
        # adjust_text repels the labels apart while keeping a leader line back
        # to the point they belong to.
        texts = [ax.text(r, m, f"$\\lambda$={lam}", fontsize=7)
                 for lam, r, m in zip(lams, risk, mrr)]
        adjust_text(texts, ax=ax, x=risk, y=mrr,
                   arrowprops=dict(arrowstyle="-", color="#8a8a86", lw=0.6),
                   expand=(1.4, 1.6), force_text=(0.6, 0.8))

        w = d["wilcoxon_lambda0_vs_lambda_max"]
        ax.text(0.03, 0.03,
                f"$\\lambda$=0 vs $\\lambda$={lams[-1]} (7 seeds, Wilcoxon):\n"
                f"MRR $p$={w['p_mrr']:.4f}   risk $p$={w['p_risk']:.4f}",
                transform=ax.transAxes, fontsize=7, va="bottom", style="italic",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                          edgecolor="#8a8a86", alpha=0.9))

        ax.set_xlabel("Mean risk of top-10 recommended candidates")
        ax.set_ylabel("Filtered MRR (link-prediction quality)")
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        fig.colorbar(sc, ax=ax, label="$\\lambda$ (sustainability weight)")

    fig.suptitle("Sustainability-aware multi-objective loss: utility vs. risk, "
                 "mean of 7 seeds with standard deviation", fontsize=11.5)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig9_pareto_frontier.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
