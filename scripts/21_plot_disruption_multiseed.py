"""Disruption-simulation figure from the seven-seed run (scripts/20).

Replaces the single-run bar chart: each bar is now a mean over seven seeds
with a standard-deviation error bar, and the two panels carry the paired
Wilcoxon p-value for plain versus sustainability-aware RGCN, so a reader can
see which of the two differences clears seed variance.

Run: python -u scripts/21_plot_disruption_multiseed.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "paper" / "figures"

BLUE, ORANGE, AQUA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#008300"
GRID, TEXT = "#d8d8d4", "#0b0b0b"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#8a8a86", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
})

ORDER = ["random", "popularity", "rgcn_plain", "rgcn_sustainability_aware"]
LABELS = ["Random", "Popularity", "RGCN (plain)", "RGCN\n(sustainability-aware)"]
COLORS = [BLUE, AQUA, ORANGE, GREEN]


def main():
    d = json.loads((RESULTS_DIR / "phase4_disruption_multiseed_lodhalal.json").read_text())
    s, w = d["summary"], d["wilcoxon_plain_vs_sustainability_aware"]

    panels = [
        ("substitute_recall_at_k", "Substitute recall@5",
         "Recall of the disrupted supplier", w["substitute_recall_at_k"]),
        ("mean_risk_at_k", "Mean risk of recommended set",
         "Sustainability risk of top-5 substitutes", w["mean_risk_at_k"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6))
    for ax, (key, ylab, title, pval) in zip(axes, panels):
        means = [s[m][key]["mean"] for m in ORDER]
        stds = [s[m][key]["std"] for m in ORDER]
        bars = ax.bar(LABELS, means, yerr=stds, color=COLORS, capsize=3,
                      error_kw=dict(ecolor="#5a5a56", elinewidth=1.0))
        axmax = max(m + sd for m, sd in zip(means, stds))
        ax.set_ylim(0, axmax * 1.30)
        for b, v, sd in zip(bars, means, stds):
            lab = f"{v:.4f}" if v < 0.01 else f"{v:.3f}"
            ax.text(b.get_x() + b.get_width() / 2, v + sd + axmax * 0.03, lab,
                    ha="center", va="bottom", fontsize=7, color=TEXT)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
        ax.text(0.98, 0.96,
                f"plain vs sustainability-aware\nWilcoxon $p$={pval:.4f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=6.8,
                style="italic",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#8a8a86", alpha=0.92))

    fig.suptitle("Disruption simulation (LODHalal, $k$=5), mean of 7 seeds "
                 "with standard deviation", fontsize=11)
    fig.tight_layout()
    out = FIG_DIR / "fig7_disruption_simulation.png"
    fig.savefig(out, dpi=600, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
