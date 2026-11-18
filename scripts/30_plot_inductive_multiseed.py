"""Cold-start inductive-evaluation figure from the seven-seed run (scripts/29).

Replaces the single-run bar chart previously produced by
16_generate_paper_figures.py's fig_inductive_evaluation(), which read a
stale pre-multiseed CSV and a hardcoded popularity_auc=0.865 constant, and so
no longer matched Table~\ref{tab:inductive} (mean +/- std over seven seeds,
with Wilcoxon p-values against the popularity control) once that table was
corrected. This script reads the same JSON scripts/29 already wrote and
plots mean +/- std with the paired significance test, matching the style of
scripts/21's disruption-simulation figure.

Run: python -u scripts/30_plot_inductive_multiseed.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "paper" / "figures"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRID, TEXT = "#d8d8d4", "#0b0b0b"
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#8a8a86", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
})

ORDER = ["embedding", "text", "popularity"]
LABELS = ["Embedding-table\n(GNN)", "Text-feature\n(GNN)", "Popularity\n(heuristic control)"]
COLORS = [BLUE, ORANGE, AQUA]


def main():
    d = json.loads((RESULTS_DIR / "phase4_inductive_multiseed_lodhalal.json").read_text())
    s = d["comparison"]["summary"]
    p = d["comparison"]["pairwise"]

    means = [s[m]["mean"] for m in ORDER]
    stds = [s[m]["std"] for m in ORDER]

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    bars = ax.bar(LABELS, means, yerr=stds, color=COLORS, capsize=3,
                  error_kw=dict(ecolor="#5a5a56", elinewidth=1.0))
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cold-start AUC-ROC")
    ax.set_title("Cold-start link prediction: feature mode vs. a popularity-only\n"
                 "control, mean of 7 seeds with standard deviation", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    for b, v, sd in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, v + sd + 0.02, f"{v:.4f}",
                ha="center", va="bottom", fontsize=8.5, color=TEXT)
    ax.text(0.98, 0.04,
            f"embedding vs popularity $p$={p['embedding_vs_popularity']['p_value']:.4f}\n"
            f"text vs popularity $p$={p['text_vs_popularity']['p_value']:.4f}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7.5,
            style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor="#8a8a86", alpha=0.92))
    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig8_inductive_evaluation.{ext}"
        fig.savefig(out, dpi=600, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
