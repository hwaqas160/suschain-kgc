"""Generates every data-driven figure used in the LaTeX manuscript
(docs/paper/) directly from results/metrics/*.json and results/tables/*.csv
-- nothing here is hand-drawn or illustrative, every figure is a plot of
the project's actual measured results.

Color palette: the eight-hue categorical order from the dataviz skill's
validated reference palette (light-mode hex values), used in fixed slot
order throughout so a given method/model keeps the same color across
figures. Rankings/magnitudes are single-hue where appropriate; no dual-axis
charts; thin bars with a light recessive grid, matching this project's
print/PDF medium (no hover/interactivity applies here).

Run: python scripts/16_generate_paper_figures.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METRICS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
TABLES_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
FIG_DIR = Path(__file__).resolve().parents[1] / "docs" / "paper" / "figures"

# Fixed categorical slot order (dataviz skill reference palette, light mode).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
)
GRID = "#d8d8d4"
TEXT = "#0b0b0b"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": "#8a8a86", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True, "text.color": TEXT, "axes.labelcolor": TEXT,
    "xtick.color": TEXT, "ytick.color": TEXT,
})


def load(name: str):
    return json.loads((METRICS_DIR / name).read_text())


def fig_benchmark_vs_base_paper(dataset: str, base_paper: dict, outfile: str):
    """Grouped bar: base paper vs. this project, MRR and Hits@10, per model.

    Every bar gets an exact value label -- some (e.g. GAT's MRR) are small
    enough to look like "zero" at this axis scale without one, and a bare
    axis can't distinguish "very small" from "exactly zero."

    The legend is placed as its own row above both subplots (a figure-level
    legend, not an axes-level one anchored to a data corner) so it can never
    collide with a bar or value label -- an axes-anchored "upper left"
    legend collided with the base-paper GraphSAGE label on the AnonNet plot
    specifically because that dataset's tallest bar happens to sit right
    where the legend box would otherwise go; a fixed corner can't adapt to
    every dataset's value distribution, so the legend no longer shares the
    data area at all.
    """
    models = list(base_paper.keys())
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
    legend_handles = None
    for ax, metric, label in zip(axes, ["mrr", "hits@10"], ["MRR", "Hits@10"]):
        x = np.arange(len(models))
        width = 0.32
        base_vals = [base_paper[m][metric] for m in models]
        ours_vals = [base_paper[m][f"ours_{metric}"] for m in models]
        bars_base = ax.bar(x - width / 2, base_vals, width, label="Base paper", color=BLUE)
        bars_ours = ax.bar(x + width / 2, ours_vals, width, label="This work", color=ORANGE)
        if legend_handles is None:
            legend_handles = (bars_base[0], bars_ours[0])
        axmax = max(base_vals + ours_vals)
        ax.set_ylim(0, axmax * 1.20)
        offset = axmax * 0.025
        for bars, vals in ((bars_base, base_vals), (bars_ours, ours_vals)):
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v + offset, f"{v:.3f}",
                         ha="center", va="bottom", fontsize=6.3, color=TEXT)
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.spines[["top", "right"]].set_visible(False)
    fig.legend(legend_handles, ["Base paper", "This work"], loc="upper center",
               ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.07))
    fig.suptitle(f"{dataset}: base paper vs. this work", fontsize=11, y=1.20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / outfile, dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_full_ranking(csv_name: str, dataset: str, outfile: str):
    """Horizontal bar: MRR across every method (GNN/heuristic/KGE), sorted.

    LODHalal's four KGE models collapse to MRR of 0.0002-0.0012 -- real,
    nonzero numbers, but invisible as bars at this scale. Both an exact
    value label per bar and a short on-chart note (pointing to the
    diagnosis in the text) make clear these are a measured, explained
    result and not a missing-data artifact.
    """
    df = pd.read_csv(TABLES_DIR / csv_name).sort_values("mrr_mean")
    color_map = {"GNN": BLUE, "heuristic": AQUA, "KGE": VIOLET}
    colors = [color_map[t] for t in df["type"]]
    fig, ax = plt.subplots(figsize=(7.6, 3.9))
    bars = ax.barh(df["method"], df["mrr_mean"], color=colors)
    axmax = df["mrr_mean"].max()
    ax.set_xlim(0, axmax * 1.24)
    for b, v in zip(bars, df["mrr_mean"]):
        value_label = f"{v:.4f}" if v < 0.01 else f"{v:.3f}"
        ax.text(b.get_width() + axmax * 0.012, b.get_y() + b.get_height() / 2, value_label,
                 ha="left", va="center", fontsize=7, color=TEXT)
    ax.set_xlabel("MRR (filtered, full-catalog ranking)")
    ax.set_title(f"{dataset}: full-catalog link-prediction MRR by method type", fontsize=10.5)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in color_map.values()]
    ax.legend(handles, color_map.keys(), frameon=True, facecolor="white",
              edgecolor="#8a8a86", framealpha=0.95, loc="lower right", fontsize=8)
    if dataset == "LODHalal":
        # Mid-right area: clear of the long RGCN/SAGE bars (top rows) and
        # the legend box (bottom-right corner).
        # Plain text only: LaTeX markup such as ~ or \  does not render in
        # matplotlib and would appear literally in the figure.
        ax.text(0.985, 0.50,
                 "KGE models' MRR $\\approx$ 0 here: type-imbalanced negative\n"
                 "sampling breaks default KGE evaluation (Section 4.1), not a\n"
                 "general KGE failure (cf. RotatE on AnonNet, Figure 4).",
                 transform=ax.transAxes, ha="right", va="center", fontsize=6.3,
                 color=TEXT, style="italic")
    fig.tight_layout()
    fig.savefig(FIG_DIR / outfile, dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_confusion_matrix(dataset: str, model: str, outfile: str):
    """The two confusion matrices are placed side by side as subfigures in the
    manuscript, so they must come out at IDENTICAL pixel dimensions. Saving
    with bbox_inches="tight" would crop each to its own content and yield
    different aspect ratios (LODHalal's class names are much longer than
    AnonNet's), which makes the two panels different heights and forces LaTeX
    to stack them vertically. A fixed figsize with constrained layout keeps the
    canvas identical and fits the content inside it instead.
    """
    d = load(f"phase2_nodeclf_{dataset}.json")
    cm = np.array(d["results"][model]["test_metrics"]["confusion_matrix"])
    labels = d["class_names"]
    fig, ax = plt.subplots(figsize=(4.2, 3.9), layout="constrained")
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(labels, rotation=90, va="center", fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    display_name = "LODHalal" if dataset == "lodhalal" else "AnonNet"
    ax.set_title(f"{display_name}: {model.upper()} confusion matrix", fontsize=10)
    ax.grid(False)
    vmax = cm.max()
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > vmax / 2 else TEXT
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # no bbox_inches="tight": keep both panels dimensionally identical
    fig.savefig(FIG_DIR / outfile, dpi=600)
    plt.close(fig)


def fig_disruption_simulation(outfile: str):
    """Random's recall@5 (0.0021) and popularity's mean_risk@5 (0.0003) are
    genuinely tiny, not zero -- labeled explicitly so they read as measured
    (and, for random, expected-by-design: it has no reason to recover the
    specific removed supplier) rather than as missing bars.
    """
    df = pd.read_csv(TABLES_DIR / "table5_disruption_simulation.csv")
    labels = {"random": "Random", "popularity": "Popularity", "rgcn_plain": "RGCN (plain)",
              "rgcn_sustainability_aware": "RGCN\n(sustainability-aware)"}
    df["label"] = df["method"].map(labels)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    colors = [BLUE, AQUA, ORANGE, GREEN]
    bars0 = axes[0].bar(df["label"], df["recall@5"], color=colors)
    axes[0].set_ylabel("Substitute recall@5")
    axes[0].set_title("Recall of the disrupted supplier")
    bars1 = axes[1].bar(df["label"], df["mean_risk@5"], color=colors)
    axes[1].set_ylabel("Mean risk of recommended set")
    axes[1].set_title("Sustainability risk of top-5 substitutes")
    for ax, bars, vals in ((axes[0], bars0, df["recall@5"]), (axes[1], bars1, df["mean_risk@5"])):
        axmax = vals.max()
        ax.set_ylim(0, axmax * 1.22)
        for b, v in zip(bars, vals):
            value_label = f"{v:.4f}" if v < 0.01 else f"{v:.3f}"
            ax.text(b.get_x() + b.get_width() / 2, v + axmax * 0.025, value_label,
                     ha="center", va="bottom", fontsize=7, color=TEXT)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=7.5)
    fig.suptitle("Disruption simulation (LODHalal, k=5)", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIG_DIR / outfile, dpi=600, bbox_inches="tight")
    plt.close(fig)


def fig_inductive_evaluation(outfile: str):
    df = pd.read_csv(TABLES_DIR / "table6_inductive_evaluation.csv")
    popularity_auc = 0.865  # popularity-only control, docs/PHASE4_INDUCTIVE_RESULTS.md
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    labels = ["Embedding-table\n(GNN)", "Text-feature\n(GNN)", "Popularity\n(heuristic control)"]
    values = [df.loc[df.feature_mode == "embedding", "coldstart_auc"].item(),
              df.loc[df.feature_mode == "text", "coldstart_auc"].item(),
              popularity_auc]
    colors = [BLUE, ORANGE, AQUA]
    bars = ax.bar(labels, values, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Cold-start AUC-ROC")
    ax.set_title("Cold-start link prediction: feature mode vs. a popularity-only control")
    ax.spines[["top", "right"]].set_visible(False)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / outfile, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    lod_gnn = load("phase1_multiseed_lodhalal.json")
    lod_base_paper = {
        "GraphSAGE": {"mrr": 0.0419, "hits@10": 0.3462,
                      "ours_mrr": lod_gnn["comparisons"]["mrr"]["summary"]["sage"]["mean"],
                      "ours_hits@10": lod_gnn["comparisons"]["hits@10"]["summary"]["sage"]["mean"]},
        "GAT": {"mrr": 0.0455, "hits@10": 0.3884,
                "ours_mrr": lod_gnn["comparisons"]["mrr"]["summary"]["gat"]["mean"],
                "ours_hits@10": lod_gnn["comparisons"]["hits@10"]["summary"]["gat"]["mean"]},
        "RGCN": {"mrr": 0.0574, "hits@10": 0.4315,
                 "ours_mrr": lod_gnn["comparisons"]["mrr"]["summary"]["rgcn"]["mean"],
                 "ours_hits@10": lod_gnn["comparisons"]["hits@10"]["summary"]["rgcn"]["mean"]},
    }
    fig_benchmark_vs_base_paper("LODHalal", lod_base_paper, "fig1_lodhalal_vs_base_paper.png")

    anon_gnn = load("phase1_multiseed_anonnet.json")
    anon_base_paper = {
        "GraphSAGE": {"mrr": 0.0436, "hits@10": 0.4000,
                      "ours_mrr": anon_gnn["comparisons"]["mrr"]["summary"]["sage"]["mean"],
                      "ours_hits@10": anon_gnn["comparisons"]["hits@10"]["summary"]["sage"]["mean"]},
        "GAT": {"mrr": 0.0477, "hits@10": 0.4200,
                "ours_mrr": anon_gnn["comparisons"]["mrr"]["summary"]["gat"]["mean"],
                "ours_hits@10": anon_gnn["comparisons"]["hits@10"]["summary"]["gat"]["mean"]},
    }
    fig_benchmark_vs_base_paper("AnonNet", anon_base_paper, "fig2_anonnet_vs_base_paper.png")

    fig_full_ranking("table2_lodhalal_benchmark.csv", "LODHalal", "fig3_lodhalal_full_ranking.png")
    fig_full_ranking("table3_anonnet_benchmark.csv", "AnonNet", "fig4_anonnet_full_ranking.png")

    fig_confusion_matrix("lodhalal", "gat", "fig5_confusion_lodhalal_gat.png")
    fig_confusion_matrix("anonnet", "gat", "fig6_confusion_anonnet_gat.png")

    fig_disruption_simulation("fig7_disruption_simulation.png")
    fig_inductive_evaluation("fig8_inductive_evaluation.png")

    print(f"Saved 8 figures to {FIG_DIR}")
    for f in sorted(FIG_DIR.glob("*.png")):
        print(" ", f.name)


if __name__ == "__main__":
    main()
