"""Generates the Pareto frontier figure (link-prediction quality vs. mean
risk of top-10 recommendations) from both lambda sweeps.

Run: python scripts/12_plot_pareto_frontier.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"


def load(name: str):
    data = json.loads((RESULTS_DIR / name).read_text())
    lambdas = [d["sustainability_lambda"] for d in data]
    mrr = [d["test_ranking"]["mrr"] for d in data]
    risk = [d["mean_risk_at_10"] for d in data]
    return lambdas, mrr, risk


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    for ax, (name, title) in zip(
        axes,
        [
            ("phase3_pareto_frontier_lodhalal_rgcn.json", "LODHalal (RGCN)\nrisk = 1 - certification coverage"),
            ("phase3_pareto_frontier_anonnet_gat.json", "AnonNet (GAT)\nrisk = country x category carbon intensity"),
        ],
    ):
        lambdas, mrr, risk = load(name)
        sc = ax.scatter(risk, mrr, c=lambdas, cmap="viridis", s=80, zorder=3)
        ax.plot(risk, mrr, color="gray", alpha=0.4, zorder=2, linestyle="--")
        for lam, r, m in zip(lambdas, risk, mrr):
            ax.annotate(f"$\\lambda$={lam}", (r, m), fontsize=7, xytext=(4, 4), textcoords="offset points")
        ax.set_xlabel("Mean risk of top-10 recommended candidates")
        ax.set_ylabel("Filtered MRR (link-prediction quality)")
        ax.set_title(title, fontsize=10)
        fig.colorbar(sc, ax=ax, label="$\\lambda$ (sustainability weight)")

    fig.suptitle("Sustainability-aware multi-objective loss: utility vs. risk tradeoff", fontsize=12)
    fig.tight_layout()
    out_path = FIGURES_DIR / "phase3_pareto_frontier.png"
    fig.savefig(out_path, dpi=600)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
