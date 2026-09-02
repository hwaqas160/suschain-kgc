"""Phase 5: consolidates every phase's raw JSON results into clean,
paper-ready CSV tables under results/tables/. Every number here traces
back to a specific script/config/seed in results/metrics/ (per this
project's own verification standard) -- this script only reformats, it
recomputes nothing.

Run: python scripts/15_consolidate_results.py
"""
import json
from pathlib import Path

import pandas as pd

METRICS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
TABLES_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"


def load(name: str):
    return json.loads((METRICS_DIR / name).read_text())


def _gnn_rows_with_full_metrics(multiseed_json: dict) -> list[dict]:
    """Averages accuracy/AUC-ROC/F1/MRR/Hits@10 across seeds -- the exact
    metric set the base paper's Table 4/5 reports, for direct comparison."""
    rows = []
    per_seed = multiseed_json["per_model_per_seed"]
    for model, seeds in per_seed.items():
        n = len(seeds["accuracy"])
        rows.append({
            "method": model.upper(), "type": "GNN",
            "accuracy_mean": round(sum(seeds["accuracy"]) / n, 4),
            "auc_roc_mean": round(sum(seeds["auc_roc"]) / n, 4),
            "f1_mean": round(sum(seeds["f1"]) / n, 4),
            "mrr_mean": round(sum(seeds["mrr"]) / n, 4),
            "mrr_std": round(pd.Series(seeds["mrr"]).std(), 4),
            "hits@10_mean": round(sum(seeds["hits@10"]) / n, 4),
            "hits@10_std": round(pd.Series(seeds["hits@10"]).std(), 4),
        })
    return rows


def table_lodhalal_benchmark():
    """Table 2: LODHalal link prediction -- GNN (multi-seed) + heuristics + KGE.
    Includes Accuracy/AUC-ROC/F1 alongside MRR/Hits@10 -- the same metric
    set as the base paper's Table 4, for direct comparability."""
    rows = _gnn_rows_with_full_metrics(load("phase1_multiseed_lodhalal.json"))
    heur = load("phase1_heuristics_lodhalal.json")
    for method, r in heur.items():
        rows.append({
            "method": method, "type": "heuristic",
            "accuracy_mean": round(r["accuracy"], 4), "auc_roc_mean": round(r["auc_roc"], 4),
            "f1_mean": round(r["f1"], 4),
            "mrr_mean": round(r["mrr"], 4), "mrr_std": None,
            "hits@10_mean": round(r["hits@10"], 4), "hits@10_std": None,
        })
    kge = load("phase1_kge_lodhalal.json")
    for method, r in kge.items():
        rows.append({
            "method": method, "type": "KGE",
            "accuracy_mean": round(r["test_metrics"]["accuracy"], 4),
            "auc_roc_mean": round(r["test_metrics"]["auc_roc"], 4),
            "f1_mean": round(r["test_metrics"]["f1"], 4),
            "mrr_mean": round(r["ranking_metrics"]["mrr"], 4), "mrr_std": None,
            "hits@10_mean": round(r["ranking_metrics"]["hits@10"], 4), "hits@10_std": None,
        })
    df = pd.DataFrame(rows).sort_values("mrr_mean", ascending=False)
    df.to_csv(TABLES_DIR / "table2_lodhalal_benchmark.csv", index=False)
    return df


def table_anonnet_benchmark():
    """Table 3: AnonNet link prediction -- GNN (multi-seed) + heuristics + KGE."""
    rows = _gnn_rows_with_full_metrics(load("phase1_multiseed_anonnet.json"))
    heur = load("phase1_heuristics_anonnet.json")
    for method, r in heur.items():
        rows.append({
            "method": method, "type": "heuristic",
            "accuracy_mean": round(r["accuracy"], 4), "auc_roc_mean": round(r["auc_roc"], 4),
            "f1_mean": round(r["f1"], 4),
            "mrr_mean": round(r["mrr"], 4), "mrr_std": None,
            "hits@10_mean": round(r["hits@10"], 4), "hits@10_std": None,
        })
    kge = load("phase1_kge_anonnet.json")
    for method, r in kge.items():
        rows.append({
            "method": method, "type": "KGE",
            "accuracy_mean": round(r["test_metrics"]["accuracy"], 4),
            "auc_roc_mean": round(r["test_metrics"]["auc_roc"], 4),
            "f1_mean": round(r["test_metrics"]["f1"], 4),
            "mrr_mean": round(r["ranking_metrics"]["mrr"], 4), "mrr_std": None,
            "hits@10_mean": round(r["ranking_metrics"]["hits@10"], 4), "hits@10_std": None,
        })
    df = pd.DataFrame(rows).sort_values("mrr_mean", ascending=False)
    df.to_csv(TABLES_DIR / "table3_anonnet_benchmark.csv", index=False)
    return df


def table_node_classification():
    """Table 4: node classification, both datasets."""
    rows = []
    for dataset in ["anonnet", "lodhalal"]:
        d = load(f"phase2_nodeclf_{dataset}.json")
        for model, r in d["results"].items():
            rows.append({
                "dataset": dataset, "model": model.upper(),
                "accuracy": round(r["test_metrics"]["accuracy"], 4),
                "macro_f1": round(r["test_metrics"]["macro_f1"], 4),
                "majority_baseline": round(r["majority_baseline"], 4),
            })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table4_node_classification.csv", index=False)
    return df


def table_disruption_simulation():
    """Table 5: disruption simulation, decision-level metrics."""
    d = load("phase4_disruption_simulation_lodhalal.json")
    rows = []
    for method, r in d.items():
        rows.append({
            "method": method,
            "recall@5": round(r["substitute_recall_at_k"], 4),
            "mean_risk@5": round(r["mean_risk_at_k"], 4),
            "pct_certified@5": round(r["pct_fully_certified_at_k"], 4),
            "delta_risk_vs_removed": round(r["delta_risk_vs_removed"], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table5_disruption_simulation.csv", index=False)
    return df


def table_inductive_evaluation():
    """Table 6: inductive/cold-start comparison."""
    d = load("phase4_inductive_lodhalal.json")
    rows = []
    for mode, r in d.items():
        rows.append({
            "feature_mode": mode,
            "transductive_auc": round(r["transductive_metrics"]["auc_roc"], 4),
            "coldstart_auc": round(r["coldstart_metrics"]["auc_roc"], 4),
            "coldstart_f1": round(r["coldstart_metrics"]["f1"], 4),
        })
    df = pd.DataFrame(rows)
    df.to_csv(TABLES_DIR / "table6_inductive_evaluation.csv", index=False)
    return df


def main():
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    print("Table 2 -- LODHalal benchmark:")
    print(table_lodhalal_benchmark().to_string(index=False))
    print("\nTable 3 -- AnonNet benchmark:")
    print(table_anonnet_benchmark().to_string(index=False))
    print("\nTable 4 -- Node classification:")
    print(table_node_classification().to_string(index=False))
    print("\nTable 5 -- Disruption simulation:")
    print(table_disruption_simulation().to_string(index=False))
    print("\nTable 6 -- Inductive evaluation:")
    print(table_inductive_evaluation().to_string(index=False))
    print(f"\nAll tables saved to {TABLES_DIR}")


if __name__ == "__main__":
    main()
