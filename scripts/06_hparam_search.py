"""Phase 1 (C2): per-architecture Optuna hyperparameter search, optimizing
validation-set full-catalog filtered MRR. Prioritizes GAT first on both
datasets (open question from docs/PHASE1_BASELINES.md: is its ranking
anomaly a hyperparameter-sensitivity issue?), then SAGE/RGCN.

Run: python scripts/06_hparam_search.py [--models gat,sage,rgcn] [--trials 25]
"""
import argparse
import json
import logging
from pathlib import Path

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.training.hparam_search import run_search
from suschain.training.train_linkpred import TrainConfig, train_link_prediction

logging.getLogger("optuna").setLevel(logging.WARNING)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
CONFIGS_DIR = Path(__file__).resolve().parents[1] / "configs" / "model"


def search_and_confirm(data, edge_type, rev_edge_type, conv_kind, dataset_name, n_trials):
    print(f"\n=== Optuna search: {dataset_name} / {conv_kind} ({n_trials} trials) ===")
    study = run_search(data, edge_type, rev_edge_type, conv_kind=conv_kind, n_trials=n_trials, seed=0)
    print(f"best_val_mrr={study.best_value:.4f} best_params={study.best_params}")

    # Confirm with a full 100-epoch run + patience 15 at the best config found.
    best_cfg = TrainConfig(conv_kind=conv_kind, epochs=100, patience=15, seed=0, **study.best_params)
    result = train_link_prediction(data, edge_type, rev_edge_type, best_cfg)
    print(f"confirmed full-run test_ranking={result.test_ranking}")
    print(f"confirmed full-run test_metrics={result.test_metrics}")

    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    (CONFIGS_DIR / f"{dataset_name}_{conv_kind}_best.json").write_text(
        json.dumps({"best_params": study.best_params, "best_val_mrr": study.best_value}, indent=2)
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"phase1_hparam_search_{dataset_name}_{conv_kind}.json").write_text(
        json.dumps(
            {
                "best_params": study.best_params,
                "best_val_mrr": study.best_value,
                "confirmed_test_ranking": result.test_ranking,
                "confirmed_test_metrics": result.test_metrics,
                "trials": [{"params": t.params, "value": t.value} for t in study.trials],
            },
            indent=2,
        )
    )
    return study, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="gat")
    parser.add_argument("--trials", type=int, default=25)
    args = parser.parse_args()
    models = args.models.split(",")

    lod_g = load_lodhalal()
    lod_built = build_lodhalal_heterodata(lod_g, make_undirected=True)
    lod_edge_type = ("food", "has_manufacturer", "manufacturer")
    lod_rev_edge_type = ("manufacturer", "rev_has_manufacturer", "food")

    anon_g = load_anonnet()
    anon_built = build_anonnet_heterodata(anon_g, make_undirected=True)
    anon_edge_type = ("node", "supplies", "node")

    for conv_kind in models:
        search_and_confirm(lod_built.data, lod_edge_type, lod_rev_edge_type, conv_kind, "lodhalal", args.trials)
        search_and_confirm(anon_built.data, anon_edge_type, None, conv_kind, "anonnet", args.trials)


if __name__ == "__main__":
    main()
