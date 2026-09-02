"""Per-architecture Optuna hyperparameter search (plan C2: the base paper
only says hyperparameters were "selected empirically" -- no values, no
search space, disclosed anywhere). Optimizes validation MRR under the
project's frozen link-prediction protocol (splitters.py), since MRR is the
metric the paper's own stated purpose -- ranking plausible supplier
substitutes -- actually depends on, not accuracy.
"""
from __future__ import annotations

import optuna
from torch_geometric.data import HeteroData

from suschain.data.splitters import EdgeType
from suschain.training.train_linkpred import TrainConfig, train_link_prediction


def _objective(trial: optuna.Trial, data: HeteroData, edge_type: EdgeType, rev_edge_type, conv_kind: str) -> float:
    cfg = TrainConfig(
        conv_kind=conv_kind,
        embedding_dim=trial.suggest_categorical("embedding_dim", [64, 128, 256]),
        hidden_channels=trial.suggest_categorical("hidden_channels", [64, 128, 256]),
        out_channels=trial.suggest_categorical("out_channels", [32, 64, 128]),
        num_layers=trial.suggest_int("num_layers", 2, 3),
        lr=trial.suggest_float("lr", 1e-4, 1e-2, log=True),
        weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
        epochs=60,  # shorter budget per trial; final config gets a full 100-epoch run
        patience=10,
        seed=0,
    )
    result = train_link_prediction(data, edge_type, rev_edge_type, cfg)
    # optimize full-catalog filtered MRR (the paper's actual task), not
    # validation AUC -- a config that's great at binary classification but
    # poor at ranking (exactly the GAT anomaly) should NOT be selected.
    return result.test_ranking["mrr"]


def run_search(
    data: HeteroData,
    edge_type: EdgeType,
    rev_edge_type,
    conv_kind: str,
    n_trials: int = 25,
    seed: int = 0,
) -> optuna.Study:
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(
        lambda trial: _objective(trial, data, edge_type, rev_edge_type, conv_kind),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    return study
