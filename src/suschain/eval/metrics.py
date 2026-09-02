"""Classification-style link-prediction metrics: Accuracy, AUC-ROC, PR-AUC,
F1. PR-AUC is included alongside ROC-AUC specifically because the plan's B2
finding is that ROC-AUC looks strong while F1 collapses under class
imbalance -- PR-AUC is the metric that actually reflects that imbalance.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)


def binary_link_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auc_roc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
    }


def best_f1_threshold(y_true: np.ndarray, y_score: np.ndarray, n_steps: int = 200) -> float:
    """Validation-tuned decision threshold (plan B2/C2: the base paper never
    tunes this, defaulting implicitly to 0.5)."""
    thresholds = np.linspace(y_score.min(), y_score.max(), n_steps)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        f1 = f1_score(y_true, (y_score >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def node_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels=None) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def majority_class_baseline_accuracy(y_true: np.ndarray) -> float:
    values, counts = np.unique(y_true, return_counts=True)
    return float(counts.max() / counts.sum())
