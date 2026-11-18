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


def best_accuracy_threshold(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Validation threshold that maximises accuracy (v4).

    best_f1_threshold() can return 'everything positive' for a weak or heavily
    tied scorer, because labelling every pair positive already scores F1 = 0.667
    when there is one negative per positive. Accuracy has no such corner: the
    all-positive rule scores 0.5. Checks every distinct score as a cut-off."""
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=np.float64)
    cands = np.unique(y_score)
    cands = np.concatenate([cands, [cands[-1] + 1.0]])      # last one: everything negative
    order = np.argsort(y_score)
    s_sorted, t_sorted = y_score[order], y_true[order]
    pos_total = t_sorted.sum()
    # predicted positive = score >= t; negatives below the cut are correct
    idx = np.searchsorted(s_sorted, cands, side="left")
    neg_below = np.concatenate([[0], np.cumsum(1 - t_sorted)])[idx]
    pos_above = pos_total - np.concatenate([[0], np.cumsum(t_sorted)])[idx]
    acc = (neg_below + pos_above) / len(y_true)
    return float(cands[int(np.argmax(acc))])


def ranks_with_ties(row: np.ndarray, true_score: float) -> dict[str, float]:
    """Rank of the true item under the three standard tie policies. `row` must
    already exclude other known positives (set to -inf) and include the true
    item itself. 'realistic' is the expected rank under random tie-breaking."""
    greater = float((row > true_score).sum())
    equal_others = float((row == true_score).sum()) - 1.0
    return {"optimistic": 1.0 + greater,
            "realistic": 1.0 + greater + equal_others / 2.0,
            "pessimistic": 1.0 + greater + equal_others}
