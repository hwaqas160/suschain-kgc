"""Multi-seed aggregation and pairwise significance testing (plan B4: the
base paper reports a single run per cell, no seeds, no variance, no
significance testing -- "the GAT-vs-GraphSAGE gap on LODHalal (0.6887 vs
0.6658) is well inside plausible seed variance.")

Wilcoxon signed-rank (not a paired t-test) because a handful of seeds is
too small a sample to assume normality of the per-seed metric differences.

Important, verified caveat: the exact two-sided Wilcoxon test's minimum
achievable p-value with n paired seeds is 2/2^n. At n=5 (the plan's
literal spec) that floor is 0.0625 -- p<0.05 is mathematically unreachable
no matter how large or consistent the effect is (confirmed directly in
tests/test_significance.py). This project therefore uses **7 seeds**, not
5, for any comparison meant to support a significance claim (floor at n=7
is 0.0156). Report exact p-values alongside the significance flag, not
just the boolean, so a "not significant at n=5" result isn't
misread as "no effect" when it may just be an underpowered test.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def summarize_metric(per_seed_values: list[float]) -> dict:
    mean, std = mean_std(per_seed_values)
    return {"mean": mean, "std": std, "n_seeds": len(per_seed_values), "values": per_seed_values}


def wilcoxon_pairwise(values_a: list[float], values_b: list[float]) -> dict:
    """Paired Wilcoxon signed-rank test between two models' per-seed values
    on the SAME metric (seeds must be aligned, i.e. values_a[i] and
    values_b[i] come from the same seed)."""
    if len(values_a) != len(values_b):
        raise ValueError("paired test requires equal-length, seed-aligned value lists")
    diffs = np.array(values_a) - np.array(values_b)
    if np.allclose(diffs, 0):
        return {"statistic": 0.0, "p_value": 1.0, "significant_at_0.05": False, "note": "identical values"}
    try:
        stat, p = stats.wilcoxon(values_a, values_b)
    except ValueError as e:
        return {"statistic": None, "p_value": None, "significant_at_0.05": False, "note": str(e)}
    return {"statistic": float(stat), "p_value": float(p), "significant_at_0.05": bool(p < 0.05)}


def compare_models(model_results: dict[str, list[float]]) -> dict:
    """model_results: {model_name: [per-seed metric values]}. Returns
    mean±std per model plus all pairwise Wilcoxon comparisons."""
    summary = {name: summarize_metric(vals) for name, vals in model_results.items()}
    names = list(model_results.keys())
    pairwise = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            pairwise[f"{a}_vs_{b}"] = wilcoxon_pairwise(model_results[a], model_results[b])
    return {"summary": summary, "pairwise": pairwise}
