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


# ---------------------------------------------------------------------------
# v4 additions
# ---------------------------------------------------------------------------

def exact_wilcoxon(values_a, values_b) -> float:
    """Exact two-sided Wilcoxon signed-rank p-value by full enumeration of
    sign patterns, with average ranks for ties and zero differences dropped.
    Exact for any n up to about 20 (2^n patterns). Its smallest attainable
    p-value is 2 / 2^n: 0.0156 at 7 seeds, 0.00195 at 10."""
    from itertools import product

    d = [float(x) - float(y) for x, y in zip(values_a, values_b) if float(x) != float(y)]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    w_plus = sum(r for r, dv in zip(ranks, d) if dv > 0)
    total = sum(ranks)
    hi, lo = max(w_plus, total - w_plus), min(w_plus, total - w_plus)
    count = 0
    for signs in product((0, 1), repeat=n):
        s = sum(r for r, sg in zip(ranks, signs) if sg)
        if s >= hi - 1e-12 or s <= lo + 1e-12:
            count += 1
    return min(1.0, count / 2 ** n)


def paired_bootstrap_ci(per_query_a, per_query_b, n_boot: int = 10_000,
                        alpha: float = 0.05, seed: int = 0, clusters=None) -> dict:
    """Percentile bootstrap interval for mean(a) - mean(b). a and b must be
    aligned query by query (same scenario, same queries). For several seeds,
    concatenate each seed's aligned arrays first.

    clusters: optional label per query (e.g. "seed:customer"). When given,
    whole clusters are resampled, which keeps correlated queries together;
    otherwise single queries are resampled."""
    a = np.asarray(per_query_a, dtype=np.float64)
    b = np.asarray(per_query_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("per-query arrays must be aligned and of equal length")
    diff = a - b
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    if clusters is None:
        n = len(diff)
        for i in range(n_boot):
            boots[i] = diff[rng.integers(0, n, n)].mean()
        n_units = n
    else:
        labels, inv = np.unique(np.asarray(clusters), return_inverse=True)
        sums = np.bincount(inv, weights=diff)
        counts = np.bincount(inv).astype(np.float64)
        n_units = len(labels)
        for i in range(n_boot):
            pick = rng.integers(0, n_units, n_units)
            boots[i] = sums[pick].sum() / counts[pick].sum()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"mean_diff": float(diff.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "n_queries": int(len(diff)), "n_resampling_units": int(n_units), "n_boot": int(n_boot)}


def holm_correction(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Holm-Bonferroni step-down adjustment for a declared family of tests."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for rank, (name, p) in enumerate(items):
        adj = min(1.0, (m - rank) * p)
        running = max(running, adj)
        out[name] = {"p_raw": float(p), "p_holm": float(running), "reject": bool(running < alpha)}
    return out
