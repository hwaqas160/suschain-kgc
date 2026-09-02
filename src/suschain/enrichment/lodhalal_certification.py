"""Real (non-synthetic) sustainability-adjacent node-classification task for
LODHalal, using the dataset's own genuine certification data -- something
the base paper never attempted (it only ran node classification on
AnonNet's synthetic labels; LODHalal was excluded entirely with the stated
reason "the dataset does not provide suitable node-level attributes for
sustainability-oriented classification").

Given the very low Open Food Facts product-match rate measured directly
(1.42%, 610/43,101 -- see docs/PHASE2_SUSTAINABILITY.md), OFF-derived
Eco-Scores cannot support a robust, representative LODHalal-wide
classification task on their own. Certification coverage is used instead
as the primary real LODHalal signal: it is intrinsic to the dataset (no
external join needed), covers all 2,520 manufacturers completely, and is
a genuine compliance/governance sustainability proxy (the paper's own
stated sustainability scope explicitly includes "compliance or
certification-related attributes", §1).

Label: per-manufacturer certification coverage = (# certified foods it
produces) / (# total foods it produces), banded into tertiles.
"""
from __future__ import annotations

import pandas as pd

from suschain.data.lodhalal_loader import LODHalalGraph


def compute_manufacturer_cert_coverage(g: LODHalalGraph) -> pd.DataFrame:
    """Returns one row per manufacturer: id, n_foods, n_certified_foods,
    cert_coverage in [0, 1]."""
    foods_per_manufacturer = g.has_manufacturer.groupby("manufacturer")["food"].apply(set)

    certified_food_ids = set(g.has_cert["food"].unique())

    rows = []
    for manufacturer_id, food_ids in foods_per_manufacturer.items():
        n_foods = len(food_ids)
        n_certified = len(food_ids & certified_food_ids)
        rows.append({
            "manufacturer": manufacturer_id,
            "n_foods": n_foods,
            "n_certified_foods": n_certified,
            "cert_coverage": n_certified / n_foods if n_foods else 0.0,
        })
    out = pd.DataFrame(rows)
    # manufacturers with zero produced foods after cleaning shouldn't occur
    # (every manufacturer row comes from a real hasmanufacturer edge) but
    # guard explicitly rather than silently producing a NaN coverage.
    assert out["n_foods"].gt(0).all(), "found a manufacturer with 0 foods post-cleaning"
    return out


def band_cert_coverage(coverage: pd.Series, labels: tuple[str, str, str] = ("Low", "Medium", "High")) -> pd.Series:
    """Quantile (tertile) bands -- provided for API symmetry with
    anonnet_sustainability.py::band_scores, but NOT used as this dataset's
    actual task labels. Verified directly on the real data: cert_coverage
    is so heavily skewed (2,423/2,520 manufacturers, 96.2%, are at EXACTLY
    1.0; even the 90th percentile is 1.0) that tertile quantiles collapse
    to duplicate boundaries and this banding degenerates to a single class.
    Use `band_cert_coverage_exact` instead for the real task."""
    q1, q2 = coverage.quantile([1 / 3, 2 / 3])

    def band(x: float) -> str:
        if x <= q1:
            return labels[0]
        if x <= q2:
            return labels[1]
        return labels[2]

    return coverage.apply(band)


def band_cert_coverage_exact(coverage: pd.Series) -> pd.Series:
    """The actual banding used for this task, chosen to match the real
    distribution's shape (verified: 96.2% exactly 1.0, 3.1% exactly 0.0,
    only 0.75% in between) rather than forcing a data-driven quantile split
    that would be degenerate here. "Fully certified" is the overwhelming
    majority class -- this is itself a real, reportable finding about
    LODHalal (a halal-compliance-focused database has overwhelmingly
    compliant manufacturers, with only a small minority of exceptions),
    and it means the operationally meaningful task is detecting the
    minority "Not fully certified" class, not predicting the majority
    (evaluate with macro-F1/precision/recall on the minority class, not
    accuracy alone -- plan B10)."""
    def band(x: float) -> str:
        if x >= 1.0:
            return "Fully certified"
        if x <= 0.0:
            return "Not certified"
        return "Partially certified"

    return coverage.apply(band)


def majority_class_baseline(bands: pd.Series) -> float:
    return float(bands.value_counts(normalize=True).max())
