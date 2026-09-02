"""Real, cited carbon-intensity grounding for AnonNet, replacing the base
paper's undocumented "generative process informed by publicly reported
industry-level emissions ranges" (plan C1 -- the paper's central weakness).

Two components, combined into a composite score:

1. Country-level production carbon intensity: Our World in Data's CO2 &
   Greenhouse Gas Emissions dataset (github.com/owid/co2-data, itself built
   from the Global Carbon Project), `co2_per_capita` for the most recent
   year with full coverage. Verified directly: 103/105 (98.1%) of AnonNet's
   unique `location` values match after trivial normalization (whitespace,
   "Brasil"->"Brazil", ISO-name variants like "Russian Federation"->
   "Russia") -- only Cayman Islands and British Virgin Islands are
   genuinely absent from OWID's national accounts (small territories,
   consistent with why they're typically excluded from CO2 accounting
   datasets, not a bug in this matching).

2. Product-category carbon-intensity multiplier: `products` free text is
   parsed into a coarse category (see product_taxonomy.py), and each
   category gets a relative intensity multiplier from a small, explicitly
   cited reference table below (metals/chemicals/machinery rated more
   carbon-intensive than textiles/food, consistent with widely published
   sectoral breakdowns e.g. IEA "Tracking Industry" and IPCC AR6 WG3
   Chapter 11 industrial-sector emissions figures). This is a **documented
   simplification**, not a full multi-region input-output model (EXIOBASE/
   USEEIO) -- flagged explicitly as a limitation/follow-up, not hidden.

composite_score = normalized(country_co2_per_capita) * category_multiplier
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OWID_PATH = Path(__file__).resolve().parents[3] / "data" / "external" / "owid_co2_data.csv"

# Country-name normalization: AnonNet raw value (post strip()) -> OWID name.
# Built and verified directly against the actual raw location values and
# OWID's country list (see docs/PHASE2_SUSTAINABILITY.md for the check).
COUNTRY_ALIASES = {
    "Bosnia And Herzegovina": "Bosnia and Herzegovina",
    "Brasil": "Brazil",
    "Czech Republic": "Czechia",
    'Lao People"s Democratic Republic': "Laos",
    "Libyan Arab Jamahiriya": "Libya",
    "Philipines": "Philippines",
    "Russian Federation": "Russia",
}

# Cited reference: relative sectoral GHG-intensity ranking, not absolute
# factors, sourced from published sectoral emissions breakdowns (IEA
# "Tracking Industry" reports; IPCC AR6 WG3 Ch.11 industry emissions
# shares). Metals/chemicals/cement production are consistently reported as
# the most carbon-intensive manufacturing sectors globally; electronics/
# textiles/food processing are consistently lower. Values are a coarse
# 0.5-2.0 multiplier scale, deliberately not claiming precise per-unit
# factors (that precision requires a full MRIO model -- documented
# limitation, see module docstring).
CATEGORY_INTENSITY_MULTIPLIER = {
    "metals": 2.0,
    "mining_minerals": 1.9,
    "chemicals": 1.9,
    "cement_minerals": 1.9,
    "automotive": 1.4,
    "machinery": 1.3,
    "plastics": 1.3,
    "electronics": 1.1,
    "energy_equipment": 1.2,
    "packaging": 1.0,
    "forestry_paper": 0.9,
    "textiles": 0.8,
    "food_agriculture": 0.7,
    "pharma_chemicals_consumer": 0.9,
    "services": 0.3,
    "other": 1.0,
}


def load_country_intensity(year: int = 2022) -> pd.Series:
    """country name (post-alias-normalization) -> co2_per_capita for `year`."""
    owid = pd.read_csv(OWID_PATH, usecols=["country", "year", "co2_per_capita"])
    row = owid[owid["year"] == year].dropna(subset=["co2_per_capita"])
    return row.set_index("country")["co2_per_capita"]


def normalize_location(raw_location: str) -> str:
    stripped = raw_location.strip()
    return COUNTRY_ALIASES.get(stripped, stripped)


def compute_country_component(locations: pd.Series, year: int = 2022) -> pd.Series:
    """Returns co2_per_capita per row (NaN where unmatched/missing location),
    min-max normalized to [0, 1] over the matched values."""
    intensity = load_country_intensity(year)
    normalized_locs = locations.apply(lambda x: normalize_location(x) if isinstance(x, str) else None)
    raw = normalized_locs.map(intensity)
    if raw.notna().sum() == 0:
        raise ValueError("no AnonNet locations matched OWID country data -- check alias table")
    lo, hi = raw.min(), raw.max()
    return (raw - lo) / (hi - lo) if hi > lo else raw * 0 + 0.5


def compute_category_component(categories: pd.Series) -> pd.Series:
    return categories.map(lambda c: CATEGORY_INTENSITY_MULTIPLIER.get(c, CATEGORY_INTENSITY_MULTIPLIER["other"]))


def compute_composite_score(locations: pd.Series, categories: pd.Series, year: int = 2022) -> pd.DataFrame:
    country_component = compute_country_component(locations, year)
    category_component = compute_category_component(categories)
    composite = country_component * category_component
    return pd.DataFrame({
        "country_component": country_component,
        "category_component": category_component,
        "composite_score": composite,
    })


def band_scores(scores: pd.Series, labels: tuple[str, str, str] = ("Low", "Medium", "High")) -> pd.Series:
    """Quantile-based (tertile) risk bands -- reproducible thresholds
    derived from the score distribution itself, not arbitrary cutoffs
    (plan C1: the base paper's bands were never documented at all)."""
    valid = scores.dropna()
    q1, q2 = valid.quantile([1 / 3, 2 / 3])
    def band(x):
        if pd.isna(x):
            return np.nan
        if x <= q1:
            return labels[0]
        if x <= q2:
            return labels[1]
        return labels[2]
    return scores.apply(band)
