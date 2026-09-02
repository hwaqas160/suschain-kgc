"""Matches LODHalal's `food.csv` product names against Open Food Facts'
`environmental_score_score`/`environmental_score_grade` (OFF's renamed
Eco-Score, calculated from the Agribalyse LCA database, ADEME/INRAE) and
`carbon-footprint_100g`, giving a real, non-synthetic sustainability signal
for LODHalal (plan C1). Bulk data: `data/external/off_products.csv.gz`
(~1.2GB, tab-separated despite the .csv name -- verified directly), the
full public product export from static.openfoodfacts.org.

Match-rate honesty (plan §9 risk note: "If <20%, report the match rate as
a finding about real-world data linkage difficulty"): this module reports
the exact match count/rate, never assumes or hides it, and exposes a
matched-vs-unmatched comparison so systematic bias in what does/doesn't
match can be checked (plan C1 §4: "analyse whether matched vs unmatched
products differ systematically").
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

OFF_PATH = Path(__file__).resolve().parents[3] / "data" / "external" / "off_products.csv.gz"
OFF_COLUMNS = [
    "code", "product_name", "brands", "categories_en", "countries_en",
    "environmental_score_score", "environmental_score_grade", "carbon-footprint_100g",
]

_WHITESPACE_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize_name(name: str) -> str:
    if not isinstance(name, str):
        return ""
    s = name.lower().strip()
    s = _PUNCT_RE.sub(" ", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    return s


def build_off_lookup(chunksize: int = 250_000) -> pd.DataFrame:
    """Streams the OFF export once, keeping only rows with a usable
    environmental_score_score (the vast majority of OFF's ~3M products
    don't have one computed -- filtering early keeps this in memory).
    Returns one row per normalized product name (first occurrence wins on
    duplicates, since OFF product names are not unique keys)."""
    reader = pd.read_csv(
        OFF_PATH, sep="\t", usecols=OFF_COLUMNS, chunksize=chunksize,
        low_memory=False, on_bad_lines="skip",
    )
    kept_chunks = []
    for chunk in reader:
        # environmental_score_score is inconsistently typed across the raw
        # export -- most chunks parse it as float64, but some contain
        # quoted numeric strings (e.g. '-1', '69') that make the whole
        # chunk's column object-dtype (verified directly: this broke
        # parquet serialization downstream with an ArrowInvalid error on
        # a chunk containing '-1'). Coerced to numeric explicitly here so
        # every chunk has a consistent float64 dtype before concatenation.
        chunk["environmental_score_score"] = pd.to_numeric(
            chunk["environmental_score_score"], errors="coerce"
        )
        scored = chunk[chunk["environmental_score_score"].notna()].copy()
        if len(scored):
            kept_chunks.append(scored)
    off_scored = pd.concat(kept_chunks, ignore_index=True) if kept_chunks else pd.DataFrame(columns=OFF_COLUMNS)
    off_scored["carbon-footprint_100g"] = pd.to_numeric(off_scored["carbon-footprint_100g"], errors="coerce")
    off_scored["norm_name"] = off_scored["product_name"].apply(normalize_name)
    off_scored = off_scored[off_scored["norm_name"] != ""]
    off_scored = off_scored.drop_duplicates(subset="norm_name", keep="first")
    return off_scored.set_index("norm_name")


def match_lodhalal_foods(food_df: pd.DataFrame, off_lookup: pd.DataFrame) -> pd.DataFrame:
    """food_df: LODHalal food.csv-derived DataFrame with columns id, nama.
    Returns food_df augmented with match columns; unmatched rows keep NaN
    (never silently dropped, so the caller can compute an honest rate)."""
    norm = food_df["nama"].apply(normalize_name)
    matched = norm.isin(off_lookup.index)

    out = food_df.copy()
    out["norm_name"] = norm
    out["off_matched"] = matched
    out["environmental_score_score"] = np.nan
    out["environmental_score_grade"] = None
    out["carbon_footprint_100g"] = np.nan

    matched_rows = out[matched]
    off_rows = off_lookup.loc[matched_rows["norm_name"]]
    out.loc[matched, "environmental_score_score"] = off_rows["environmental_score_score"].to_numpy()
    out.loc[matched, "environmental_score_grade"] = off_rows["environmental_score_grade"].to_numpy()
    out.loc[matched, "carbon_footprint_100g"] = off_rows["carbon-footprint_100g"].to_numpy()
    return out


def match_rate_report(matched_df: pd.DataFrame) -> dict:
    n_total = len(matched_df)
    n_matched = int(matched_df["off_matched"].sum())
    return {
        "n_total": n_total,
        "n_matched": n_matched,
        "match_rate_pct": round(100.0 * n_matched / n_total, 2) if n_total else 0.0,
    }
