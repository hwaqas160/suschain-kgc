"""Loads and cleans the raw LODHalal CSVs into typed node/edge tables.

Source: https://github.com/utomogirraz/graph-app-kit (cited by the base paper
as the LODHalal dataset origin). Raw files copied into data/raw/lodhalal/.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from suschain.data.csv_utils import read_two_col_csv_robust

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "lodhalal"


@dataclass
class LODHalalGraph:
    food: pd.DataFrame  # columns: id, nama
    manufacturer: pd.DataFrame  # columns: id, nama_man
    ingredient: pd.DataFrame  # columns: id, name_ing
    cert: pd.DataFrame  # columns: id, number, org
    has_manufacturer: pd.DataFrame  # columns: food, manufacturer
    has_cert: pd.DataFrame  # columns: food, cert
    contains: pd.DataFrame  # columns: food, ingredients
    n_orphan_hascert_dropped: int
    n_contains_rows_raw: int
    n_contains_food_id_mismatch: int


def load_lodhalal(raw_dir: Path = DEFAULT_RAW_DIR) -> LODHalalGraph:
    """Load the seven LODHalal CSVs and clean known FK/integrity bugs.

    1. hascert.csv references food IDs absent from food.csv (verified: 831
       such rows) — dropped and logged.
    2. contains.csv's `food` column is keyed to an almost entirely different
       ID namespace than food.csv/hasmanufacturer.csv/hascert.csv: only
       ~1.4% of its referenced food IDs (218 of 15,739 unique values)
       actually match a real food.csv entity, even though the numeric
       ranges overlap (verified directly). This is not a handful of stray
       orphan rows like the hascert.csv case — it means the food-ingredient
       relation as shipped is not usable to build food->ingredient edges
       for the food entities used elsewhere in the graph. We keep only the
       rows that do resolve, and report the mismatch rate rather than
       silently building a near-empty relation and calling it complete.
       Document this prominently in docs/DATA_PROVENANCE.md.
    """
    food = read_two_col_csv_robust(raw_dir / "food.csv", ("id", "nama"))
    manufacturer = read_two_col_csv_robust(raw_dir / "manufacturer.csv", ("id", "nama_man"))
    ingredient = read_two_col_csv_robust(raw_dir / "ingredient.csv", ("id", "name_ing"))
    cert = pd.read_csv(raw_dir / "cert.csv", engine="python")
    has_manufacturer = pd.read_csv(raw_dir / "hasmanufacturer.csv", engine="python")
    has_cert = pd.read_csv(raw_dir / "hascert.csv", engine="python")
    contains = pd.read_csv(raw_dir / "contains.csv", engine="python")

    valid_food_ids = set(food["id"])
    before = len(has_cert)
    has_cert = has_cert[has_cert["food"].isin(valid_food_ids)].reset_index(drop=True)
    n_orphan_dropped = before - len(has_cert)

    # Defensive: also enforce referential integrity on the other edge tables,
    # in case future raw-data updates introduce similar issues.
    valid_manufacturer_ids = set(manufacturer["id"])
    has_manufacturer = has_manufacturer[
        has_manufacturer["food"].isin(valid_food_ids)
        & has_manufacturer["manufacturer"].isin(valid_manufacturer_ids)
    ].reset_index(drop=True)

    valid_cert_ids = set(cert["id"])
    has_cert = has_cert[has_cert["cert"].isin(valid_cert_ids)].reset_index(drop=True)

    valid_ingredient_ids = set(ingredient["id"])
    n_contains_raw = len(contains)
    n_food_id_mismatch = n_contains_raw - contains["food"].isin(valid_food_ids).sum()
    contains = contains[
        contains["food"].isin(valid_food_ids)
        & contains["ingredients"].isin(valid_ingredient_ids)
    ].reset_index(drop=True)

    return LODHalalGraph(
        food=food,
        manufacturer=manufacturer,
        ingredient=ingredient,
        cert=cert,
        has_manufacturer=has_manufacturer,
        has_cert=has_cert,
        contains=contains,
        n_orphan_hascert_dropped=n_orphan_dropped,
        n_contains_rows_raw=n_contains_raw,
        n_contains_food_id_mismatch=int(n_food_id_mismatch),
    )
