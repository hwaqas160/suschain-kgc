"""Parses AnonNet's free-text `products` column (comma-separated, e.g.
"Oil Pump Part of Vehicle, Gasket Part of Vehicle, Steel") into a coarse
category used by anonnet_sustainability.py's category-intensity component.

Keyword rules designed from a direct sample of 60 random `products` values
(not guessed) -- see docs/PHASE2_SUSTAINABILITY.md. The data is dominated
by automotive parts ("... Part of Vehicle" is a near-universal suffix),
metals/mining, and chemicals, but also includes genuine service-sector
entities (Real Estate Services, Software, Banking, Insurance, Asset
Management) that are not manufacturing at all -- these get their own
"services" category with a low intensity multiplier rather than being
forced into a manufacturing bucket.

Category order matters: text is checked against each category's keywords
in the order below, first match wins, so more specific categories
(automotive, electronics) are checked before generic fallbacks.
"""
from __future__ import annotations

import re

import pandas as pd

# (category_name, [keyword patterns]) -- checked in order, case-insensitive.
# IMPORTANT: "... Part of Vehicle" is a near-universal SUFFIX in this data
# regardless of what the component is actually made of (an electronic
# sensor, a rubber gasket, and a steel bracket can all carry it) --
# verified directly: an earlier ordering with "automotive" first caused
# EVERY electronics-labeled auto part to be miscategorized as "automotive"
# because the generic suffix matched before the specific material keyword
# got a chance (coverage_report() showed zero "electronics" matches despite
# "Electronics Parts Part of Vehicle" being a common raw value). Specific
# material/component categories are therefore checked BEFORE the generic
# "automotive" catch-all, which only fires when nothing more specific
# matched first.
CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("electronics", ["electronic", "semiconductor", "pcb", "optical fiber", "telematics",
                      "ic part", "sensor"]),
    ("energy_equipment", ["battery charger", "solar", "clean energy", "generator", "turbo charger"]),
    ("metals", ["steel", "aluminium", "aluminum", "copper", "molten metal", "tinplate",
                 "niobium", "palladium", "\\bpgm\\b", "billet", "rail", "heavy angle",
                 "galvanized", "iron", "metal mesh", "wire rod"]),
    ("chemicals", ["sulphuric acid", "sulfuric acid", "alumina", "sodium tripolyphosphate",
                    "chemical grade", "\\bacid\\b"]),
    ("mining_minerals", ["coal", "limestone", "uranium", "rutile", "sinter", "coking coal",
                           "\\bore\\b", "mining"]),
    ("cement_minerals", ["cement", "concrete"]),
    ("plastics", ["rubber part", "gasket", "foaming", "plastic"]),
    ("packaging", ["containers & packaging", "packaging"]),
    ("textiles", ["apparel", "textile"]),
    ("food_agriculture", ["beverage", "fertilizer", "food", "agri"]),
    ("pharma_chemicals_consumer", ["household product", "pharma", "cosmetic"]),
    ("forestry_paper", ["forestry", "paper & wood", "\\bwood\\b"]),
    ("services", ["real estate", "software", "banking", "insurance", "asset management",
                   "internet media", "entertainment content", "industrial support services",
                   "wholesale", "leisure", "information system"]),
    # Catch-all: fires only when nothing more specific above matched --
    # see the module-level note on why this must be checked last.
    ("automotive", ["part of vehicle", "motorcycle part", "truck body", "automotive"]),
]

_COMPILED_RULES = [(cat, re.compile("|".join(pats), re.IGNORECASE)) for cat, pats in CATEGORY_RULES]


def classify_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return "other"
    for category, pattern in _COMPILED_RULES:
        if pattern.search(text):
            return category
    return "other"


def classify_products_column(products: pd.Series) -> pd.Series:
    categories = products.apply(classify_text)
    return categories


def coverage_report(categories: pd.Series) -> pd.Series:
    """Category distribution + the "other" (unmatched) rate -- logged
    honestly rather than silently absorbed, per this project's standing
    practice of reporting match/coverage rates (docs/DATA_PROVENANCE.md)."""
    counts = categories.value_counts()
    pct = (counts / len(categories) * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})
