import pandas as pd

from suschain.enrichment.anonnet_sustainability import (
    CATEGORY_INTENSITY_MULTIPLIER,
    band_scores,
    compute_composite_score,
    normalize_location,
)
from suschain.enrichment.product_taxonomy import classify_text


def test_classify_text_automotive_fallback_when_no_specific_material():
    # "Oil Pump" alone has no more-specific material/component keyword, so
    # the generic "part of vehicle" catch-all fires.
    assert classify_text("Oil Pump Part of Vehicle") == "automotive"


def test_classify_text_specific_material_wins_over_automotive_suffix():
    # "Gasket" (-> plastics) is more specific than the generic "part of
    # vehicle" suffix and must be checked first -- this is the exact bug
    # this project caught and fixed (see product_taxonomy.py module note).
    assert classify_text("Gasket Part of Vehicle") == "plastics"


def test_classify_text_metals():
    assert classify_text("Copper cathode, Copper") == "metals"


def test_classify_text_services():
    assert classify_text("Real Estate Services, Software") == "services"


def test_classify_text_missing_returns_other():
    assert classify_text(None) == "other"
    assert classify_text("") == "other"
    assert classify_text("completely unrecognized gibberish xyz123") == "other"


def test_every_category_has_a_multiplier():
    from suschain.enrichment.product_taxonomy import CATEGORY_RULES
    for cat, _ in CATEGORY_RULES:
        assert cat in CATEGORY_INTENSITY_MULTIPLIER, f"{cat} missing from intensity table"


def test_normalize_location_aliases_and_whitespace():
    assert normalize_location(" China ") == "China"
    assert normalize_location("Brasil") == "Brazil"
    assert normalize_location("Russian Federation") == "Russia"
    assert normalize_location("Germany") == "Germany"


def test_band_scores_tertiles():
    scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
    bands = band_scores(scores)
    assert set(bands.unique()) == {"Low", "Medium", "High"}
    # roughly a third in each band
    counts = bands.value_counts()
    assert counts["Low"] == 3 and counts["High"] == 3


def test_compute_composite_score_end_to_end():
    locations = pd.Series(["Germany", "Japan", "Unknownistan"])
    categories = pd.Series(["metals", "services", "automotive"])
    df = compute_composite_score(locations, categories)
    assert list(df.columns) == ["country_component", "category_component", "composite_score"]
    assert df["category_component"].tolist() == [
        CATEGORY_INTENSITY_MULTIPLIER["metals"],
        CATEGORY_INTENSITY_MULTIPLIER["services"],
        CATEGORY_INTENSITY_MULTIPLIER["automotive"],
    ]
    assert pd.isna(df.loc[2, "country_component"])  # unmatched location -> NaN, not silently zero
