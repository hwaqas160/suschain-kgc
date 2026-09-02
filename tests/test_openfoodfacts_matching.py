import pandas as pd

from suschain.enrichment.openfoodfacts_agribalyse import (
    match_lodhalal_foods,
    match_rate_report,
    normalize_name,
)


def test_normalize_name_case_punct_whitespace():
    assert normalize_name("  Nescafe   Dolce-Gusto!! ") == "nescafe dolce gusto"
    assert normalize_name(None) == ""
    assert normalize_name("") == ""


def _toy_off_lookup() -> pd.DataFrame:
    df = pd.DataFrame({
        "norm_name": ["nescafe original", "kapal api signature"],
        "product_name": ["Nescafe Original", "Kapal Api Signature"],
        "environmental_score_score": [65.0, 40.0],
        "environmental_score_grade": ["b", "d"],
        "carbon-footprint_100g": [1.2, 3.4],
    }).set_index("norm_name")
    return df


def test_match_lodhalal_foods_partial_match():
    food_df = pd.DataFrame({
        "id": [1, 2, 3],
        "nama": ["Nescafe Original", "Totally Unknown Product XYZ", "Kapal Api Signature"],
    })
    off_lookup = _toy_off_lookup()
    matched = match_lodhalal_foods(food_df, off_lookup)

    assert matched.loc[0, "off_matched"] == True  # noqa: E712
    assert matched.loc[0, "environmental_score_score"] == 65.0
    assert matched.loc[1, "off_matched"] == False  # noqa: E712
    assert pd.isna(matched.loc[1, "environmental_score_score"])
    assert matched.loc[2, "off_matched"] == True  # noqa: E712
    assert matched.loc[2, "environmental_score_grade"] == "d"


def test_match_rate_report():
    food_df = pd.DataFrame({"id": [1, 2, 3, 4], "nama": ["a", "b", "c", "d"]})
    off_lookup = pd.DataFrame({
        "norm_name": ["a", "c"],
        "product_name": ["A", "C"],
        "environmental_score_score": [50.0, 60.0],
        "environmental_score_grade": ["c", "c"],
        "carbon-footprint_100g": [1.0, 2.0],
    }).set_index("norm_name")
    matched = match_lodhalal_foods(food_df, off_lookup)
    report = match_rate_report(matched)
    assert report == {"n_total": 4, "n_matched": 2, "match_rate_pct": 50.0}
