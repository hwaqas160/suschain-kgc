import pandas as pd

from suschain.data.lodhalal_loader import LODHalalGraph
from suschain.enrichment.lodhalal_certification import (
    band_cert_coverage,
    band_cert_coverage_exact,
    compute_manufacturer_cert_coverage,
    majority_class_baseline,
)


def _toy_graph() -> LODHalalGraph:
    # manufacturer 1: 2 foods, both certified -> coverage 1.0
    # manufacturer 2: 4 foods, 1 certified -> coverage 0.25
    # manufacturer 3: 2 foods, 0 certified -> coverage 0.0
    has_manufacturer = pd.DataFrame({
        "food": [1, 2, 3, 4, 5, 6, 7, 8],
        "manufacturer": [1, 1, 2, 2, 2, 2, 3, 3],
    })
    has_cert = pd.DataFrame({"food": [1, 2, 3], "cert": [100, 100, 100]})
    return LODHalalGraph(
        food=pd.DataFrame({"id": range(1, 9), "nama": [f"f{i}" for i in range(1, 9)]}),
        manufacturer=pd.DataFrame({"id": [1, 2, 3], "nama_man": ["m1", "m2", "m3"]}),
        ingredient=pd.DataFrame({"id": [], "name_ing": []}),
        cert=pd.DataFrame({"id": [100], "number": [1], "org": ["MUI"]}),
        has_manufacturer=has_manufacturer,
        has_cert=has_cert,
        contains=pd.DataFrame({"food": [], "ingredients": []}),
        n_orphan_hascert_dropped=0,
        n_contains_rows_raw=0,
        n_contains_food_id_mismatch=0,
    )


def test_compute_manufacturer_cert_coverage_hand_computed():
    g = _toy_graph()
    df = compute_manufacturer_cert_coverage(g).set_index("manufacturer")
    assert df.loc[1, "cert_coverage"] == 1.0
    assert df.loc[2, "cert_coverage"] == 0.25
    assert df.loc[3, "cert_coverage"] == 0.0
    assert df.loc[2, "n_foods"] == 4
    assert df.loc[2, "n_certified_foods"] == 1


def test_band_cert_coverage_and_majority_baseline():
    coverage = pd.Series([0.0, 0.0, 0.0, 0.25, 0.5, 0.75, 1.0, 1.0, 1.0])
    bands = band_cert_coverage(coverage)
    assert set(bands.unique()) <= {"Low", "Medium", "High"}
    baseline = majority_class_baseline(bands)
    assert 0.0 < baseline <= 1.0


def test_band_cert_coverage_exact_matches_real_data_shape():
    # Mirrors the real distribution's skew (verified: 96.2% exactly 1.0,
    # 3.1% exactly 0.0, 0.75% in between) -- tertile banding degenerates
    # here (q1=q2=1.0), which is exactly why band_cert_coverage_exact
    # exists instead of forcing quantile-based bands onto this shape.
    coverage = pd.Series([1.0] * 96 + [0.0] * 3 + [0.5])
    tertile_bands = band_cert_coverage(coverage)
    assert tertile_bands.nunique() == 1  # degenerate: collapses to one class

    exact_bands = band_cert_coverage_exact(coverage)
    assert set(exact_bands.unique()) == {"Fully certified", "Not certified", "Partially certified"}
    counts = exact_bands.value_counts()
    assert counts["Fully certified"] == 96
    assert counts["Not certified"] == 3
    assert counts["Partially certified"] == 1
    baseline = majority_class_baseline(exact_bands)
    assert abs(baseline - 0.96) < 1e-9
