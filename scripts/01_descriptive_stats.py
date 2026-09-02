"""Phase 0 deliverable: descriptive graph-statistics table (Table 1) for both
datasets, plus the data-cleaning log (orphan rows dropped). Run:

    python scripts/01_descriptive_stats.py
"""
from pathlib import Path

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.descriptive_stats import anonnet_stats, lodhalal_stats, to_dataframe
from suschain.data.lodhalal_loader import load_lodhalal

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    lod = load_lodhalal()
    lod_stats = lodhalal_stats(lod)
    lod_df = to_dataframe(lod_stats)
    lod_df.to_csv(RESULTS_DIR / "table1_lodhalal_descriptive_stats.csv", index=False)

    anon = load_anonnet()
    anon_stats = anonnet_stats(anon)
    anon_df = to_dataframe(anon_stats)
    anon_df.to_csv(RESULTS_DIR / "table1_anonnet_descriptive_stats.csv", index=False)

    print("=" * 70)
    print("LODHalal descriptive statistics")
    print("=" * 70)
    print(lod_df.to_string(index=False))
    print(f"\n[data quality] {lod.n_orphan_hascert_dropped} orphan hascert.csv rows dropped "
          f"(food IDs referenced but absent from food.csv)")
    mismatch_pct = 100.0 * lod.n_contains_food_id_mismatch / lod.n_contains_rows_raw
    print(f"[data quality] contains.csv: {lod.n_contains_food_id_mismatch}/{lod.n_contains_rows_raw} "
          f"({mismatch_pct:.1f}%) rows reference a food ID not present in food.csv -- the "
          f"food-ingredient relation is keyed to a near-disjoint ID namespace and is NOT "
          f"reliably usable as shipped. Only {len(lod.contains)} rows survive cleaning. "
          f"See docs/DATA_PROVENANCE.md.")

    print()
    print("=" * 70)
    print("AnonNet descriptive statistics")
    print("=" * 70)
    print(anon_df.to_string(index=False))
    print(f"\n[data quality] {anon.n_orphan_edges_dropped} orphan edges dropped")
    print(f"\n[reproducibility note] Base paper reports 3,813 AnonNet nodes / 106 locations; "
          f"raw data has {anon_stats['n_nodes']} nodes / {anon_stats['n_unique_locations']} "
          f"locations. No documented subsetting procedure exists — see docs/DATA_PROVENANCE.md.")

    print(f"\nSaved: {RESULTS_DIR / 'table1_lodhalal_descriptive_stats.csv'}")
    print(f"Saved: {RESULTS_DIR / 'table1_anonnet_descriptive_stats.csv'}")


if __name__ == "__main__":
    main()
