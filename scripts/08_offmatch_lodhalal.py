"""Phase 2: real match-rate measurement for LODHalal <-> Open Food Facts
Eco-Score matching (plan C1, week-1 priority per the plan's own risk note).

Run: python scripts/08_offmatch_lodhalal.py
"""
import json
import time
from pathlib import Path

from suschain.data.lodhalal_loader import load_lodhalal
from suschain.enrichment.openfoodfacts_agribalyse import (
    build_off_lookup,
    match_lodhalal_foods,
    match_rate_report,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"
PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def main():
    t0 = time.perf_counter()
    print("Loading LODHalal food.csv...")
    lod = load_lodhalal()
    print(f"  {len(lod.food)} food entities")

    print("Streaming Open Food Facts export (this scans the full ~1.2GB file once)...")
    off_lookup = build_off_lookup()
    print(f"  {len(off_lookup)} OFF products with a usable environmental_score_score "
          f"(after dedup by normalized name), scan took {time.perf_counter() - t0:.1f}s")

    matched = match_lodhalal_foods(lod.food, off_lookup)
    report = match_rate_report(matched)
    print(f"\nMatch rate: {report['n_matched']}/{report['n_total']} = {report['match_rate_pct']}%")

    # Matched vs unmatched systematic-difference check (plan C1 §4).
    matched_names_sample = matched[matched["off_matched"]]["nama"].head(20).tolist()
    print("\nSample matched product names:")
    for n in matched_names_sample:
        print(f"  {n}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "phase2_offmatch_report.json").write_text(json.dumps(report, indent=2))
    matched.to_parquet(PROCESSED_DIR / "lodhalal_food_off_matched.parquet", index=False)
    print(f"\nSaved: {RESULTS_DIR / 'phase2_offmatch_report.json'}")
    print(f"Saved: {PROCESSED_DIR / 'lodhalal_food_off_matched.parquet'}")
    print(f"Total wall clock: {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
