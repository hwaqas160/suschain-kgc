"""Re-run LODHalal KGE baselines with type-aware negative sampling and
type-restricted evaluation.

The original run (scripts/04) trains with PyKEEN's default negative sampler,
which corrupts head/tail with any entity in the combined food+manufacturer
vocabulary, and evaluates ranking against that same full vocabulary. On
LODHalal that vocabulary is 43,101 food + 2,520 manufacturer = 45,621
entities sharing one relation, so most corruptions and most ranking
candidates are the wrong entity type by construction.

This script re-runs all four models with PyKEEN's PseudoTypedNegativeSampler
(restricts corruptions to entities that actually occur in that structural
position for the relation) AND reports ranking under this project's own
type-restricted filtered protocol (candidates limited to the 2,520
manufacturers, matching every other row in Table 2). Both the original
(untyped) and corrected (type-aware) numbers are saved side by side for a
direct before/after comparison.

Run: python -u scripts/26_kge_type_aware_lodhalal.py
"""
import json
import time
from dataclasses import asdict
from pathlib import Path

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_link_prediction
from suschain.models.kge import train_kge_baseline

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
KGE_MODELS = ["TransE", "DistMult", "ComplEx", "RotatE"]
EPOCHS = 100


def main():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    edge_type = ("food", "has_manufacturer", "manufacturer")
    rev_edge_type = ("manufacturer", "rev_has_manufacturer", "food")
    split = split_link_prediction(built.data, edge_type, rev_edge_type, seed=0)
    num_food = built.data["food"].num_nodes
    num_man = built.data["manufacturer"].num_nodes

    results = {}
    for name in KGE_MODELS:
        print(f"\n--- LODHalal / KGE:{name} / pseudotyped negative sampling ---", flush=True)
        t0 = time.perf_counter()
        r = train_kge_baseline(
            split, edge_type, num_food, num_man, same_node_type=False,
            model_name=name, epochs=EPOCHS, seed=0, negative_sampler="pseudotyped",
        )
        dt = time.perf_counter() - t0
        print(f"wall_clock={dt:.1f}s")
        print(f"pykeen full-vocab ranking (untyped eval) = {r.ranking_metrics}")
        print(f"type-restricted ranking (2,520 candidates) = {r.ranking_metrics_type_restricted}")
        print(f"test_metrics (1:1 classification) = {r.test_metrics}")
        results[name] = asdict(r)
        results[name]["wall_clock_seconds"] = dt

    out = {
        "negative_sampler": "pseudotyped",
        "note": (
            "ranking_metrics = PyKEEN's own full-vocabulary evaluation (same "
            "protocol as scripts/04, for direct comparison). "
            "ranking_metrics_type_restricted = this project's filtered ranking "
            "restricted to the 2,520 manufacturer candidates, matching every "
            "other row in Table 2."
        ),
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    p = RESULTS_DIR / "phase1_kge_lodhalal_type_aware.json"
    p.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {p}")


if __name__ == "__main__":
    main()
