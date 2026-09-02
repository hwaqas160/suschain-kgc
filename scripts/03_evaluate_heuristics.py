"""Phase 1 (C2): non-learned heuristic floors, using the exact same split/
threshold-tuning/ranking protocol as the GNN training loop so results are
directly comparable to scripts/02_reproduce_baseline.py's output.

Run: python scripts/03_evaluate_heuristics.py
"""
import json
from pathlib import Path

from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import split_link_prediction
from suschain.eval.heuristic_eval import PopularityScorer, evaluate_heuristic
from suschain.eval.ranking import build_known_positives
from suschain.models.heuristics import HeuristicScorer, build_adjacency, popularity_scores

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"


def run_anonnet_heuristics():
    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=True)
    edge_type = ("node", "supplies", "node")
    split = split_link_prediction(built.data, edge_type, rev_edge_type=None, seed=0)

    num_nodes = built.data["node"].num_nodes
    A = build_adjacency(split.train[edge_type].edge_index, num_nodes)
    scorer = HeuristicScorer(A)

    known_pos = build_known_positives(
        split.train[edge_type].edge_index,
        split.val[edge_type].edge_label_index[:, split.val[edge_type].edge_label == 1.0],
        split.test[edge_type].edge_label_index[:, split.test[edge_type].edge_label == 1.0],
    )

    results = {}
    for method in ["common_neighbors", "adamic_adar", "preferential_attachment"]:
        print(f"\n--- AnonNet / heuristic:{method} ---")
        r = evaluate_heuristic(
            scorer, method,
            split.val[edge_type].edge_label_index, split.val[edge_type].edge_label,
            split.test[edge_type].edge_label_index, split.test[edge_type].edge_label,
            known_pos, num_nodes,
        )
        results[method] = r
        print(r)
    return results


def run_lodhalal_popularity():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    edge_type = ("food", "has_manufacturer", "manufacturer")
    rev_edge_type = ("manufacturer", "rev_has_manufacturer", "food")
    split = split_link_prediction(built.data, edge_type, rev_edge_type, seed=0)

    num_manufacturers = built.data["manufacturer"].num_nodes
    pop_vec = popularity_scores(split.train[edge_type].edge_index, num_manufacturers)
    scorer = PopularityScorer(pop_vec)

    known_pos = build_known_positives(
        split.train[edge_type].edge_index,
        split.val[edge_type].edge_label_index[:, split.val[edge_type].edge_label == 1.0],
        split.test[edge_type].edge_label_index[:, split.test[edge_type].edge_label == 1.0],
    )

    print("\n--- LODHalal / heuristic:popularity ---")
    r = evaluate_heuristic(
        scorer, "popularity",
        split.val[edge_type].edge_label_index, split.val[edge_type].edge_label,
        split.test[edge_type].edge_label_index, split.test[edge_type].edge_label,
        known_pos, num_manufacturers,
    )
    print(r)
    return {"popularity": r}


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    anon_results = run_anonnet_heuristics()
    lod_results = run_lodhalal_popularity()

    (RESULTS_DIR / "phase1_heuristics_anonnet.json").write_text(json.dumps(anon_results, indent=2))
    (RESULTS_DIR / "phase1_heuristics_lodhalal.json").write_text(json.dumps(lod_results, indent=2))
    print(f"\nSaved: {RESULTS_DIR / 'phase1_heuristics_anonnet.json'}")
    print(f"Saved: {RESULTS_DIR / 'phase1_heuristics_lodhalal.json'}")


if __name__ == "__main__":
    main()
