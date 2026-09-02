"""Phase 4 (C4/B7): the base paper claims GraphSAGE generalizes "to
previously unseen entities without retraining" but never tests it. This
holds out 10% of LODHalal food entities completely (zero training edges)
and compares an embedding-table model (this project's default everywhere
else, matching the paper's one-hot/categorical features) against a
sentence-transformer text-feature model on the resulting true cold-start
task.

Run: python scripts/13_inductive_evaluation.py
"""
import json
from dataclasses import asdict
from pathlib import Path

import torch

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.inductive_splitters import inductive_entity_holdout
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.enrichment.text_features import encode_series
from suschain.training.train_inductive import InductiveTrainConfig, train_and_eval_inductive

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"


def main():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    edge_type = ("food", "has_manufacturer", "manufacturer")

    print("Building inductive split (holding out 10% of food entities)...")
    split = inductive_entity_holdout(built.data, edge_type, holdout_node_type="food", holdout_fraction=0.1, seed=0)
    print(f"  {len(split.heldout_entity_ids)} held-out food entities, "
          f"{split.heldout_test_edges.shape[1]} cold-start test edges")

    print("Encoding food names with sentence-transformers (cached after first run)...")
    food_text_emb = encode_series(g.food["nama"], cache_key="lodhalal_food_nama_v1")
    text_features = {"food": torch.from_numpy(food_text_emb)}
    print(f"  encoded {food_text_emb.shape}")

    num_nodes_dict = {nt: built.data[nt].num_nodes for nt in built.data.node_types}
    results = {}
    for mode, tf in [("embedding", None), ("text", text_features)]:
        print(f"\n--- feature_mode={mode} ---")
        cfg = InductiveTrainConfig(conv_kind="sage", epochs=100, patience=15, seed=0)
        result = train_and_eval_inductive(split, edge_type, built.data.metadata(), num_nodes_dict, cfg, tf, mode)
        print(f"transductive: {result.transductive_metrics}")
        print(f"cold-start ({result.n_coldstart_edges} edges): {result.coldstart_metrics}")
        results[mode] = result

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = {k: asdict(v) for k, v in results.items()}
    path = RESULTS_DIR / "phase4_inductive_lodhalal.json"
    path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
