"""Phase 4 (C5): decision-level evaluation via disruption simulation.
Removes 10% of LODHalal food->manufacturer edges (simulated supplier
disruption), then compares four substitute-recommendation methods on
managerial outcomes (plan B13): random, popularity, plain RGCN, and the
Phase-3 sustainability-aware RGCN (lambda=1.0) -- directly testing whether
integrating sustainability into training (Phase 3) produces measurably
better real-world recommendations than the unconstrained baseline, not
just better toy-metric numbers.

Run: python scripts/14_disruption_simulation.py
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from suschain.data.graph_builder import build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.decision_eval.disruption_simulation import (
    evaluate_recommendations,
    recommend_by_scores,
    recommend_random,
    simulate_disruption,
)
from suschain.enrichment.lodhalal_certification import compute_manufacturer_cert_coverage
from suschain.models.heuristics import popularity_scores
from suschain.models.link_predictor import HeteroLinkPredictor
from suschain.training.multi_objective_loss import normalize_risk, sustainability_penalty

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "metrics"
K = 5
DISRUPTION_FRACTION = 0.1
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_model(data, edge_type, rev_edge_type, remaining_edges, risk_norm, sustainability_lambda, seed=0):
    torch.manual_seed(seed)
    src_type, _, dst_type = edge_type
    num_nodes_dict = {nt: data[nt].num_nodes for nt in data.node_types}
    model = HeteroLinkPredictor(
        metadata=data.metadata(), num_nodes_dict=num_nodes_dict,
        embedding_dim=128, hidden_channels=128, out_channels=64, num_layers=2, conv_kind="rgcn",
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)

    node_id_dict = {ntype: data[ntype].node_id.to(DEVICE) for ntype in data.node_types}
    edge_index_dict = {et: data[et].edge_index.to(DEVICE) for et in data.edge_types}
    edge_index_dict[edge_type] = remaining_edges.to(DEVICE)
    if rev_edge_type in edge_index_dict:
        edge_index_dict[rev_edge_type] = remaining_edges.flip(0).to(DEVICE)

    num_dst = data[dst_type].num_nodes
    known_dst_by_src: dict[int, set[int]] = {}
    for s, d in zip(remaining_edges[0].tolist(), remaining_edges[1].tolist()):
        known_dst_by_src.setdefault(s, set()).add(d)

    rng = np.random.default_rng(seed)
    neg_dst = []
    for s in remaining_edges[0].tolist():
        known = known_dst_by_src.get(s, set())
        while True:
            cand = int(rng.integers(0, num_dst))
            if cand not in known:
                neg_dst.append(cand)
                break
    neg_dst = torch.tensor(neg_dst, dtype=torch.long, device=DEVICE)
    pos_edges = remaining_edges.to(DEVICE)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        z_dict = model.encode(node_id_dict, edge_index_dict)
        pos_scores = model.decode(z_dict, pos_edges, src_type, dst_type)
        neg_edges = torch.stack([pos_edges[0], neg_dst])
        neg_scores = model.decode(z_dict, neg_edges, src_type, dst_type)
        scores = torch.cat([pos_scores, neg_scores])
        labels = torch.cat([torch.ones_like(pos_scores), torch.zeros_like(neg_scores)])
        bce = F.binary_cross_entropy_with_logits(scores, labels)
        if sustainability_lambda > 0:
            all_dst = torch.cat([pos_edges[1], neg_dst])
            penalty = sustainability_penalty(scores, all_dst, risk_norm)
            loss = bce + sustainability_lambda * penalty
        else:
            loss = bce
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        z_dict = model.encode(node_id_dict, edge_index_dict)
    return z_dict[src_type], z_dict[dst_type]


def main():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=True)
    edge_type = ("food", "has_manufacturer", "manufacturer")
    rev_edge_type = ("manufacturer", "rev_has_manufacturer", "food")
    num_man = built.data["manufacturer"].num_nodes

    cov_df = compute_manufacturer_cert_coverage(g)
    man_map = built.id_maps["manufacturer"]
    risk_raw = torch.full((num_man,), float("nan"))
    local_idx = cov_df["manufacturer"].map(man_map).to_numpy()
    cert_coverage = cov_df["cert_coverage"].to_numpy()
    risk_raw[local_idx] = torch.tensor(1.0 - cert_coverage, dtype=torch.float32)
    risk_np = risk_raw.numpy()
    fully_certified = np.zeros(num_man, dtype=bool)
    fully_certified[local_idx] = cert_coverage >= 1.0

    orig_edges = built.data[edge_type].edge_index
    scenario = simulate_disruption(orig_edges, fraction=DISRUPTION_FRACTION, seed=0)
    print(f"Disrupted {scenario.disrupted_edges.shape[1]} of {orig_edges.shape[1]} food-manufacturer edges")

    known_dst_by_disrupted_src = {}
    for i, s in enumerate(scenario.disrupted_edges[0].tolist()):
        known_dst_by_disrupted_src[i] = set()  # nothing extra to exclude beyond what's already removed

    results = {}

    # --- Random baseline ---
    n_disrupted = scenario.disrupted_edges.shape[1]
    rec_random = recommend_random(n_disrupted, num_man, K, seed=0)
    results["random"] = evaluate_recommendations(rec_random, scenario.disrupted_edges, risk_np, fully_certified)

    # --- Popularity baseline ---
    pop = popularity_scores(scenario.remaining_edge_index, num_man)
    pop_scores = torch.from_numpy(np.tile(pop, (n_disrupted, 1)))
    rec_pop = recommend_by_scores(pop_scores, K)
    results["popularity"] = evaluate_recommendations(rec_pop, scenario.disrupted_edges, risk_np, fully_certified)

    risk_norm = normalize_risk(risk_raw.to(DEVICE))

    # --- Plain RGCN (lambda=0) ---
    print("\nTraining plain RGCN (lambda=0)...")
    z_food, z_man = train_model(built.data, edge_type, rev_edge_type, scenario.remaining_edge_index, risk_norm, 0.0)
    disrupted_src = scenario.disrupted_edges[0].to(DEVICE)
    scores_plain = (z_food[disrupted_src] @ z_man.T).cpu()
    rec_plain = recommend_by_scores(scores_plain, K)
    results["rgcn_plain"] = evaluate_recommendations(rec_plain, scenario.disrupted_edges, risk_np, fully_certified)

    # --- Sustainability-aware RGCN (lambda=1.0) ---
    print("Training sustainability-aware RGCN (lambda=1.0)...")
    z_food_s, z_man_s = train_model(built.data, edge_type, rev_edge_type, scenario.remaining_edge_index, risk_norm, 1.0)
    scores_sust = (z_food_s[disrupted_src] @ z_man_s.T).cpu()
    rec_sust = recommend_by_scores(scores_sust, K)
    results["rgcn_sustainability_aware"] = evaluate_recommendations(rec_sust, scenario.disrupted_edges, risk_np, fully_certified)

    print("\n=== Disruption simulation results (k=5, 10% of food-manufacturer edges disrupted) ===")
    print(f"{'method':<28}{'recall@5':>10}{'mean_risk@5':>13}{'%certified@5':>14}{'delta_risk':>12}")
    for name, m in results.items():
        print(f"{name:<28}{m['substitute_recall_at_k']:>10.4f}{m['mean_risk_at_k']:>13.4f}"
              f"{m['pct_fully_certified_at_k']:>14.4f}{m['delta_risk_vs_removed']:>12.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "phase4_disruption_simulation_lodhalal.json"
    path.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
