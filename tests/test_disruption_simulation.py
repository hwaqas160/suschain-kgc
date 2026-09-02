import numpy as np
import torch

from suschain.decision_eval.disruption_simulation import (
    evaluate_recommendations,
    recommend_by_scores,
    recommend_random,
    simulate_disruption,
)


def test_simulate_disruption_partitions_edges_correctly():
    edge_index = torch.stack([torch.arange(100), torch.arange(100) % 10])
    scenario = simulate_disruption(edge_index, fraction=0.2, seed=0)
    assert scenario.disrupted_edges.shape[1] == 20
    assert scenario.remaining_edge_index.shape[1] == 80
    # no overlap: same edge shouldn't appear in both
    disrupted_set = set(map(tuple, scenario.disrupted_edges.t().tolist()))
    remaining_set = set(map(tuple, scenario.remaining_edge_index.t().tolist()))
    assert disrupted_set.isdisjoint(remaining_set)


def test_recommend_by_scores_topk_and_exclusion():
    scores = torch.tensor([[0.1, 0.9, 0.5, 0.3]])  # row 0: best is idx1, then idx2, then idx3
    rec = recommend_by_scores(scores, k=2)
    assert list(rec[0]) == [1, 2]

    # exclude idx1 -> next best pair should be [2, 3]
    rec_excl = recommend_by_scores(scores, k=2, exclude_dst={0: {1}})
    assert list(rec_excl[0]) == [2, 3]


def test_evaluate_recommendations_hand_computed():
    # 2 disrupted edges: (src=0, true_dst=1), (src=1, true_dst=3)
    disrupted_edges = torch.tensor([[0, 1], [1, 3]])
    # recommendations (k=2): row0 -> [1, 2] (contains true dst 1); row1 -> [0, 2] (misses true dst 3)
    recommendations = np.array([[1, 2], [0, 2]])
    risk_by_dst = np.array([0.9, 0.1, 0.5, 0.8])  # dst0=0.9(risky) dst1=0.1(safe) dst2=0.5 dst3=0.8(risky)
    fully_certified = np.array([False, True, False, False])

    metrics = evaluate_recommendations(recommendations, disrupted_edges, risk_by_dst, fully_certified)
    assert metrics["k"] == 2
    assert metrics["n_disrupted"] == 2
    assert metrics["substitute_recall_at_k"] == 0.5  # only row 0 found its true dst

    # mean_risk_at_k = mean of risk[[1,2],[0,2]] = mean(0.1,0.5,0.9,0.5) = 0.5
    assert abs(metrics["mean_risk_at_k"] - 0.5) < 1e-9
    # pct_fully_certified = mean of certified[[1,2],[0,2]] = mean(True,False,False,False) = 0.25
    assert abs(metrics["pct_fully_certified_at_k"] - 0.25) < 1e-9
    # delta_risk_vs_removed: mean per-row recommended risk = [(0.1+0.5)/2, (0.9+0.5)/2] = [0.3, 0.7] -> mean 0.5
    # removed_risk = risk[[1,3]] = [0.1, 0.8] -> mean 0.45
    # delta = 0.5 - 0.45 = 0.05
    assert abs(metrics["delta_risk_vs_removed"] - 0.05) < 1e-9


def test_recommend_random_shape_and_validity():
    rec = recommend_random(n_disrupted=5, num_dst=20, k=3, seed=0)
    assert rec.shape == (5, 3)
    assert (rec >= 0).all() and (rec < 20).all()
    for row in rec:
        assert len(set(row)) == 3  # no duplicate candidates within a row
