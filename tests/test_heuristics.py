import math

import numpy as np
import torch

from suschain.models.heuristics import HeuristicScorer, build_adjacency, popularity_scores


def _toy_edge_index():
    # undirected: 0-1, 0-2, 1-2, 1-3  (node 4 isolated)
    src = [0, 0, 1, 1]
    dst = [1, 2, 2, 3]
    return torch.tensor([src, dst], dtype=torch.long)


def test_common_neighbors_hand_computed():
    ei = _toy_edge_index()
    A = build_adjacency(ei, num_nodes=5)
    scorer = HeuristicScorer(A)
    row = scorer.common_neighbors_row(0)
    # N(0)={1,2}; CN(0,1)=|{1,2}n{0,2,3}|=|{2}|=1; CN(0,2)=|{1,2}n{0,1}|=|{1}|=1
    # CN(0,3)=|{1,2}n{1}|=|{1}|=1; CN(0,4)=0
    assert row[1] == 1
    assert row[2] == 1
    assert row[3] == 1
    assert row[4] == 0


def test_adamic_adar_hand_computed():
    ei = _toy_edge_index()
    A = build_adjacency(ei, num_nodes=5)
    scorer = HeuristicScorer(A)
    row = scorer.adamic_adar_row(0)
    # AA(0,1): common={2}, deg(2)=2 -> weight 1/ln(2)
    assert math.isclose(row[1], 1.0 / math.log(2), rel_tol=1e-6)
    # AA(0,3): common={1}, deg(1)=3 -> weight 1/ln(3)
    assert math.isclose(row[3], 1.0 / math.log(3), rel_tol=1e-6)


def test_preferential_attachment_hand_computed():
    ei = _toy_edge_index()
    A = build_adjacency(ei, num_nodes=5)
    scorer = HeuristicScorer(A)
    row = scorer.preferential_attachment_row(0)
    # deg(0)=2 (neighbors 1,2), deg(1)=3, deg(2)=2, deg(3)=1, deg(4)=0
    assert row[1] == 2 * 3
    assert row[2] == 2 * 2
    assert row[3] == 2 * 1
    assert row[4] == 0


def test_popularity_scores_is_indegree():
    # directed bipartite-style: food(0,1,2) -> manufacturer(0 or 1)
    ei = torch.tensor([[0, 1, 2], [0, 0, 1]])
    scores = popularity_scores(ei, num_dst_nodes=2)
    assert scores.tolist() == [2.0, 1.0]
