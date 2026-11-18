import numpy as np
import torch

from suschain.eval.metrics import best_f1_threshold, binary_link_metrics, majority_class_baseline_accuracy
from suschain.eval.ranking import build_known_positives, filtered_ranking_metrics


def test_filtered_ranking_hand_computed():
    # 1 source node (index 0), 4 destination candidates.
    z_src = torch.tensor([[1.0, 0.0]])
    z_dst = torch.tensor([
        [1.0, 0.0],   # d0: dot = 1  (the queried true edge)
        [0.0, 1.0],   # d1: dot = 0
        [2.0, 0.0],   # d2: dot = 2  (a DIFFERENT known-true edge for src0 -- must be filtered)
        [-1.0, 0.0],  # d3: dot = -1
    ])
    test_edge_index = torch.tensor([[0], [0]])  # (src=0, dst=0)
    known_pos_by_src = {0: {0, 2}}  # src0 has two true tails: d0 (queried) and d2

    mrr, hits = filtered_ranking_metrics(z_src, z_dst, test_edge_index, known_pos_by_src, ks=(1, 3))
    # With d2 filtered out of the candidate pool, d0's score (1) is the best
    # remaining among {d0=1, d1=0, d3=-1} -> rank 1 -> MRR = 1.0, Hits@1 = 1.0
    assert mrr == 1.0
    assert hits[1] == 1.0
    assert hits[3] == 1.0


def test_unfiltered_would_rank_lower_without_masking():
    # Sanity check the *raw* ranking would give rank 2 (d2 outscores d0),
    # confirming the filtering step in the test above is actually doing
    # something, not a no-op.
    z_src = torch.tensor([[1.0, 0.0]])
    z_dst = torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [-1.0, 0.0]])
    test_edge_index = torch.tensor([[0], [0]])
    known_pos_by_src = {0: {0}}  # only the queried edge known -- no filtering happens
    mrr, hits = filtered_ranking_metrics(z_src, z_dst, test_edge_index, known_pos_by_src, ks=(1,))
    assert mrr == 0.5  # rank 2
    assert hits[1] == 0.0


def test_build_known_positives_union():
    e1 = torch.tensor([[0, 1], [5, 6]])
    e2 = torch.tensor([[0], [7]])
    known = build_known_positives(e1, e2)
    assert known[0] == {5, 7}
    assert known[1] == {6}


def test_binary_link_metrics_toy():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.6, 0.4, 0.1])
    m = binary_link_metrics(y_true, y_score, threshold=0.5)
    assert m["accuracy"] == 1.0
    assert m["auc_roc"] == 1.0
    assert m["f1"] == 1.0


def test_best_f1_threshold_finds_separating_point():
    y_true = np.array([1, 1, 0, 0])
    y_score = np.array([0.9, 0.6, 0.55, 0.1])
    t = best_f1_threshold(y_true, y_score)
    from sklearn.metrics import f1_score
    f1 = f1_score(y_true, (y_score >= t).astype(int))
    assert f1 == 1.0  # a perfectly-separating threshold must be found (scores 0.9,0.6 vs 0.55,0.1)


def test_majority_class_baseline():
    y_true = np.array([0, 0, 0, 1])
    assert majority_class_baseline_accuracy(y_true) == 0.75


def test_best_accuracy_threshold_avoids_all_positive_corner():
    import numpy as np
    from suschain.eval.metrics import best_accuracy_threshold, best_f1_threshold
    # heavily tied, weakly informative scorer: most pairs score 0
    y = np.array([1] * 50 + [0] * 50)
    s = np.array([1.0] * 10 + [0.0] * 40 + [1.0] * 2 + [0.0] * 48)
    t_acc = best_accuracy_threshold(y, s)
    acc = ((s >= t_acc).astype(int) == y).mean()
    assert acc == 0.58                          # 10 TP + 48 TN
    t_f1 = best_f1_threshold(y, s)
    assert ((s >= t_f1).astype(int) == y).mean() == 0.5   # F1 picks "all positive"


def test_ranks_with_ties():
    import numpy as np
    from suschain.eval.metrics import ranks_with_ties
    r = ranks_with_ties(np.array([3.0, 1.0, 1.0, 1.0, 0.0]), 1.0)
    assert r == {"optimistic": 2.0, "realistic": 3.0, "pessimistic": 4.0}
