import torch

from suschain.eval.ranking import sampled_ranking_metrics


def test_sampled_ranking_easy_case_gives_top_rank():
    # true dst (index 0) has by far the highest score against src -> should
    # essentially always rank #1 among any sampled negative pool.
    torch.manual_seed(0)
    z_src = torch.tensor([[10.0, 0.0]])
    z_dst = torch.zeros(50, 2)
    z_dst[0] = torch.tensor([10.0, 0.0])  # true positive: near-perfect match
    z_dst[1:] = torch.randn(49, 2) * 0.01  # everything else near-zero score

    test_edge_index = torch.tensor([[0], [0]])
    known_pos_by_src = {0: {0}}

    mrr, hits = sampled_ranking_metrics(
        z_src, z_dst, test_edge_index, known_pos_by_src, num_negatives=10, ks=(1,), seed=0
    )
    assert mrr == 1.0
    assert hits[1] == 1.0


def test_sampled_ranking_smaller_pool_than_filtered_full_pool():
    """Sanity check the mechanism the Phase-0 hypothesis relies on: the same
    embeddings can rank very differently under a small sampled pool vs. the
    full catalog if there happen to be a few strong distractors elsewhere
    in the full candidate set that a small random sample is likely to miss.
    """
    from suschain.eval.ranking import filtered_ranking_metrics

    torch.manual_seed(0)
    n = 1000
    z_src = torch.tensor([[1.0, 0.0]])
    z_dst = torch.randn(n, 2) * 0.1
    z_dst[0] = torch.tensor([1.0, 0.0])  # true positive, modest score ~1.0
    # plant many strong distractors so full-pool ranking is hard
    z_dst[1:200] = torch.tensor([2.0, 0.0]) + torch.randn(199, 2) * 0.05

    test_edge_index = torch.tensor([[0], [0]])
    known_pos_by_src = {0: {0}}

    mrr_full, hits_full = filtered_ranking_metrics(z_src, z_dst, test_edge_index, known_pos_by_src, ks=(10,))
    mrr_sampled, hits_sampled = sampled_ranking_metrics(
        z_src, z_dst, test_edge_index, known_pos_by_src, num_negatives=10, ks=(10,), seed=1
    )
    # full pool has ~199 items that outscore the true positive -> rank ~200+
    assert mrr_full < 0.05
    # a small 10-negative sample is very unlikely to draw many of the 199
    # strong distractors out of 999 candidates -> true positive usually
    # still ranks near the top
    assert mrr_sampled > mrr_full
