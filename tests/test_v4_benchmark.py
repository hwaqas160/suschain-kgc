import math

import numpy as np
import pytest
import torch

from suschain.eval.ranking_v4 import filtered_ranks, known_from_edges, summarize_ranks
from suschain.models.kge_torch import KGEModel


def test_filtered_ranks_ties_and_filter():
    # source 0: scores over 5 destinations; true dst 2; dst 1 is another known positive
    S = torch.tensor([[0.1, 0.9, 0.5, 0.5, 0.7]])
    q = torch.tensor([[0], [2]])
    known = known_from_edges(torch.tensor([[0, 0], [1, 2]]))
    r = filtered_ranks(lambda ids: S[ids], q, known)
    # after filtering dst 1: greater = {4}, equal others = {3}
    assert r["optimistic"][0] == 2
    assert r["realistic"][0] == 2.5
    assert r["pessimistic"][0] == 3


def test_filtered_ranks_all_tied():
    S = torch.zeros(1, 11)
    r = filtered_ranks(lambda ids: S[ids], torch.tensor([[0], [3]]), {})
    assert r["optimistic"][0] == 1 and r["pessimistic"][0] == 11 and r["realistic"][0] == 6


def test_summarize_ranks():
    s = summarize_ranks(np.array([1, 2, 20]))
    assert s["mrr"] == pytest.approx((1 + 0.5 + 0.05) / 3)
    assert s["hits@1"] == pytest.approx(1 / 3) and s["hits@10"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("kind", ["TransE", "DistMult", "ComplEx", "RotatE"])
def test_kge_score_matches_reference(kind):
    torch.manual_seed(0)
    m = KGEModel(kind, num_entities=7, num_relations=2, dim=6, gamma=5.0)
    h, r, cand = torch.tensor([0, 3]), torch.tensor([1, 0]), torch.arange(7)
    S = m.score(h, r, cand)
    eh, er, E = m.ent(h), m.rel(r), m.ent.weight
    for i in range(2):
        for j in range(7):
            if kind == "DistMult":
                ref = (eh[i] * er[i] * E[j]).sum()
            elif kind == "TransE":
                ref = 5.0 - torch.linalg.norm(eh[i] + er[i] - E[j])
            elif kind == "ComplEx":
                hc = torch.complex(*eh[i].chunk(2))
                rc = torch.complex(*er[i].chunk(2))
                tc = torch.complex(*E[j].chunk(2))
                ref = (hc * rc * tc.conj()).real.sum()
            else:
                hc = torch.complex(*eh[i].chunk(2))
                rot = torch.polar(torch.ones_like(er[i]), er[i])
                tc = torch.complex(*E[j].chunk(2))
                ref = 5.0 - torch.linalg.norm(hc * rot - tc)
            assert S[i, j].item() == pytest.approx(ref.item(), abs=1e-4)
    # score_pairs agrees with score
    t = torch.tensor([4, 1])
    assert torch.allclose(m.score_pairs(h, r, t), S[torch.arange(2), t], atol=1e-5)
