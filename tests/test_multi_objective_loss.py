import math

import torch

from suschain.training.multi_objective_loss import normalize_risk, sustainability_penalty


def test_sustainability_penalty_hand_computed():
    # 2 candidate edges: dst 0 has risk 1.0 (bad), dst 1 has risk 0.0 (safe).
    # Both scored with logit 0 -> sigmoid(0) = 0.5 for each.
    scores = torch.tensor([0.0, 0.0])
    dst_idx = torch.tensor([0, 1])
    risk_by_dst = torch.tensor([1.0, 0.0])
    penalty = sustainability_penalty(scores, dst_idx, risk_by_dst)
    # penalty = mean(0.5*1.0, 0.5*0.0) = mean(0.5, 0.0) = 0.25
    assert math.isclose(penalty.item(), 0.25, rel_tol=1e-6)


def test_sustainability_penalty_zero_when_all_scores_target_safe_nodes():
    scores = torch.tensor([5.0, 5.0])  # high confidence
    dst_idx = torch.tensor([0, 1])
    risk_by_dst = torch.tensor([0.0, 0.0])  # both perfectly safe
    penalty = sustainability_penalty(scores, dst_idx, risk_by_dst)
    assert penalty.item() == 0.0


def test_normalize_risk_min_max():
    raw = torch.tensor([0.0, 5.0, 10.0])
    normalized = normalize_risk(raw)
    assert torch.allclose(normalized, torch.tensor([0.0, 0.5, 1.0]))


def test_normalize_risk_nan_safe():
    raw = torch.tensor([0.0, 10.0, float("nan")])
    normalized = normalize_risk(raw)
    assert not torch.isnan(normalized).any()
    # the NaN entry should be filled with the mean of the valid normalized values (0.0, 1.0) -> 0.5
    assert math.isclose(normalized[2].item(), 0.5, rel_tol=1e-6)
