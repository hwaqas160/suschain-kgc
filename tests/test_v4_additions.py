import numpy as np
import pytest
import torch
from scipy import stats

from suschain.decision_eval.disruption_simulation import (
    cert_cooccurrence_scores,
    evaluate_recommendations,
    per_query_outcomes,
    recommend_by_scores,
    recommend_score_penalty,
)
from suschain.eval.significance import exact_wilcoxon, holm_correction, paired_bootstrap_ci


# --- statistics -----------------------------------------------------------

def test_exact_wilcoxon_matches_scipy_exact():
    rng = np.random.default_rng(1)
    for _ in range(20):
        a, b = rng.normal(size=10), rng.normal(size=10)
        ref = stats.wilcoxon(a, b, method="exact").pvalue
        assert exact_wilcoxon(a, b) == pytest.approx(ref, abs=1e-9)


def test_exact_wilcoxon_floor():
    a = np.arange(10) + 1.0
    assert exact_wilcoxon(a, np.zeros(10)) == pytest.approx(2 / 2 ** 10)
    assert exact_wilcoxon(a[:7], np.zeros(7)) == pytest.approx(2 / 2 ** 7)
    assert exact_wilcoxon(a, a) == 1.0


def test_paired_bootstrap_ci_brackets_true_difference():
    rng = np.random.default_rng(0)
    b = rng.normal(size=2000)
    a = b + 0.3 + rng.normal(scale=0.1, size=2000)
    r = paired_bootstrap_ci(a, b, n_boot=2000)
    assert r["ci_low"] < 0.3 < r["ci_high"]
    with pytest.raises(ValueError):
        paired_bootstrap_ci(a, b[:-1])


def test_holm_correction_step_down():
    out = holm_correction({"x": 0.01, "y": 0.04, "z": 0.03})
    assert out["x"]["p_holm"] == pytest.approx(0.03)
    assert out["z"]["p_holm"] == pytest.approx(0.06)
    assert out["y"]["p_holm"] == pytest.approx(0.06)   # monotone: never below z's
    assert out["x"]["reject"] and not out["y"]["reject"]


# --- recommenders -----------------------------------------------------------

def test_score_penalty_beta_zero_equals_plain_ranking():
    scores = torch.randn(30, 12)
    risk = np.linspace(0, 1, 12)
    assert np.array_equal(recommend_score_penalty(scores, risk, k=4, beta=0.0),
                          recommend_by_scores(scores, k=4))


def test_score_penalty_large_beta_prefers_low_risk():
    scores = torch.randn(30, 12)
    risk = np.linspace(0, 1, 12)
    recs = recommend_score_penalty(scores, risk, k=3, beta=1e6)
    assert (np.sort(recs, axis=1) == np.array([0, 1, 2])).all()


def test_per_query_outcomes_agree_with_aggregate():
    rng = np.random.default_rng(3)
    recs = rng.integers(0, 20, size=(50, 5))
    disrupted = torch.stack([torch.arange(50), torch.from_numpy(rng.integers(0, 20, size=50))])
    risk = rng.random(20)
    cert = rng.random(20) > 0.5
    agg = evaluate_recommendations(recs, disrupted, risk, cert)
    pq = per_query_outcomes(recs, disrupted, risk, cert)
    assert pq["hit"].mean() == pytest.approx(agg["substitute_recall_at_k"])
    assert pq["risk"].mean() == pytest.approx(agg["mean_risk_at_k"])
    assert pq["certified"].mean() == pytest.approx(agg["pct_fully_certified_at_k"])
    assert pq["delta_risk"].mean() == pytest.approx(agg["delta_risk_vs_removed"])


def test_cert_cooccurrence_votes_and_excludes_self():
    # food 0 and food 1 share cert 7; food 1's training manufacturer is 2.
    scores = cert_cooccurrence_scores(
        query_src=[0],
        certs_of_src={0: {7}, 1: {7}},
        srcs_of_cert={7: {0, 1}},
        train_dst_of_src={0: 4, 1: 2},
        popularity=np.array([5.0, 0, 0, 0, 1.0]),
    )
    assert int(scores[0].argmax()) == 2      # the vote, not popularity, decides
    assert scores[0, 4] < 1.0                # the query's own manufacturer gets no vote


# --- graph presets ----------------------------------------------------------

def test_lodhalal_relation_presets():
    from suschain.data.graph_builder import build_lodhalal_heterodata
    from suschain.data.lodhalal_loader import load_lodhalal

    g = load_lodhalal()
    full = build_lodhalal_heterodata(g).data
    no_cert = build_lodhalal_heterodata(g, relations="no_cert").data
    fm = build_lodhalal_heterodata(g, relations="food_manufacturer_only").data

    assert set(full.node_types) == {"food", "manufacturer", "ingredient", "cert"}
    assert set(no_cert.node_types) == {"food", "manufacturer", "ingredient"}
    assert set(fm.node_types) == {"food", "manufacturer"}
    et = ("food", "has_manufacturer", "manufacturer")
    for d in (no_cert, fm):
        assert torch.equal(d[et].edge_index, full[et].edge_index)
        assert not any("cert" in e[1] for e in d.edge_types)
    with pytest.raises(ValueError):
        build_lodhalal_heterodata(g, relations="nonsense")


def test_cluster_bootstrap_is_wider_for_correlated_queries():
    rng = np.random.default_rng(0)
    cl = np.repeat(np.arange(50), 40)                 # 50 clusters of 40 queries
    shift = rng.normal(size=50)[cl]                   # strong within-cluster correlation
    b = rng.normal(size=2000) * 0.1
    a = b + 0.2 + shift
    plain = paired_bootstrap_ci(a, b, n_boot=2000)
    clustered = paired_bootstrap_ci(a, b, n_boot=2000, clusters=cl)
    assert (clustered["ci_high"] - clustered["ci_low"]) > 2 * (plain["ci_high"] - plain["ci_low"])
    assert clustered["n_resampling_units"] == 50
