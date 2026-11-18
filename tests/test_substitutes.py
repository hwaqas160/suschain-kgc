import numpy as np
import pytest

from suschain.decision_eval.substitutes import (
    build_substitution_scenario,
    evaluate_substitution,
    idf_product_matrix,
    product_items,
    rank_candidates,
)


def test_product_items_normalises():
    assert product_items([" Steel ", "steel", "", None]) == frozenset({"steel"})
    assert product_items(float("nan")) == frozenset()


def test_idf_matrix_downweights_common_items():
    items = [frozenset({"common", "rare"}), frozenset({"common"}), frozenset({"common"}),
             frozenset({"rare"}), frozenset()]
    X = idf_product_matrix(items)
    S = (X @ X.T).toarray()
    assert S[0, 3] > S[0, 1]            # sharing the rare item counts for more
    assert S[4].sum() == 0              # a firm with no products is similar to nothing
    assert np.allclose(np.diag(S)[:4], 1.0)


def _toy():
    # suppliers 0-3 make "bolt", 4 makes "paint"; customers 10, 11.
    items = [frozenset({"bolt"})] * 4 + [frozenset({"paint"})] + [frozenset()] * 7
    X = idf_product_matrix(items)
    edges = np.array([[0, 1, 2, 3, 4, 0, 1],
                      [10, 10, 10, 11, 10, 11, 11]])
    return X, edges


def test_queries_exclude_failed_and_visible_suppliers_and_keep_truth_hidden():
    X, edges = _toy()
    for seed in range(20):
        sc = build_substitution_scenario(edges, X, tau=0.5, hidden_fraction=0.3, seed=seed)
        visible = {(int(s), int(c)) for s, c in sc.train_edges.T}
        for q in sc.queries:
            assert q.disrupted not in q.candidates and q.customer not in q.candidates
            for v in q.truth:
                assert (int(v), q.customer) not in visible         # truth is hidden
                assert v in q.candidates                           # and reachable
            vis_sup = {s for s, c in visible if c == q.customer}
            assert not set(q.candidates.tolist()) & vis_sup


def test_evaluate_substitution_metrics():
    X, edges = _toy()
    sc = build_substitution_scenario(edges, X, tau=0.5, hidden_fraction=0.3, seed=1)
    assert sc.queries, "toy scenario should produce at least one query"
    risk = np.linspace(0, 1, X.shape[0])
    perfect = [np.concatenate([q.truth, np.setdiff1d(q.candidates, q.truth)]) for q in sc.queries]
    out = evaluate_substitution(perfect, sc.queries, risk, k=1)
    assert out["hit"].mean() == 1.0 and out["rr"].mean() == 1.0


def test_rank_candidates_tie_break():
    from suschain.decision_eval.substitutes import SubstitutionQuery
    q = SubstitutionQuery(customer=9, disrupted=8, truth=np.array([2]), candidates=np.array([1, 2, 3]))
    scores = np.zeros(10)
    tb = np.zeros(10); tb[3] = 1.0
    assert rank_candidates(scores, q, tb)[0] == 3
