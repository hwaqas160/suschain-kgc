"""Supplier-substitution queries with real, held-out ground truth (AnonNet).

The original disruption simulation scored a method on whether it recovered the
disrupted supplier itself, which in a real disruption is unavailable. Here the
disrupted supplier is excluded, and a recommendation counts as correct only if
it is a real supplier of the customer that the model never saw and that sells
products similar to the disrupted one.

Construction, for one seed:
  1. Hide a fraction of supply links (supplier v -> customer c). The rest form
     the training graph.
  2. For a customer c with a hidden supplier v and a visible supplier s, where
     v and s are product-similar (sim >= tau), form the query
         "c loses s: recommend replacements"
     whose correct answers are T(c, s) = {hidden suppliers v of c : sim(v, s) >= tau}.
  3. Candidates are firms capable of replacing s, C(s) = {v : sim(v, s) >= tau},
     minus s, c and c's visible suppliers. T(c, s) is inside C(s) by construction.

Product similarity is the cosine between inverse-document-frequency weighted
binary vectors of a firm's listed product items, so items listed by hundreds of
firms (e.g. "etc part of vehicle") carry little weight.

Edge direction: the dataset does not document it. We read source -> target as
supplier -> customer; docs/v4/anonnet_edge_direction.md gives the evidence.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp


def product_items(products_list) -> frozenset:
    if not isinstance(products_list, (list, tuple)):
        return frozenset()
    return frozenset(p.strip().lower() for p in products_list if isinstance(p, str) and p.strip())


def idf_product_matrix(items_per_firm: list[frozenset]) -> sp.csr_matrix:
    """Rows: firms; columns: product items; values: idf weights, rows L2-normalised.
    Firms with no listed product get an all-zero row (similar to nothing)."""
    vocab: dict[str, int] = {}
    rows, cols = [], []
    for i, items in enumerate(items_per_firm):
        for it in items:
            j = vocab.setdefault(it, len(vocab))
            rows.append(i)
            cols.append(j)
    n = len(items_per_firm)
    X = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, len(vocab)))
    df = np.asarray((X > 0).sum(axis=0)).ravel()
    idf = np.log(n / df)
    X = X.multiply(idf[None, :]).tocsr()
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sp.diags(1.0 / norms) @ X


@dataclass
class SubstitutionQuery:
    customer: int
    disrupted: int                 # the visible supplier s that fails
    truth: np.ndarray              # hidden product-similar suppliers of the customer
    candidates: np.ndarray         # capable firms, excluding s, c and c's visible suppliers


@dataclass
class SubstitutionScenario:
    train_edges: np.ndarray        # [2, m] supplier -> customer, visible
    hidden_edges: np.ndarray       # [2, h] supplier -> customer, hidden
    queries: list[SubstitutionQuery]


def build_substitution_scenario(edges: np.ndarray, X: sp.csr_matrix, *, tau: float,
                                hidden_fraction: float = 0.1, seed: int = 0,
                                max_queries_per_customer: int | None = None) -> SubstitutionScenario:
    """edges: [2, E] supplier -> customer. Self-loops are dropped."""
    edges = edges[:, edges[0] != edges[1]]
    rng = np.random.default_rng(seed)
    E = edges.shape[1]
    hide = np.zeros(E, dtype=bool)
    hide[rng.choice(E, size=int(round(E * hidden_fraction)), replace=False)] = True
    train, hidden = edges[:, ~hide], edges[:, hide]

    visible_sup = defaultdict(set)
    for s, c in train.T:
        visible_sup[int(c)].add(int(s))
    hidden_sup = defaultdict(set)
    for v, c in hidden.T:
        if int(v) not in visible_sup[int(c)]:      # a duplicate pair is not hidden
            hidden_sup[int(c)].add(int(v))

    queries = []
    for c in sorted(hidden_sup):
        H = np.array(sorted(hidden_sup[c]))
        vis = sorted(visible_sup[c])
        if not vis:
            continue
        sims_HS = (X[H] @ X[vis].T).toarray()      # [|H|, |vis|]
        made = 0
        for j in rng.permutation(len(vis)):
            s = vis[j]
            T = H[sims_HS[:, j] >= tau]
            T = T[T != s]
            if len(T) == 0:
                continue
            sim_s = (X @ X[s].T).toarray().ravel()
            cand = np.flatnonzero(sim_s >= tau)
            excl = visible_sup[c] | {s, c}
            cand = cand[~np.isin(cand, list(excl))]
            queries.append(SubstitutionQuery(c, s, T, cand))
            made += 1
            if max_queries_per_customer and made >= max_queries_per_customer:
                break
    return SubstitutionScenario(train, hidden, queries)


def evaluate_substitution(ranked: list[np.ndarray], queries: list[SubstitutionQuery],
                          risk: np.ndarray, k: int) -> dict[str, np.ndarray]:
    """ranked[i]: candidate firms for query i, best first (only the first k used,
    plus the full list for the reciprocal rank). Returns per-query arrays."""
    n = len(queries)
    hit, rec, rr, mrisk, drisk = (np.zeros(n) for _ in range(5))
    for i, (r, q) in enumerate(zip(ranked, queries)):
        top = r[:k]
        inT = np.isin(r, q.truth)
        hit[i] = float(inT[:k].any())
        rec[i] = float(np.isin(top, q.truth).sum() / min(len(q.truth), k))
        pos = np.flatnonzero(inT)
        rr[i] = 1.0 / (pos[0] + 1) if len(pos) else 0.0
        mrisk[i] = float(risk[top].mean()) if len(top) else np.nan
        drisk[i] = mrisk[i] - float(risk[q.disrupted])
    return {"hit": hit, "recall": rec, "rr": rr, "risk": mrisk, "delta_risk": drisk}


def rank_candidates(scores: np.ndarray, q: SubstitutionQuery, tie_break: np.ndarray | None = None) -> np.ndarray:
    """Sorts q.candidates by descending score; ties broken by tie_break, then index."""
    s = scores[q.candidates]
    if tie_break is None:
        order = np.argsort(-s, kind="stable")
    else:
        order = np.lexsort((-tie_break[q.candidates], -s))
    return q.candidates[order]
