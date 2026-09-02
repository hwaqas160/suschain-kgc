"""Non-learned heuristic baselines for link prediction (plan C2/B3: the base
paper has zero heuristic floors -- "you cannot know whether their GNN beats
a 1990s heuristic").

Common Neighbours / Adamic-Adar / Preferential Attachment (Liben-Nowell &
Kleinberg, 2003) are classic same-node-type heuristics: they compare a
source node's and a candidate destination node's neighborhoods in a
homogeneous graph. They apply directly and meaningfully to **AnonNet**
(single node type, single relation). They do NOT transfer naively to
**LODHalal**'s food->manufacturer relation: food nodes only ever have
exactly one manufacturer neighbor each (verified, docs/DATA_PROVENANCE.md),
so a food and manufacturer node's raw neighbor sets never overlap (they
live in disjoint node-type spaces one hop apart) -- CN/AA/PA computed
naively there would be uniformly zero and uninformative, not a fair
baseline. For LODHalal we instead report the **popularity** baseline
(score a candidate manufacturer by its training-set in-degree, i.e.
"recommend whichever manufacturer already makes the most products"), which
is well-defined on any relation and is the honest, standard fallback when
same-type-neighborhood heuristics don't apply. This asymmetry is stated
here explicitly rather than silently applying a meaningless heuristic to
get a number.

All scoring is vectorized via sparse adjacency matrices -- a naive
per-pair networkx loop would need ~n_test * num_nodes heuristic
evaluations (tens of millions for these graph sizes), which is
prohibitively slow in pure Python.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch


def build_adjacency(edge_index: torch.Tensor, num_nodes: int) -> sp.csr_matrix:
    """Symmetric, unweighted, binary adjacency matrix (no self-loops) from
    training-positive edges only (callers must pass the train-split
    edge_index to avoid val/test leakage)."""
    src, dst = edge_index[0].numpy(), edge_index[1].numpy()
    data = np.ones(len(src), dtype=np.float64)
    A = sp.csr_matrix((data, (src, dst)), shape=(num_nodes, num_nodes))
    A = A.maximum(A.T)  # symmetrize
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1.0  # binarize (in case of parallel edges)
    return A.tocsr()


class HeuristicScorer:
    """Precomputes what each heuristic needs once, then scores a source
    node against ALL destination-type node indices in one sparse matvec --
    fast enough for full-catalog filtered ranking."""

    def __init__(self, A: sp.csr_matrix):
        self.A = A
        self.degree = np.asarray(A.sum(axis=1)).ravel()
        with np.errstate(divide="ignore"):
            aa_weight = np.where(self.degree > 1, 1.0 / np.log(np.maximum(self.degree, 2)), 0.0)
        # AA(u,v) = sum_k A[u,k] * weight[k] * A[k,v] -- the weight applies to
        # the CONTRACTED (common-neighbor) index k, so it must scale A's
        # columns on the operand whose row we extract for `src` (giving
        # A[src,k]*weight[k] before multiplying by the *unweighted* A on the
        # right to sum over k). Scaling the wrong operand/axis here would
        # silently compute weight[v] (destination degree) instead of
        # weight[k] (common-neighbor degree) -- caught by the toy-example
        # test in tests/test_heuristics.py.
        self.A_aa_weighted = A.multiply(aa_weight[np.newaxis, :]).tocsr()  # cols scaled by AA weight

    def common_neighbors_row(self, src: int) -> np.ndarray:
        # (A[src,:] @ A)[v] = |N(src) intersect N(v)| for unweighted binary A
        return np.asarray((self.A[src] @ self.A).todense()).ravel()

    def adamic_adar_row(self, src: int) -> np.ndarray:
        return np.asarray((self.A_aa_weighted[src] @ self.A).todense()).ravel()

    def preferential_attachment_row(self, src: int) -> np.ndarray:
        return self.degree[src] * self.degree

    def score_row(self, src: int, method: str) -> np.ndarray:
        return {
            "common_neighbors": self.common_neighbors_row,
            "adamic_adar": self.adamic_adar_row,
            "preferential_attachment": self.preferential_attachment_row,
        }[method](src)


def popularity_scores(edge_index: torch.Tensor, num_dst_nodes: int) -> np.ndarray:
    """In-degree of each destination-type node in the training graph --
    the "recommend the most popular target" baseline, well-defined for any
    relation regardless of node-type structure."""
    counts = np.zeros(num_dst_nodes, dtype=np.float64)
    dst = edge_index[1].numpy()
    np.add.at(counts, dst, 1.0)
    return counts
