"""Builds torch_geometric HeteroData objects from the cleaned LODHalal /
AnonNet tables produced by lodhalal_loader.py / anonnet_loader.py.

Node features: neither raw dataset has meaningful numeric node features
(paper §4.2 used "one-hot or categorical representations"). Rather than
materializing huge one-hot matrices, each node type gets a `node_id`
LongTensor (0..num_nodes-1) and models look up a trainable
`torch.nn.Embedding(num_nodes, dim)` at train time -- functionally
equivalent to a learned one-hot embedding table, standard PyG practice for
featureless graphs, and far cheaper in memory. Phase 4 (C4, inductive
evaluation) swaps this for sentence-transformer text embeddings so unseen
nodes at test time still get a feature vector, which this id-embedding
scheme cannot provide (that gap is exactly what C4 is designed to expose).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from suschain.data.anonnet_loader import AnonNetGraph
from suschain.data.lodhalal_loader import LODHalalGraph

IdMap = dict[int, int]


@dataclass
class BuiltGraph:
    data: HeteroData
    id_maps: dict[str, IdMap]


def _id_map(series) -> IdMap:
    """Maps raw entity IDs (int for LODHalal, e.g. 'n0' string for AnonNet)
    to contiguous 0-indexed local IDs. Kept generic over the raw ID type."""
    ids = series.unique()
    return {rid: i for i, rid in enumerate(ids)}


def _edge_index(df, src_col: str, dst_col: str, src_map: IdMap, dst_map: IdMap) -> torch.Tensor:
    src = df[src_col].map(src_map).to_numpy(dtype="int64")
    dst = df[dst_col].map(dst_map).to_numpy(dtype="int64")
    return torch.from_numpy(np.stack([src, dst]))


def build_lodhalal_heterodata(g: LODHalalGraph, make_undirected: bool = True) -> BuiltGraph:
    food_map = _id_map(g.food["id"])
    man_map = _id_map(g.manufacturer["id"])
    ing_map = _id_map(g.ingredient["id"])
    cert_map = _id_map(g.cert["id"])

    data = HeteroData()
    for ntype, m in [("food", food_map), ("manufacturer", man_map), ("ingredient", ing_map), ("cert", cert_map)]:
        n = len(m)
        data[ntype].num_nodes = n
        data[ntype].node_id = torch.arange(n, dtype=torch.long)

    data["food", "has_manufacturer", "manufacturer"].edge_index = _edge_index(
        g.has_manufacturer, "food", "manufacturer", food_map, man_map
    )
    data["food", "has_cert", "cert"].edge_index = _edge_index(
        g.has_cert, "food", "cert", food_map, cert_map
    )
    data["food", "contains", "ingredient"].edge_index = _edge_index(
        g.contains, "food", "ingredients", food_map, ing_map
    )

    if make_undirected:
        data = ToUndirected()(data)

    return BuiltGraph(
        data=data,
        id_maps={"food": food_map, "manufacturer": man_map, "ingredient": ing_map, "cert": cert_map},
    )


def build_anonnet_heterodata(g: AnonNetGraph, make_undirected: bool = True) -> BuiltGraph:
    """Single node-type ("node"), single relation ("supplies") graph, matching
    the raw data as-is (the base paper's "single edge type" setting). Richer
    relation types (SAME_LOCATION / SHARES_CATEGORY, derived from the
    `location`/`products` columns) are added later by
    enrichment/anonnet_relations.py, not baked in here, so this function
    stays a faithful, minimal representation of the raw graph.
    """
    node_map = _id_map(g.nodes["id"])

    data = HeteroData()
    n = len(node_map)
    data["node"].num_nodes = n
    data["node"].node_id = torch.arange(n, dtype=torch.long)

    data["node", "supplies", "node"].edge_index = _edge_index(
        g.edges, "source", "target", node_map, node_map
    )

    if make_undirected:
        data = ToUndirected()(data)

    return BuiltGraph(data=data, id_maps={"node": node_map})
