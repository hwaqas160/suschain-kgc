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


# Which LODHalal relations enter the graph.
#   full                   - manufacturer, certificate and ingredient relations (the original setting)
#   no_cert                - drops the certificate relation and node type. The certification-
#                            compliance risk is computed from certificates, so this variant is
#                            the leakage-free control for any experiment that scores that risk.
#   food_manufacturer_only - the manufacturer relation alone; the graph Peeris et al. (2026) used.
LODHALAL_RELATION_PRESETS = {
    "full": ("has_manufacturer", "has_cert", "contains"),
    "no_cert": ("has_manufacturer", "contains"),
    "food_manufacturer_only": ("has_manufacturer",),
}


def build_lodhalal_heterodata(g: LODHalalGraph, make_undirected: bool = True,
                              relations: str = "full") -> BuiltGraph:
    if relations not in LODHALAL_RELATION_PRESETS:
        raise ValueError(f"unknown relations preset {relations!r}; "
                         f"expected one of {sorted(LODHALAL_RELATION_PRESETS)}")
    keep = LODHALAL_RELATION_PRESETS[relations]

    food_map = _id_map(g.food["id"])
    man_map = _id_map(g.manufacturer["id"])
    id_maps = {"food": food_map, "manufacturer": man_map}
    node_maps = [("food", food_map), ("manufacturer", man_map)]
    if "contains" in keep:
        id_maps["ingredient"] = _id_map(g.ingredient["id"])
        node_maps.append(("ingredient", id_maps["ingredient"]))
    if "has_cert" in keep:
        id_maps["cert"] = _id_map(g.cert["id"])
        node_maps.append(("cert", id_maps["cert"]))

    data = HeteroData()
    for ntype, m in node_maps:
        n = len(m)
        data[ntype].num_nodes = n
        data[ntype].node_id = torch.arange(n, dtype=torch.long)

    data["food", "has_manufacturer", "manufacturer"].edge_index = _edge_index(
        g.has_manufacturer, "food", "manufacturer", food_map, man_map
    )
    if "has_cert" in keep:
        data["food", "has_cert", "cert"].edge_index = _edge_index(
            g.has_cert, "food", "cert", food_map, id_maps["cert"]
        )
    if "contains" in keep:
        data["food", "contains", "ingredient"].edge_index = _edge_index(
            g.contains, "food", "ingredients", food_map, id_maps["ingredient"]
        )

    if make_undirected:
        data = ToUndirected()(data)

    return BuiltGraph(data=data, id_maps=id_maps)


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


def build_anonnet_directed_heterodata(g: AnonNetGraph) -> BuiltGraph:
    """v4: leak-free AnonNet graph for link prediction.

    build_anonnet_heterodata(make_undirected=True) folds every reverse edge into
    the same ("node", "supplies", "node") relation. A transductive split of that
    relation assigns a->b and b->a independently, so about 80% of test
    positives have their reverse in training, and a symmetric (dot-product)
    decoder scores the two identically. Here the reverse direction is a separate
    relation, "rev_supplies", so passing it as rev_edge_type to the splitter
    removes the reverses of validation and test edges from message passing.
    Reciprocal pairs present in the raw data (a supplies b AND b supplies a)
    remain: they are facts, not split artefacts.
    """
    node_map = _id_map(g.nodes["id"])
    data = HeteroData()
    n = len(node_map)
    data["node"].num_nodes = n
    data["node"].node_id = torch.arange(n, dtype=torch.long)
    ei = _edge_index(g.edges, "source", "target", node_map, node_map)
    ei = torch.unique(ei, dim=1)
    data["node", "supplies", "node"].edge_index = ei
    data["node", "rev_supplies", "node"].edge_index = ei.flip(0)
    return BuiltGraph(data=data, id_maps={"node": node_map})


def build_anonnet_product_location(g: AnonNetGraph, variant: str = "firm") -> BuiltGraph:
    """v4 work package 9: reconstruction of the 3,813-node AnonNet graph of
    Peeris et al. (2026).

    Nodes (confirmed by their first author: "the unique ids mapped and stored
    in the data frame"): every distinct value of the `location` column and
    every distinct `products` string, the missing value counting once in each
    (106 + 3,707 = 3,813).

    Edges (not stated; two readings of "relationships between products and
    locations"):
      variant="firm"    one link per firm between its products string and its
                        location (5,165 distinct pairs)
      variant="supply"  one link per supply relation, from the supplier's
                        products string to the customer's location (10,389)
    Target relation ("product", "at", "location"); reverse added by ToUndirected.
    """
    import pandas as pd

    nodes = g.nodes
    loc = nodes["location"].fillna("<missing>").astype(str)
    prod = nodes["products"].fillna("<missing>").astype(str)
    pmap = {p: i for i, p in enumerate(sorted(prod.unique()))}
    lmap = {l: i for i, l in enumerate(sorted(loc.unique()))}
    if variant == "firm":
        pairs = pd.DataFrame({"p": prod.map(pmap), "l": loc.map(lmap)})
    elif variant == "supply":
        p_of = dict(zip(nodes["id"], prod.map(pmap)))
        l_of = dict(zip(nodes["id"], loc.map(lmap)))
        pairs = pd.DataFrame({"p": g.edges["source"].map(p_of), "l": g.edges["target"].map(l_of)}).dropna()
    else:
        raise ValueError(variant)
    pairs = pairs.drop_duplicates().astype(int)
    data = HeteroData()
    data["product"].num_nodes = len(pmap)
    data["product"].node_id = torch.arange(len(pmap))
    data["location"].num_nodes = len(lmap)
    data["location"].node_id = torch.arange(len(lmap))
    data["product", "at", "location"].edge_index = torch.tensor(pairs[["p", "l"]].to_numpy().T, dtype=torch.long)
    data = ToUndirected()(data)
    return BuiltGraph(data=data, id_maps={"product": pmap, "location": lmap})
