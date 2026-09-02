"""Produces the descriptive graph-statistics table the base paper omits
(plan limitation B15): node/edge counts by type, density, degree
distribution summary, and connected-component structure, for both datasets.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from suschain.data.anonnet_loader import AnonNetGraph
from suschain.data.lodhalal_loader import LODHalalGraph


def _edge_stats(name: str, src_count: int, dst_count: int, n_edges: int, degrees_src: pd.Series) -> dict[str, Any]:
    possible = src_count * dst_count
    return {
        "relation": name,
        "n_edges": int(n_edges),
        "n_src_nodes": int(src_count),
        "n_dst_nodes": int(dst_count),
        "possible_pairs": int(possible),
        "density_pct": 100.0 * n_edges / possible if possible else float("nan"),
        "src_degree_mean": float(degrees_src.mean()) if len(degrees_src) else 0.0,
        "src_degree_median": float(degrees_src.median()) if len(degrees_src) else 0.0,
        "src_degree_max": int(degrees_src.max()) if len(degrees_src) else 0,
        "src_degree_min": int(degrees_src.min()) if len(degrees_src) else 0,
    }


def lodhalal_stats(g: LODHalalGraph) -> dict[str, Any]:
    node_counts = {
        "food": len(g.food),
        "manufacturer": len(g.manufacturer),
        "ingredient": len(g.ingredient),
        "cert": len(g.cert),
    }

    relations = [
        _edge_stats(
            "food-has_manufacturer-manufacturer",
            len(g.food), len(g.manufacturer), len(g.has_manufacturer),
            g.has_manufacturer.groupby("food").size(),
        ),
        _edge_stats(
            "food-has_cert-cert",
            len(g.food), len(g.cert), len(g.has_cert),
            g.has_cert.groupby("food").size(),
        ),
        _edge_stats(
            "food-contains-ingredient",
            len(g.food), len(g.ingredient), len(g.contains),
            g.contains.groupby("food").size(),
        ),
    ]

    return {
        "dataset": "LODHalal",
        "node_counts": node_counts,
        "total_nodes": sum(node_counts.values()),
        "relations": relations,
        "n_orphan_hascert_rows_dropped": g.n_orphan_hascert_dropped,
    }


def anonnet_stats(g: AnonNetGraph) -> dict[str, Any]:
    n_nodes = len(g.nodes)
    n_locations = g.nodes["location"].nunique()
    n_edges = len(g.edges)
    possible = n_nodes * (n_nodes - 1)

    G = nx.from_pandas_edgelist(g.edges, source="source", target="target", create_using=nx.DiGraph)
    G.add_nodes_from(g.nodes["id"])  # include isolated nodes, if any
    G_undirected = G.to_undirected()

    degrees = pd.Series(dict(G_undirected.degree()))
    components = sorted(
        (len(c) for c in nx.connected_components(G_undirected)), reverse=True
    )

    return {
        "dataset": "AnonNet",
        "n_nodes": int(n_nodes),
        "n_unique_locations": int(n_locations),
        "n_edges": int(n_edges),
        "possible_directed_pairs": int(possible),
        "density_pct": 100.0 * n_edges / possible if possible else float("nan"),
        "degree_mean": float(degrees.mean()),
        "degree_median": float(degrees.median()),
        "degree_max": int(degrees.max()),
        "degree_min": int(degrees.min()),
        "n_connected_components": len(components),
        "largest_component_size": components[0] if components else 0,
        "n_isolated_nodes": int((degrees == 0).sum()),
        "n_orphan_edges_dropped": g.n_orphan_edges_dropped,
    }


def to_dataframe(stats: dict[str, Any]) -> pd.DataFrame:
    """Flatten a stats dict (LODHalal has a nested `relations` list) into a
    tidy long-format DataFrame suitable for a paper table / CSV export."""
    rows = []
    for k, v in stats.items():
        if k == "relations":
            for rel in v:
                for rk, rv in rel.items():
                    rows.append({"key": f"relation.{rel['relation']}.{rk}", "value": rv})
        elif k == "node_counts":
            for nk, nv in v.items():
                rows.append({"key": f"node_count.{nk}", "value": nv})
        else:
            rows.append({"key": k, "value": v})
    return pd.DataFrame(rows)
