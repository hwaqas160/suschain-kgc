"""Loads the raw AnonNet CSVs.

Source: https://github.com/lx249/anon-network (Xu, Long & Brintrup, 2025),
the AnonNet supply-network dataset the base paper cites as its second
dataset. Raw files copied into data/raw/anonnet/.

Note (docs/DATA_PROVENANCE.md): the base paper reports 3,813 nodes / 106
locations for this dataset. The raw file actually has 7,245 nodes and no
documented subsetting procedure is given anywhere in the paper. This loader
returns the full raw graph; any subsetting decision is made explicitly and
separately in anonnet_subsetting.py so it stays auditable rather than baked
silently into the loader.
"""
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "anonnet"


@dataclass
class AnonNetGraph:
    nodes: pd.DataFrame  # columns: id, location, products
    edges: pd.DataFrame  # columns: source, target
    n_orphan_edges_dropped: int


def load_anonnet(raw_dir: Path = DEFAULT_RAW_DIR) -> AnonNetGraph:
    nodes = pd.read_csv(raw_dir / "nodes_anon.csv", engine="python")
    edges = pd.read_csv(raw_dir / "edges_anon.csv", engine="python")

    valid_ids = set(nodes["id"])
    before = len(edges)
    edges = edges[
        edges["source"].isin(valid_ids) & edges["target"].isin(valid_ids)
    ].reset_index(drop=True)
    n_orphan_dropped = before - len(edges)

    # Parse the free-text, comma-separated `products` column into a list per
    # node — needed downstream for the product-category taxonomy used to
    # derive richer relation types (see product_taxonomy.py).
    nodes = nodes.copy()
    nodes["products_list"] = nodes["products"].fillna("").apply(
        lambda s: [p.strip() for p in s.split(",") if p.strip()]
    )

    return AnonNetGraph(nodes=nodes, edges=edges, n_orphan_edges_dropped=n_orphan_dropped)
