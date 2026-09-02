from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal


def test_lodhalal_heterodata_metadata():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g)
    data = built.data
    node_types, edge_types = data.metadata()
    assert set(node_types) == {"food", "manufacturer", "ingredient", "cert"}
    assert ("food", "has_manufacturer", "manufacturer") in edge_types
    assert ("manufacturer", "rev_has_manufacturer", "food") in edge_types  # ToUndirected reverse
    assert data["food"].num_nodes == 43101
    assert data["manufacturer"].num_nodes == 2520
    assert data["food", "has_manufacturer", "manufacturer"].edge_index.shape[1] == 43101
    # all node_ids within valid range, no NaNs from a bad id-map lookup
    ei = data["food", "has_manufacturer", "manufacturer"].edge_index
    assert ei.min() >= 0
    assert ei[0].max() < data["food"].num_nodes
    assert ei[1].max() < data["manufacturer"].num_nodes


def test_anonnet_heterodata_metadata():
    g = load_anonnet()

    # Raw directed graph: edge count matches the cleaned raw edge list exactly.
    built_directed = build_anonnet_heterodata(g, make_undirected=False)
    ei_dir = built_directed.data["node", "supplies", "node"].edge_index
    assert built_directed.data["node"].num_nodes == 7245
    assert ei_dir.shape[1] == 43926
    assert ei_dir.min() >= 0
    assert ei_dir.max() < 7245

    # Undirected (symmetrized + coalesced) version is >= the directed count,
    # since ToUndirected merges any edge pairs that were already bidirectional.
    built_undirected = build_anonnet_heterodata(g, make_undirected=True)
    ei_undir = built_undirected.data["node", "supplies", "node"].edge_index
    assert ei_undir.shape[1] >= ei_dir.shape[1]
