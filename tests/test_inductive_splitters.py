import torch
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected

from suschain.data.inductive_splitters import inductive_entity_holdout


def _toy_bipartite_data() -> HeteroData:
    data = HeteroData()
    data["food"].num_nodes = 10
    data["food"].node_id = torch.arange(10)
    data["manufacturer"].num_nodes = 5
    data["manufacturer"].node_id = torch.arange(5)
    # each food -> exactly one manufacturer, food i -> manufacturer i % 5
    src = torch.arange(10)
    dst = torch.arange(10) % 5
    data["food", "has_manufacturer", "manufacturer"].edge_index = torch.stack([src, dst])
    return ToUndirected()(data)


def test_heldout_entities_are_degree_zero_in_train_graph():
    data = _toy_bipartite_data()
    edge_type = ("food", "has_manufacturer", "manufacturer")
    split = inductive_entity_holdout(data, edge_type, holdout_node_type="manufacturer", holdout_fraction=0.4, seed=0)

    assert len(split.heldout_entity_ids) == 2  # 40% of 5 manufacturers

    train_dst = split.train_data[edge_type].edge_index[1]
    for heldout_id in split.heldout_entity_ids:
        assert heldout_id not in train_dst.tolist(), f"manufacturer {heldout_id} still has train edges"

    # reverse relation must also be stripped (manufacturer -> food)
    rev_type = ("manufacturer", "rev_has_manufacturer", "food")
    train_rev_src = split.train_data[rev_type].edge_index[0]
    for heldout_id in split.heldout_entity_ids:
        assert heldout_id not in train_rev_src.tolist()


def test_heldout_test_edges_cover_exactly_the_removed_edges():
    data = _toy_bipartite_data()
    edge_type = ("food", "has_manufacturer", "manufacturer")
    split = inductive_entity_holdout(data, edge_type, holdout_node_type="manufacturer", holdout_fraction=0.4, seed=0)

    original_n_edges = data[edge_type].edge_index.shape[1]
    train_n_edges = split.train_data[edge_type].edge_index.shape[1]
    heldout_n_edges = split.heldout_test_edges.shape[1]
    assert train_n_edges + heldout_n_edges == original_n_edges

    heldout_dst = split.heldout_test_edges[1]
    assert set(heldout_dst.tolist()) <= set(split.heldout_entity_ids.tolist())


def test_holdout_source_side_instead_of_destination():
    data = _toy_bipartite_data()
    edge_type = ("food", "has_manufacturer", "manufacturer")
    split = inductive_entity_holdout(data, edge_type, holdout_node_type="food", holdout_fraction=0.3, seed=0)
    assert len(split.heldout_entity_ids) == 3
    train_src = split.train_data[edge_type].edge_index[0]
    for heldout_id in split.heldout_entity_ids:
        assert heldout_id not in train_src.tolist()
