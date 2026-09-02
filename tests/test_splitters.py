from suschain.data.anonnet_loader import load_anonnet
from suschain.data.graph_builder import build_anonnet_heterodata, build_lodhalal_heterodata
from suschain.data.lodhalal_loader import load_lodhalal
from suschain.data.splitters import assert_no_leakage, split_link_prediction


def test_lodhalal_linksplit_no_leakage():
    g = load_lodhalal()
    built = build_lodhalal_heterodata(g, make_undirected=False)
    split = split_link_prediction(
        built.data,
        edge_type=("food", "has_manufacturer", "manufacturer"),
        seed=0,
    )
    assert_no_leakage(split)
    # roughly 80/10/10 supervision-edge split (approximate: RandomLinkSplit
    # counts differ slightly due to disjoint_train_ratio bookkeeping)
    n_total = built.data["food", "has_manufacturer", "manufacturer"].edge_index.shape[1]
    n_test_pos = int((split.test["food", "has_manufacturer", "manufacturer"].edge_label == 1.0).sum())
    assert 0.05 * n_total < n_test_pos < 0.15 * n_total


def test_anonnet_linksplit_no_leakage():
    g = load_anonnet()
    built = build_anonnet_heterodata(g, make_undirected=False)
    split = split_link_prediction(
        built.data,
        edge_type=("node", "supplies", "node"),
        seed=0,
    )
    assert_no_leakage(split)
