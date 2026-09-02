"""Locks in the verified raw-data facts from docs/DATA_PROVENANCE.md so a
future change to the loaders can't silently drift from reality."""
from suschain.data.anonnet_loader import load_anonnet
from suschain.data.lodhalal_loader import load_lodhalal


def test_lodhalal_entity_counts():
    g = load_lodhalal()
    assert len(g.food) == 43101
    assert len(g.manufacturer) == 2520
    assert len(g.ingredient) == 19827
    assert len(g.cert) == 2996


def test_lodhalal_hasmanufacturer_is_clean_and_degree_one():
    g = load_lodhalal()
    assert len(g.has_manufacturer) == 43101
    deg = g.has_manufacturer.groupby("food").size()
    assert deg.min() == 1 and deg.max() == 1


def test_lodhalal_hascert_orphan_cleaning():
    g = load_lodhalal()
    assert g.n_orphan_hascert_dropped == 831
    assert g.has_cert["food"].isin(g.food["id"]).all()


def test_lodhalal_contains_namespace_mismatch():
    g = load_lodhalal()
    assert g.n_contains_rows_raw == 131340
    assert g.n_contains_food_id_mismatch == 129098
    # only the cleaned, resolvable subset should remain
    assert g.contains["food"].isin(g.food["id"]).all()
    assert g.contains["ingredients"].isin(g.ingredient["id"]).all()


def test_anonnet_raw_counts():
    g = load_anonnet()
    assert len(g.nodes) == 7245
    assert g.nodes["location"].nunique() == 105
    assert len(g.edges) == 43926
    assert g.n_orphan_edges_dropped == 0
