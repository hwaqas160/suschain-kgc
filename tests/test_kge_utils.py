import numpy as np
import torch

from suschain.models.kge import _build_entity_vocab, _edges_to_triples


def test_entity_vocab_same_node_type():
    entity_to_id, src_off, dst_off = _build_entity_vocab(num_src=5, num_dst=5, same_node_type=True)
    assert src_off == 0 and dst_off == 0
    assert len(entity_to_id) == 5
    assert entity_to_id == {str(i): i for i in range(5)}


def test_entity_vocab_disjoint_node_types_no_overlap():
    entity_to_id, src_off, dst_off = _build_entity_vocab(num_src=3, num_dst=4, same_node_type=False)
    assert src_off == 0 and dst_off == 3
    assert len(entity_to_id) == 7  # 3 + 4, no collisions
    assert entity_to_id["0"] == 0 and entity_to_id["2"] == 2
    assert entity_to_id["3"] == 3 and entity_to_id["6"] == 6  # dst ids offset by num_src


def test_edges_to_triples_shape_and_offset():
    ei = torch.tensor([[0, 1], [0, 2]])  # 2 edges: (0->0), (1->2)
    triples = _edges_to_triples(ei, "has_manufacturer", src_offset=0, dst_offset=10)
    assert triples.shape == (2, 3)
    assert list(triples[0]) == ["0", "has_manufacturer", "10"]
    assert list(triples[1]) == ["1", "has_manufacturer", "12"]
