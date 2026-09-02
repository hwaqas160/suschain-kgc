import torch
import torch.nn.functional as F

from suschain.models.inductive_link_predictor import HeteroInductiveLinkPredictor

METADATA = (
    ["food", "manufacturer"],
    [("food", "has_manufacturer", "manufacturer"), ("manufacturer", "rev_has_manufacturer", "food")],
)


def _bidirectional_edge_index_dict():
    fwd = torch.stack([torch.arange(10), torch.arange(10) % 5])
    rev = fwd.flip(0)
    return {
        ("food", "has_manufacturer", "manufacturer"): fwd,
        ("manufacturer", "rev_has_manufacturer", "food"): rev,
    }


def _toy_model(use_text_for_food: bool) -> HeteroInductiveLinkPredictor:
    text_features = {"food": torch.randn(10, 16)} if use_text_for_food else None
    return HeteroInductiveLinkPredictor(
        metadata=METADATA,
        num_nodes_dict={"food": 10, "manufacturer": 5},
        text_features=text_features,
        embedding_dim=8, hidden_channels=8, out_channels=4, num_layers=1,
    )


def test_forward_pass_with_mixed_text_and_embedding_node_types():
    model = _toy_model(use_text_for_food=True)
    node_id_dict = {"food": torch.arange(10), "manufacturer": torch.arange(5)}
    edge_index_dict = _bidirectional_edge_index_dict()
    z = model.encode(node_id_dict, edge_index_dict)
    assert z["food"].shape == (10, 4)
    assert z["manufacturer"].shape == (5, 4)
    # "food" should NOT be in the trainable embeddings dict (it uses text features)
    assert "food" not in model.embeddings
    assert "manufacturer" in model.embeddings


def test_forward_pass_all_embeddings_when_no_text_features():
    model = _toy_model(use_text_for_food=False)
    assert "food" in model.embeddings and "manufacturer" in model.embeddings
    assert len(model.text_projections) == 0


def test_text_feature_buffer_never_updated_by_training():
    # Fixed seed: this test previously used the ambient global RNG state
    # for both model init and text_features, which was flaky -- an
    # occasional random init produced a near-zero gradient sum for the
    # small toy projection layer, failing the "gradients did flow"
    # sanity check nondeterministically. Caught directly (not assumed) via
    # a real intermittent CI-style failure.
    torch.manual_seed(0)
    model = _toy_model(use_text_for_food=True)
    original_food_features = getattr(model, "text_feat_food").clone()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    node_id_dict = {"food": torch.arange(10), "manufacturer": torch.arange(5)}
    edge_index_dict = _bidirectional_edge_index_dict()
    edge_label_index = torch.stack([torch.arange(10), torch.arange(10) % 5])
    labels = torch.ones(10)

    for _ in range(5):
        optimizer.zero_grad()
        z = model.encode(node_id_dict, edge_index_dict)
        scores = model.decode(z, edge_label_index, "food", "manufacturer")
        loss = F.binary_cross_entropy_with_logits(scores, labels)
        loss.backward()
        optimizer.step()

    # the raw text feature buffer must be bit-for-bit unchanged -- only the
    # projection Linear layer's weights should have moved
    assert torch.equal(getattr(model, "text_feat_food"), original_food_features)
    # sanity: the projection layer's weights SHOULD have changed (loss was
    # non-trivial and gradients did flow through it)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.text_projections["food"].parameters())
