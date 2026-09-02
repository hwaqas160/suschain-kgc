import numpy as np

from suschain.enrichment.text_features import encode_texts


def test_encode_texts_shape_and_determinism():
    texts = ["Nescafe Original", "Kapal Api Signature", "Steel", "Real Estate Services"]
    emb1 = encode_texts(texts, cache_key="test_toy_texts_v1")
    assert emb1.shape == (4, 384)
    assert emb1.dtype == np.float32

    # second call should hit the cache and return identical values
    emb2 = encode_texts(texts, cache_key="test_toy_texts_v1")
    assert np.array_equal(emb1, emb2)


def test_encode_texts_similar_strings_closer_than_dissimilar():
    texts = ["Nescafe Original Coffee", "Nescafe Original Coffee Drink", "Steel Sheet Metal"]
    emb = encode_texts(texts, cache_key="test_similarity_check_v1")
    # cosine similarity: the two coffee variants should be closer to each
    # other than either is to the unrelated "Steel Sheet Metal" entry
    def cos(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    sim_coffee = cos(emb[0], emb[1])
    sim_cross = cos(emb[0], emb[2])
    assert sim_coffee > sim_cross
