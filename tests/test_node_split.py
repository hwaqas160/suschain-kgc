import numpy as np

from suschain.data.splitters import split_node_classification


def test_split_node_classification_stratified_and_disjoint():
    # 85 "A", 10 "B", 5 "C" -- deliberately imbalanced (mirrors the real
    # cert_coverage skew this splitter exists to handle correctly), but
    # every class has enough members for stratification to be well-defined
    # (sklearn's StratifiedShuffleSplit requires >=2 per class -- our real
    # data's smallest class, "Partially certified", has 19, so a singleton
    # class isn't a case this project's actual data hits).
    node_ids = np.arange(100)
    labels = ["A"] * 85 + ["B"] * 10 + ["C"] * 5
    split = split_node_classification(node_ids, labels, num_nodes_total=100, val_size=0.2, test_size=0.2, seed=0)

    assert set(split.train_idx) | set(split.val_idx) | set(split.test_idx) <= set(node_ids)
    assert set(split.train_idx).isdisjoint(split.val_idx)
    assert set(split.train_idx).isdisjoint(split.test_idx)
    assert set(split.val_idx).isdisjoint(split.test_idx)
    assert len(split.train_idx) + len(split.val_idx) + len(split.test_idx) == 100
    assert split.class_names == ["A", "B", "C"]


def test_split_node_classification_roughly_preserves_class_proportions():
    node_ids = np.arange(1000)
    labels = ["A"] * 800 + ["B"] * 150 + ["C"] * 50
    split = split_node_classification(node_ids, labels, num_nodes_total=1000, val_size=0.2, test_size=0.2, seed=0)

    test_labels = [labels[i] for i in split.test_idx]
    test_a_frac = test_labels.count("A") / len(test_labels)
    # stratified split should keep test-set class balance close to the
    # overall 80% "A" proportion, not drift to something wildly different
    assert 0.7 < test_a_frac < 0.9


def test_labels_are_graph_node_id_indexed_not_positional():
    # Regression test for a real bug: node_ids is a SPARSE, non-contiguous
    # subset of a larger graph (only some nodes have labels), and
    # split.train_idx/val_idx/test_idx are GRAPH node ids -- indexing
    # split.labels with them must give the right answer directly, not
    # require the caller to know about a separate positional index space.
    num_nodes_total = 20
    node_ids = np.array([2, 5, 9, 13, 17, 3, 6, 10, 14, 18])  # sparse, unlabeled gaps in between
    labels = ["X", "X", "X", "X", "X", "Y", "Y", "Y", "Y", "Y"]
    split = split_node_classification(node_ids, labels, num_nodes_total=num_nodes_total, val_size=0.2, test_size=0.2, seed=0)

    assert split.labels.shape == (num_nodes_total,)
    # every id in node_ids must map back to its original string label
    name_to_idx = {name: i for i, name in enumerate(split.class_names)}
    for nid, lab in zip(node_ids, labels):
        assert split.labels[nid] == name_to_idx[lab]
    # all train/val/test indices must be valid, in-range graph node ids
    all_idx = np.concatenate([split.train_idx, split.val_idx, split.test_idx])
    assert all_idx.max() < num_nodes_total
    assert set(all_idx) <= set(node_ids)
    # indexing labels with train_idx directly (as the training loop does)
    # must not raise and must return only valid class indices
    assert (split.labels[split.train_idx] >= 0).all()
