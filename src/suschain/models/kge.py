"""PyKEEN-based KG-embedding baselines (TransE, DistMult, ComplEx, RotatE) --
plan C2/B3: "a KG-completion paper with no KGE baseline is not benchmarked."

Uses PyKEEN's built-in filtered RankBasedEvaluator for ranking metrics --
implemented independently of this project's own eval/ranking.py, so it
doubles as a cross-check of that implementation rather than just adding
more baseline numbers.

Note on environment: pykeen==1.11.0 fails to import under a modern
class-resolver (TypeError: "Too few arguments for FunctionResolver") --
verified directly. Fixed by pinning class-resolver==0.5.3 (see
requirements.txt); this is a real, reproducible pinning requirement, not
an incidental note.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from pykeen.pipeline import pipeline
from pykeen.triples import TriplesFactory

from suschain.data.splitters import EdgeType, LinkSplit
from suschain.eval.metrics import best_f1_threshold, binary_link_metrics


@dataclass
class KGEResult:
    model_name: str
    test_metrics: dict
    ranking_metrics: dict


def _build_entity_vocab(num_src: int, num_dst: int, same_node_type: bool) -> tuple[dict[str, int], int, int]:
    """Explicit entity_to_id covering EVERY node in the graph (not just ones
    appearing in the training triples), so a node whose only edge landed in
    val/test isn't missing from PyKEEN's vocabulary (a "cold" node under
    this transductive split still needs an entity id to be scored/ranked)."""
    if same_node_type:
        entity_to_id = {str(i): i for i in range(num_src)}
        return entity_to_id, 0, 0
    entity_to_id = {str(i): i for i in range(num_src)}
    entity_to_id.update({str(num_src + j): num_src + j for j in range(num_dst)})
    return entity_to_id, 0, num_src


def _edges_to_triples(edge_index: torch.Tensor, relation_label: str, src_offset: int, dst_offset: int) -> np.ndarray:
    src = (edge_index[0].numpy() + src_offset).astype(str)
    dst = (edge_index[1].numpy() + dst_offset).astype(str)
    rel = np.full(len(src), relation_label)
    return np.stack([src, rel, dst], axis=1)


def train_kge_baseline(
    split: LinkSplit,
    edge_type: EdgeType,
    num_src_nodes: int,
    num_dst_nodes: int,
    same_node_type: bool,
    model_name: str = "TransE",
    embedding_dim: int = 64,
    epochs: int = 100,
    seed: int = 0,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
) -> KGEResult:
    relation_label = edge_type[1]
    entity_to_id, src_offset, dst_offset = _build_entity_vocab(num_src_nodes, num_dst_nodes, same_node_type)
    relation_to_id = {relation_label: 0}

    # Training triples = message-passing edges UNION supervision positives.
    # KGE has no separate message-passing/supervision concept like a GNN --
    # everything the GNN "saw" during training should be a training triple.
    train_pos_mask = split.train[edge_type].edge_label == 1.0
    train_sup_edges = split.train[edge_type].edge_label_index[:, train_pos_mask]
    mp_edges = split.train[edge_type].edge_index
    all_train_edges = torch.unique(torch.cat([mp_edges, train_sup_edges], dim=1), dim=1)

    val_pos_edges = split.val[edge_type].edge_label_index[:, split.val[edge_type].edge_label == 1.0]
    test_pos_edges = split.test[edge_type].edge_label_index[:, split.test[edge_type].edge_label == 1.0]

    tf_kwargs = dict(entity_to_id=entity_to_id, relation_to_id=relation_to_id)
    tf_train = TriplesFactory.from_labeled_triples(
        _edges_to_triples(all_train_edges, relation_label, src_offset, dst_offset), **tf_kwargs
    )
    tf_val = TriplesFactory.from_labeled_triples(
        _edges_to_triples(val_pos_edges, relation_label, src_offset, dst_offset), **tf_kwargs
    )
    tf_test = TriplesFactory.from_labeled_triples(
        _edges_to_triples(test_pos_edges, relation_label, src_offset, dst_offset), **tf_kwargs
    )

    result = pipeline(
        training=tf_train,
        validation=tf_val,
        testing=tf_test,
        model=model_name,
        model_kwargs=dict(embedding_dim=embedding_dim),
        training_kwargs=dict(num_epochs=epochs, use_tqdm=False, use_tqdm_batch=False),
        random_seed=seed,
        device=device,
        evaluation_fallback=True,
    )

    mr = result.metric_results
    ranking_metrics = {
        "mrr": mr.get_metric("both.realistic.inverse_harmonic_mean_rank"),
        "hits@1": mr.get_metric("both.realistic.hits_at_1"),
        "hits@3": mr.get_metric("both.realistic.hits_at_3"),
        "hits@10": mr.get_metric("both.realistic.hits_at_10"),
    }

    # Binary classification metrics on the SAME pos+neg test pairs used for
    # GNN/heuristic evaluation, for direct comparability across all methods.
    model = result.model
    rel_id = tf_train.relation_to_id[relation_label]

    def _score_edge_label_index(store) -> np.ndarray:
        ei = store.edge_label_index
        heads = (ei[0].numpy() + src_offset).astype(str)
        tails = (ei[1].numpy() + dst_offset).astype(str)
        head_ids = torch.tensor([entity_to_id[h] for h in heads])
        tail_ids = torch.tensor([entity_to_id[t] for t in tails])
        rels = torch.full_like(head_ids, rel_id)
        hrt = torch.stack([head_ids, rels, tail_ids], dim=1).to(device)
        with torch.no_grad():
            return model.score_hrt(hrt).squeeze(-1).cpu().numpy()

    val_scores = _score_edge_label_index(split.val[edge_type])
    val_labels = split.val[edge_type].edge_label.numpy()
    threshold = best_f1_threshold(val_labels, val_scores)

    test_scores = _score_edge_label_index(split.test[edge_type])
    test_labels = split.test[edge_type].edge_label.numpy()
    test_metrics = binary_link_metrics(test_labels, test_scores, threshold=threshold)

    return KGEResult(model_name=model_name, test_metrics=test_metrics, ranking_metrics=ranking_metrics)
