# v4 link-prediction benchmark: protocol and the defects it fixes

Written 17 November 2026, before any tuned v4 benchmark result existed.
Script: `scripts/41_v4_fair_benchmark.py`.

## Defects found in the v3 benchmark (the one behind the submitted manuscript)

1. **Tuning on the test split.**
   - `src/suschain/training/hparam_search.py`: the Optuna objective returns
     `result.test_ranking["mrr"]`, the seed-0 *test* MRR, although its comment
     says validation.
   - `scripts/07_multiseed_final.py::pick_config` then chose between the default
     and the tuned configuration by seed-0 test MRR. Seed 0 is one of the
     reported seeds.
2. **Reverse-edge leakage on AnonNet.**
   - `build_anonnet_heterodata(make_undirected=True)` folds b->a into the same
     relation as a->b, and the split (`is_undirected=False`, no reverse type)
     assigns the two directions independently.
   - Measured on seeds 0 and 1: 79-80% of test positives have their reverse
     edge in the training graph, and 56% in the message-passing edges.
   - The decoder is a dot product, so it scores a->b and b->a identically.
   - One run of default GraphSAGE, seed 0: the leaky split gives filtered
     MRR 0.0122 against 0.0046 on the leak-free split, and AUC 0.861 against 0.829.
3. **KGE baselines handicapped.**
   - The PyKEEN models were trained on the target relation only, while the
     GNNs pass messages over every relation, certificates included.
   - They were not tuned.
4. **Ranking details.**
   - Ties were counted in the model's favour, which inflated heuristic MRR
     (see `FINDINGS.md`, section 2).
   - The filter set used `split.train.edge_index`, which leaves out the
     training supervision edges of a disjoint split.
   - Accuracy was reported only at the F1-optimal threshold.

## v4 protocol (identical for every method)

- **Splits.** `split_link_prediction`: 80/10/10 split of the target relation,
  disjoint_train_ratio 0.3, one negative per positive in validation and test,
  seeds 0-9.
  - AnonNet uses `build_anonnet_directed_heterodata`, with the reverse relation
    `rev_supplies` passed as `rev_edge_type`. At test time no test positive's
    reverse is in message passing.
  - 12% of AnonNet test positives have a genuine reciprocal edge in training,
    because the raw data contains both a->b and b->a. These are data, not
    artefacts, and are kept.
- **Tuning.** For each (dataset, method): 25 TPE trials (Optuna, sampler
  seed 0) on the seed-0 split.
  - Objective: **validation** filtered MRR with realistic ties.
  - Trials run at the full training budget (GNN 100 epochs with patience 15;
    KGE up to 100 epochs, validation MRR checked every 5 epochs, stop after 4 checks without improvement), so there is no
    short-budget/full-budget mismatch.
  - Test data is never read during tuning.
- **Search spaces.**
  - GNN, as in v3: embedding {64, 128, 256}, hidden {64, 128, 256},
    out {32, 64, 128}, layers {2, 3}, lr [1e-4, 1e-2] log, weight decay
    [1e-6, 1e-3] log.
  - KGE: dim {64, 128, 256}, lr [1e-4, 1e-2] log, L2 [1e-8, 1e-2] log,
    batch {256, 512, 1024}, sampled negatives {256, 1024, 4096}, and
    gamma {3, 6, 9, 12} for TransE and RotatE.
- **KGE training.**
  - All forward relations of the training graph, with reciprocal triples.
  - Softmax cross-entropy against shared sampled negatives of the correct
    entity type.
  - Plain PyTorch implementation (`models/kge_torch.py`), with scoring
    functions unit-tested against direct formulas.
  - RotatE uses the L2 distance over the complex vector rather than the
    original sum of moduli. The exact original took about 30 times longer per
    epoch (585 s for 20 LODHalal epochs), which ruled out 25 trials.
- **Heuristics.** Popularity (training in-degree of the destination) on both
  datasets. Common Neighbours, Adamic-Adar and Preferential Attachment on
  AnonNet, over the symmetrised training graph.
- **Metrics.**
  - Filtered MRR and Hits@{1,3,10} over all destinations of the target type,
    with realistic ties; optimistic and pessimistic also saved.
  - AUC.
  - Accuracy and F1 at the F1-optimal validation threshold, and accuracy at
    the accuracy-optimal validation threshold.
  - Per-query ranks saved to `.npz`.
- **Leak measurement.** The tuned GNN configurations rerun on the old leaky
  AnonNet split, same seeds.
- **Negative-ratio sweep.** One split per seed with five negatives per
  positive. GNNs train on 1, 2 or 5 of them per positive and are evaluated at
  1, 2 and 5 test negatives per positive, with thresholds re-chosen on
  validation at the same ratio.
  - Prediction: AUC and MRR barely move, while accuracy and F1 move with the
    evaluation ratio. Accuracy and F1 numbers across papers that use different
    ratios (the base paper used 2:1) are therefore not comparable.

All comparisons from this benchmark are exploratory. The declared primary
family is in `primary_tests.md`.
