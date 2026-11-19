# Pre-registration of the v5 confirmatory round

Written on 19 November 2026, before any evaluation seed was run. The file is committed to the public repository before `scripts/47_v5_confirmatory.py` is run on seeds 10-19, so the commit time stamps it.

## What is already known (exploratory, v4)

Seeds 0-9 with settings tuned for link prediction (`docs/v4/FINDINGS.md`). On those seeds, RotatE reached hit@10 0.823, co-supply 0.635 and the tuned GNNs 0.45-0.52. The GNNs used a binary loss, one negative per positive and 100 full-batch steps; the embedding models used a softmax over hundreds of negatives and thousands of steps. The v4 result is therefore not evidence about architecture, and this round tests it with matched freedoms. On the tuning seed (100) only, an untuned GraphSAGE with a softmax loss and a 60-second NBFNet run were observed to reach hit@10 of about 0.74 and 0.81; no evaluation seed has been seen.

## Design

- **Task and metric**: the AnonNet substitution test, tau = 0.1, hit@10, as in the manuscript.
- **Evaluation seeds**: 10-19. No earlier run used them.
- **Tuning**: every learned ranker is tuned on substitution queries of scenario seed 100 (mean hit@10), 25 TPE trials each (NBFNet: a 4-point grid), with the number of negatives, batch size and epochs in every search space (`scripts/46`). Tuned settings are frozen in `configs/model_v5/` before the confirmatory run.
- **Compared methods**: co-supply; four GNN variants (GraphSAGE, GATv2, each with a binary and a softmax loss); KGE models RotatE, ComplEx, DistMult and TransE with a softmax loss; RotatE and ComplEx with a binary loss; NBFNet.
- **Selection rules** (fixed now): "the best GNN" is the GNN variant with the highest tuning-scenario hit@10; "the best KGE" is the softmax KGE with the highest tuning-scenario hit@10.
- **Test**: two-sided exact Wilcoxon signed-rank on the ten paired seed means, Holm-corrected across H1-H4.

## Hypotheses

- **H1** The best KGE has a higher hit@10 than co-supply.
- **H2** NBFNet has a higher hit@10 than co-supply.
- **H3** Loss matters: the best GNN variant has a higher hit@10 than the same encoder with the other loss.
- **H4** For the best GNN, the post-training shift reaches the learned penalty's risk (lambda = 1) with a higher hit@10 than the penalty (interpolated on each seed's shift curve).
- **H5 (guarantee, descriptive pass/fail)** For the best KGE, targets of 10% and 20% (alpha = (1 - t) x the ranker's plain risk@10 on the tuning seed), the conformal-risk-control rule satisfies mean test risk <= alpha + 0.001 averaged over 200 random calibration/test splits of the customers, in at least 9 of the 10 seeds. The uncorrected grid rule of v4 is reported alongside, with no hypothesis.

Not hypotheses, reported as exploratory: the margin between any two learned methods other than those named, the other targets (5%, 30%), runtime, and every sensitivity analysis.

## Deviations

Any departure from this plan will be listed in `docs/v5/FINDINGS.md` under "Deviations", with the reason.
