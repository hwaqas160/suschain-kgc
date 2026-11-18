# v4 findings (filled in as runs complete)

Every number here is read from a file in `results/metrics/` named beside it.

## 1. LODHalal disruption frontier (script 34, 10 seeds, k = 5)
File: `v4_disruption_frontier_lodhalal.json`

Full graph (certificate links present):

| Mechanism | Recall@5 | Mean risk@5 | Certified@5 |
|---|---|---|---|
| plain RGCN (lambda 0) | 0.5374 | 0.00347 | 0.9642 |
| learned penalty lambda 1 | 0.5385 | 0.00235 | 0.9662 |
| learned penalty lambda 5 | 0.5378 | 0.00054 | 0.9663 |
| learned, negatives only, lambda 1 | 0.5367 | 0.00348 | 0.9649 |
| inference-time shift beta 1 | 0.5371 | 0.00127 | 0.9671 |
| inference-time shift beta 3 | 0.5370 | 0.00055 | 0.9692 |
| post-hoc filter, pool 20 | 0.5153 | 0.00003 | 1.0000 |
| certificate rule | 0.9742 | 0.00242 | 0.9705 |
| certificate rule + filter, pool 10 | 0.9395 | 0.00000 | 1.0000 |
| popularity | 0.0836 | 0.00032 | 0.8000 |
| random | 0.0020 | 0.03310 | 0.9615 |

Readings
- The learned penalty reproduces the manuscript's ~33% risk reduction at no
  recall cost (exact Wilcoxon over 10 seeds, risk p = 0.002).
- The inference-time shift reaches the same or lower risk at the same recall
  without retraining. Learning adds nothing on this dataset.
- Applying the penalty to negatives only removes the risk reduction entirely:
  the effect comes from lowering risky destinations' scores across the board,
  as the first-order derivation predicts.
- Measured slope of (s_lambda - s_0) on risk: -0.31 at lambda 1, -0.56 at
  lambda 5 (prediction -1 and -5), R^2 about 0.05. Right sign, much smaller,
  dominated by run-to-run noise: a weakened, noisy version of the shift.
- The certificate rule plus a filter recovers 94% of disrupted manufacturers
  at zero risk and 100% certification. No learned method comes close.

Certificate-free graph: plain recall@5 0.0093, below popularity (0.0836).
Without certificate links a disrupted food is isolated and nothing is
learnable. LODHalal therefore cannot carry a learned-substitution claim.

## 2. Heuristic baselines with tie-aware ranks (script 39, 10 split seeds)
File: `v4_heuristics_ties.json`

| Heuristic | MRR, ties counted in its favour (as in manuscript) | MRR, realistic | Accuracy, F1 threshold | Accuracy, accuracy threshold |
|---|---|---|---|---|
| AnonNet Common Neighbors | 0.0473 | 0.0182 | 0.7983 | 0.7983 |
| AnonNet Adamic-Adar | 0.0427 | 0.0216 | 0.6738 | 0.7983 |
| AnonNet Preferential Attachment | 0.0409 | 0.0262 | 0.7755 | 0.8607 |
| LODHalal popularity | 0.0559 | 0.0558 | 0.7906 | 0.7956 |

The manuscript's statement that structural heuristics rival or surpass the GNNs
on AnonNet (GAT 0.029, GraphSAGE 0.022) does not survive realistic tie
handling and must be withdrawn. The 0.500 / 0.667 rows were produced by the
F1-optimal threshold selecting "everything positive".

## 3. AnonNet supplier substitution (script 36, tau = 0.1, k = 10, 10 seeds)
File: `v4_anonnet_substitution_tau0.1.json`. About 5,300 queries per seed
(about 900 customers); mean 1.35 hidden true substitutes per query; median
candidate pool 294. Risk is scored for 78.5% of firms; the rest are ranked at
the mean.

| Method | hit@10 | MRR | risk@10 |
|---|---|---|---|
| random | 0.079 | 0.039 | 0.159 |
| product similarity | 0.105 | 0.054 | 0.154 |
| popularity | 0.504 | 0.262 | 0.168 |
| co-supply (shared customers) | 0.635 | **0.386** | 0.165 |
| fused heuristic | 0.542 | 0.247 | 0.160 |
| GraphSAGE, plain | 0.494 | 0.238 | 0.163 |
| GAT, plain | 0.593 | 0.300 | 0.162 |
| fused GraphSAGE (RRF with co-supply) | 0.640 | 0.348 | 0.164 |
| fused GAT (RRF with co-supply) | **0.670** | 0.376 | 0.164 |
| GraphSAGE, learned penalty lambda 1 | 0.475 | 0.228 | 0.161 |
| GraphSAGE, shift beta 1 | 0.494 | 0.238 | 0.155 |
| GraphSAGE, shift beta 3 | 0.489 | 0.236 | 0.141 |
| GraphSAGE, shift beta 10 | 0.445 | 0.215 | 0.108 |
| GraphSAGE, filter pool 20 | 0.362 | 0.142 | 0.108 |

Primary family (declared in `primary_tests.md` before the full run; exact
Wilcoxon over 10 seeds, Holm-corrected; file `v4_analysis.json`):

| Test | Result | p (Holm) |
|---|---|---|
| P1 learned lambda 1 vs plain, risk@10 | lower by 0.0026 (95% CI -0.0028 to -0.0024) | 0.0098 |
| P2 learned lambda 1 vs plain, hit@10 | lower by 0.019 (CI -0.023 to -0.016) | 0.0098 |
| P3 learned vs shift at equal risk, hit@10 | learned **worse** by 0.019 | 0.0098 |
| P4 filter pool 20 vs plain, hit@10 | lower by 0.132 (CI -0.141 to -0.124) | 0.0098 |
| P5 fused GraphSAGE vs co-supply, hit@10 | +0.005 (CI -0.001 to +0.010) | 0.23, not significant |

The equal-risk comparison (exploratory) is negative at every lambda from 0.1 to 5
(p from 0.039 to 0.002): the shift beats the learned penalty at all six
penalty strengths.

Exploratory, not in the declared family:
- Fused GAT vs co-supply: hit@10 +0.035, 10 of 10 seeds, p = 0.002; hit@5
  +0.012, 10/10, p = 0.002; MRR **-0.010**, 1/10, p = 0.014. Fusion puts more
  true substitutes in the top 10 but places the first one lower.
- GAT alone vs co-supply: lower on every metric (hit@10 -0.042, 0/10).
- Learned penalty, GAT: lambda 1 gives hit@10 0.583 at risk 0.160, while
  the shift at beta 1 gives 0.590 at risk 0.154. Same direction as GraphSAGE.

## 4. Sensitivity to the similarity threshold tau (GraphSAGE, lambda 0 and 1)
Files: `v4_anonnet_substitution_tau{0.05,0.2,0.3}.json`

| tau | co-supply | GraphSAGE plain | fused GraphSAGE | learned lambda 1 (hit / risk) | shift beta 1 (hit / risk) |
|---|---|---|---|---|---|
| 0.05 | 0.629 | 0.484 | 0.632 | 0.465 / 0.161 | 0.484 / 0.155 |
| 0.10 | 0.635 | 0.494 | 0.640 | 0.475 / 0.161 | 0.494 / 0.155 |
| 0.20 | 0.668 | 0.534 | 0.680 | 0.516 / 0.159 | 0.534 / 0.153 |
| 0.30 | 0.709 | 0.600 | 0.730 | 0.580 / 0.153 | 0.599 / 0.148 |

The ordering of methods is the same at every tau. The fusion's margin over co-supply
grows with tau (+0.003 to +0.021); its significance was not tested beyond
tau 0.1.

## 5. Risk-proxy sensitivity (script 37, 10 seeds)
File: `v4_risk_sensitivity_anonnet.json`

Risk removed relative to plain GraphSAGE / hit@10 lost:

| Risk variant | shift beta 1 | shift beta 3 | learned lambda 1 | filter pool 20 |
|---|---|---|---|---|
| baseline | 5.2% / 0.000 | 13.7% / 0.005 | 1.6% / 0.019 | 33.8% / 0.132 |
| missing firms = riskiest | 31.2% / -0.004 | 52.1% / 0.013 | 7.7% / 0.041 | 60.8% / 0.108 |
| missing firms = safest | 10.7% / 0.003 | 28.7% / 0.016 | 2.9% / 0.016 | 57.4% / 0.160 |
| country term only | 4.1% / 0.000 | 11.5% / 0.009 | 1.8% / 0.029 | 23.9% / 0.110 |
| sector term only | 13.3% / 0.007 | 33.3% / 0.051 | 3.6% / 0.094 | 18.8% / 0.048 |
| 20 random sector multipliers | - | 11.9-15.5% / 0.002-0.007 | not retrained | 29.8-39.0% / - |

- In every structural variant the shift removes more risk than the learned
  penalty, for less recall lost. The main conclusion does not depend on
  how the proxy is built.
- The ordering of the fixed methods by risk does depend on it: Kendall tau
  against the baseline order falls to -0.73 when firms without a location are
  treated as the riskiest. No claim of the form "method X recommends
  greener suppliers" should be made from this proxy.

## 6. LODHalal node classification without certificate links (script 33)
File: `v4_nodeclf_lodhalal_no_cert.json`

| Encoder | macro-F1, full graph | macro-F1, no certificates | p |
|---|---|---|---|
| GraphSAGE | 0.354 | 0.363 | 0.43 |
| GAT | 0.625 | 0.627 | 1.00 |
| RGCN | 0.373 | 0.375 | 0.49 |

The leakage hypothesis is **not** supported: removing the certificate
relation does not change macro-F1. The earlier GAT result stands. Majority-class
accuracy is 0.962, so accuracy is uninformative and macro-F1 is the metric to report.

## 7. Cold-start control (script 40, embedding-table GraphSAGE, 10 seeds)
File: `v4_coldstart_control.json`

| Scoring of held-out foods | cold-start AUC |
|---|---|
| original (random-initialised, never-trained rows) | 0.882 |
| input rows set to zero (all held-out foods identical) | **0.907** |
| input rows re-drawn | 0.854 |
| encoded vectors replaced by their mean (all identical) | 0.589 |
| popularity | 0.865 |

All four differ from the original (p = 0.002). With zeroed inputs every
held-out food has the same representation, so its score carries no information
about the query, yet the AUC rises. The embedding-table model's cold-start
AUC is a destination prior that depends on an arbitrary constant vector,
not inductive generalisation. The manuscript's cold-start claim for this
model must be withdrawn. Its value above popularity says nothing about the query food.

## Gate A decision
1. **The method claim is withdrawn.** The training-time penalty lowers risk
   (P1), but at a recall cost (P2). The inference-time shift reaches equal risk at
   higher recall (P3) on AnonNet and LODHalal, under every risk-proxy variant,
   and at every tau. Retraining for sustainability is not justified here.
2. **Learning adds little over the best non-learned rule.** The declared test
   P5 is not significant. A modest exploratory claim holds: fusing GAT with
   co-supply lifts hit@10 by 3.5 points with a lower MRR. Report it as exploratory.
3. **Paper type: evaluation and benchmark.** Contributions:
   - a held-out supplier-substitution task with ground truth;
   - the shift-versus-penalty result, with its derivation;
   - corrections to the benchmark (the certificate shortcut, tie-aware ranks,
     threshold degeneracy, the cold-start artefact, the risk proxy's sensitivity
     to its construction).

## 8. Defects in the v3 link-prediction benchmark (found 17 November 2026)
Details and the corrected protocol are in `benchmark_protocol.md`.

1. **Tuning on test data.**
   - `training/hparam_search.py` optimised seed-0 *test* MRR.
   - `07_multiseed_final.py::pick_config` chose the configuration by seed-0
     test MRR, and seed 0 is a reported seed.
2. **Reverse-edge leakage on AnonNet.** On seeds 0 and 1, 79-80% of test
   positives had their reverse in training, and 56% in message passing.
   - Default GraphSAGE, seed 0: MRR 0.0122 on the leaky split against 0.0046
     on the leak-free split; AUC 0.861 against 0.829.
   - Ten-seed tuned comparison: `v4_benchmark_leaky_anonnet_leaky.json`
     against `v4_benchmark_final_anonnet.json`.
3. **KGE baselines handicapped.** They had only the target relation and no
   tuning.
   - Smoke run, untuned, 20 epochs, all relations, seed 0. LODHalal:
     TransE MRR 0.884, DistMult 0.631, against default GraphSAGE 0.325.
     AnonNet (leak-free): DistMult 0.066, TransE 0.061, against
     GraphSAGE 0.005.
   - Tuned ten-seed numbers: `v4_benchmark_final_*.json`.
4. **Ranking filter incomplete.** `split.train.edge_index` left out the
   training supervision edges. Fixed in `eval/ranking_v4.py`. Old scripts are
   unchanged, so their results still reproduce.

## 9. Tuned rankers in the substitution task (script 43)
Smoke run, seed 0 only:
- GraphSAGE tuned on link-prediction validation MRR scores **lower** on
  substitution than the default configuration (hit@10 0.441 against 0.502).
  Link-prediction tuning does not transfer to substitution.
- The shift again dominates the learned penalty: beta 1 gives 0.436 at risk
  0.145, lambda 1 gives 0.371 at 0.159.
- Ten-seed results: `v4_substitution_tuned_tau0.1.json`.

## 10. Replication of Peeris et al. in their own setting (script 44)
Setting, from their first author's answers of 8 October 2026:
- 2:1 negatives;
- reported metrics are validation-split metrics;
- LODHalal with the food-manufacturer relation only;
- AnonNet as the 3,813-node product-location graph (the edges are undocumented,
  so two readings are run).

Smoke run, seed 0:
- LODHalal, validation split: MRR over the full catalogue 0.047 (theirs
  0.042-0.057, a match). Hits@10 is 0.10 over the full catalogue and 0.57 over
  99 sampled negatives; theirs is 0.35-0.43, between the two. Accuracy at 0.5
  is 0.89 and F1 0.83, against theirs of 0.67-0.71 and 0.14-0.20, so their
  classification results are not reproduced.
- AnonNet, neither reading reproduces their numbers. Closest: the supply
  reading, ranking products over 99 sampled negatives, Hits@10 0.49-0.53
  against theirs of 0.40-0.42.
- **The RGCN-style encoder equals GraphSAGE on single-relation graphs.**
  - GraphConv with mean aggregation computes the same function as SAGEConv
    (mean), and the summation across relations only matters when a node type
    receives more than one relation. With the same seed the two give identical
    numbers on all three replication graphs.
  - Textbook R-GCN with one relation reduces to the same form. Peeris et al.'s
    different RGCN and GraphSAGE numbers on a single-relation graph must
    therefore come from configuration or training, not architecture.
  - The manuscript must state that the RGCN-style encoder differs from
    GraphSAGE only through summed per-relation messages on multi-relation
    graphs.

Ten-seed results: `v4_peeris_replication.json`.

## 11. Corrected benchmark: AnonNet results (ten seeds, final phase, 17 Nov 2026)

Source: `results/metrics/v4_benchmark_final_anonnet.json`. Directed split with
no leak; filtered ranks, ties sharing a rank (realistic).

| Method | MRR | Hits@10 | AUC |
|---|---|---|---|
| Popularity | 0.0508 ± 0.0012 | 0.080 | 0.790 |
| Common Neighbours | 0.0210 | 0.053 | 0.808 |
| Adamic–Adar | 0.0238 | 0.062 | 0.809 |
| Preferential Attachment | 0.0182 | 0.046 | 0.895 |
| GraphSAGE (tuned) | 0.0144 ± 0.0074 | 0.027 | 0.850 |
| GATv2 (tuned) | 0.0140 ± 0.0059 | 0.028 | 0.871 |
| RGCN-style (tuned) | 0.0087 ± 0.0096 | 0.017 | 0.771 |
| TransE | 0.0707 ± 0.0017 | 0.203 | 0.924 |
| DistMult | 0.0946 ± 0.0028 | 0.198 | 0.935 |
| ComplEx | 0.1350 ± 0.0045 | 0.250 | 0.930 |
| RotatE (L2 variant) | 0.1584 ± 0.0039 | 0.287 | 0.932 |

- Every KGE model beats every GNN and every heuristic. The tuned GNNs rank
  below popularity: the v3 conclusion that "GNNs beat KGE" is reversed.
- Leak, ten seeds, tuned settings (`v4_benchmark_leaky_anonnet_leaky.json`):
  MRR leaky against clean was 0.0159/0.0144 (sage), 0.0112/0.0140 (gat) and
  0.0167/0.0087 (rgcn). AUC was 0.870/0.850, 0.874/0.871 and 0.869/0.771.
  The effect is mixed and within about one standard deviation. It is much
  smaller than the single-seed default-setting effect (0.0122/0.0046).

## 12. Final ten-seed results (18 Nov 2026)

**LODHalal benchmark** (`v4_benchmark_final_lodhalal.json`):
- MRR: KGE 0.967–0.971; GraphSAGE 0.417; RGCN-style 0.463; GATv2 0.105;
  popularity 0.056.
- AUC is 0.978–0.997 for every learned model, so AUC hides the gaps.
- The KGE models exploit the certificate shortcut fully.

**Negative-ratio sweep** (`v4_benchmark_sweep_*.json`, GraphSAGE):
- AnonNet trained at 1:1: AUC stays at 0.850 for test ratios 1, 2 and 5,
  while accuracy at 0.5 goes 0.673 → 0.593 → 0.515 and F1 goes
  0.735 → 0.598 → 0.384.
- Training ratio 1 → 5: LODHalal MRR 0.398 → 0.596; AnonNet MRR
  0.014 → 0.001.

**Substitution with tuned rankers** (`v4_substitution_tuned_tau0.1.json`;
same queries as co-supply, which scores 0.635; exploratory):
- KGE hit@10: TransE 0.770, DistMult 0.791, ComplEx 0.787, RotatE 0.823
  (MRR 0.552). All beat co-supply on 10/10 seeds, p = 0.002.
- Fusing a KGE with co-supply lowers it (RotatE 0.732).
- Tuned GNNs are worse than default: sage 0.448, gat 0.517. Link-prediction
  tuning does not transfer.
- Penalty against shift with tuned GNNs:
  - sage λ1 0.379 / 0.164 vs shift β1 0.444 / 0.142;
  - gat 0.480 / 0.163 vs 0.515 / 0.152.
- RotatE shift β3: risk 0.165 → 0.147 (−11%) for −0.2 points.

**Risk budget** (seed means; achieved reduction / hit points lost):
- Calibration met or exceeded every target on held-out customers.
- At 10%: co-supply 10.6% / 0.8; sage 11.5% / 0.4; gat 11.7% / 0.5.
- At 20%: 21.9% / 3.1; 22.4% / 1.6; 24.5% / 2.6.
- KGE budgets: `v4_substitution_tuned_tau0.1_kgebudget.json`.

**Replication** (`v4_peeris_replication.json`, validation split):
- LODHalal full-catalogue MRR 0.033–0.050 vs reported 0.042–0.057.
- Accuracy / F1 0.89 / 0.82 vs reported 0.67–0.71 / 0.14–0.20.
- AnonNet readings: MRR ≈ 0.40 vs reported 0.044–0.048. Not reproduced.
- On LODHalal food–manufacturer, sage and rgcn are identical, as expected.

**Message change:** the choice of ranker matters more than the penalty. KGE
models are the strongest rankers; the shift and the risk budget apply on top.

**RotatE risk budget** (`v4_substitution_tuned_tau0.1_kgebudget.json`; RotatE
only; hit@10 0.823 reproduces the main run): achieved reduction / points
lost at targets 5/10/20/30% = 6.4/0.0, 12.1/0.4, 21.7/2.5, 31.4/6.2.

Across six rankers, the mean met every target. Per seed, 224 of 240 cells
(ranker × target × seed) met the target. The worst cell was 29.1% against a
30% target (tuned sage).
