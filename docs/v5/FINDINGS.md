# v5 findings

Every number here comes from a result file named in its section. Pre-registered tests are marked; everything else is exploratory.

## 1. Tuning scenario (AnonNet, seed 100)

Source: `configs/model_v5/substitution_*.json` (frozen at public commit 12d7a9c) and `results/metrics/v5_confirmatory_tuning.json`. Rerunning seed 100 with the frozen settings reproduces the embedding models' tuning scores exactly and the GNN and NBFNet scores within 0.0035 (the table gives the rerun).

| Ranker | Tuning hit@10 |
|---|---|
| NBFNet | 0.835 |
| RotatE, softmax | 0.826 |
| RotatE, BCE | 0.809 |
| TransE | 0.796 |
| GATv2, softmax | 0.783 |
| GraphSAGE, softmax | 0.783 |
| DistMult | 0.757 |
| ComplEx, softmax | 0.736 |
| ComplEx, BCE | 0.704 |
| GraphSAGE, BCE | 0.680 |
| GATv2, BCE | 0.634 |
| Co-supply | 0.632 |

- The pre-registered picks are GATv2 with softmax (best GNN) and RotatE with softmax (best embedding model).
- For the GNNs the loss matters a great deal (softmax beats BCE by 0.10–0.15). For the embedding models it matters little.
- The v4 claim that "GNNs lose" was a loss/negatives confound.

## 2. Confirmatory round (AnonNet, seeds 10–19)

Sources: `results/metrics/v5_confirmatory.json`, `v5_analysis.json`, `v5_h5_penalty.json`. About 5,300 queries from about 900 customers per seed.

| Ranker | hit@10 | MRR | risk@10 | train s |
|---|---|---|---|---|
| Popularity | 0.506 | 0.265 | 0.167 | – |
| Co-supply | 0.642 | 0.391 | 0.164 | – |
| GraphSAGE, BCE | 0.695 | 0.385 | 0.164 | 78 |
| GraphSAGE, softmax | 0.761 | 0.463 | 0.166 | 48 |
| GATv2, BCE | 0.638 | 0.344 | 0.163 | 481 |
| GATv2, softmax (best GNN) | 0.769 | 0.473 | 0.166 | 226 |
| NBFNet | 0.832 | 0.568 | 0.165 | 609 |
| TransE | 0.786 | 0.506 | 0.167 | 62 |
| DistMult | 0.760 | 0.444 | 0.165 | 31 |
| ComplEx, BCE / softmax | 0.703 / 0.744 | 0.418 / 0.446 | 0.164 / 0.165 | 71 / 33 |
| RotatE, BCE / softmax (best KGE) | 0.802 / 0.824 | 0.499 / 0.562 | 0.164 / 0.165 | 101 / 120 |
| GATv2 softmax + penalty λ=1 / λ=5 | 0.769 / 0.758 | 0.474 / 0.467 | 0.166 / 0.165 | 235 / 238 |

Pre-registered tests (exact Wilcoxon, Holm over H1–H4):
- **H1** RotatE vs co-supply: +18.2 points, 10/10 seeds, p = 0.002, Holm p = 0.008. **Supported.**
- **H2** NBFNet vs co-supply: +18.9 points, 10/10 seeds, Holm p = 0.008. **Supported.**
- **H3** GATv2 softmax vs GATv2 BCE: +13.1 points, 10/10 seeds, Holm p = 0.008. **Supported.**
- **H4** Shift vs penalty at equal risk, λ = 1: mean −0.1 points, 5/10 seeds, p = 0.43. **Not supported.** At λ = 1 the penalty changed neither risk (0.1660 vs 0.1660) nor hit@10 (0.7695 vs 0.7690).
- **H5** Guaranteed rule, RotatE, half calibration: within α + 0.001 on 10/10 seeds at both the 10% and 20% targets (gaps −0.005 to −0.011). **Passed.**

Exploratory results:
- λ = 5 removes 0.8% of risk for 1.1 points of hit@10. The shift at equal risk keeps 1.1 more points (8/10 seeds, p = 0.0098). At the same hit cost the shift removes 12.5% of risk.
- NBFNet vs RotatE: +0.8 points, 8/10 seeds, p = 0.027. RotatE vs the best GNN: +5.5 points, 10/10 seeds.
- Loss gaps: GraphSAGE 6.6, ComplEx 4.1, RotatE 2.2 points, each 10/10 seeds. GraphSAGE with BCE beats co-supply by 5.3 points; GATv2 with BCE does not (−0.4).
- Risk budget: every ranker meets every target on average under both rules at every calibration size.
  - With 25 calibration customers at the 20% target, the grid rule stays within α in 71–78% of splits; the guaranteed rule in 100%, at a cost of 20–27 points.
  - With half the customers, the guaranteed rule costs 5–7 points and the grid rule 5–6.
  - At the 10% target, RotatE costs 1.6 points (guaranteed rule) and 1.2 points (grid rule).
  - The 20% target lands near β = 3 (20% reduction for 1.9 points), so the coarse grid often jumps to β = 5 (29% for 5.4 points).
- Lemma 1 (monotonicity): 0 violations in 11,162,340 comparisons.

## 3. Trase Brazil soy (observed replacements)

Sources: `results/metrics/v5_trase.json`, `v5_trase_analysis.json` and `v5_trase_curves.npz`. Pre-registration: `PREREGISTRATION_TRASE.md` (commit da186fa).

- **Data.** 2,708 importers, 1,979 exporters and 10,864 pairs, after dropping unknown importers (30% of rows) and unknown exporters (3%).
- **Queries.** Each evaluation year has 242–482 queries from 24–122 importers. The median is 251 candidates per query, with 10.6 correct answers on average.

| Ranker | hit@10 | MRR | risk@10 | Years above co-supply |
|---|---|---|---|---|
| Random | 0.383 | 0.148 | 0.125 | 1/7 |
| Popularity | 0.551 | 0.212 | 0.073 | 2/7 |
| Co-supply | 0.616 | 0.317 | 0.071 | – |
| GraphSAGE, BCE | 0.651 | 0.349 | 0.115 | 4/7 |
| GraphSAGE, softmax | 0.617 | 0.300 | 0.078 | 4/7 |
| RotatE, softmax (best, chosen on 2016) | 0.738 | 0.411 | 0.084 | 7/7 |
| ComplEx, softmax | 0.537 | 0.243 | 0.089 | 1/7 |
| NBFNet | 0.716 | 0.367 | 0.077 | 5/7 |

Pre-registered tests:
- **T1** (RotatE vs co-supply): p = 0.0156, Holm p = 0.031. **Supported.**
- **T2** (NBFNet vs co-supply): p = 0.078, Holm p = 0.078. **Not supported.**
- **T3** (guaranteed rule, 20% target, α = 0.8 × 0.0697): mean test risk ≤ α in 7 of 7 years, with gaps from −0.021 to −0.051. **Passed.** The 10% and 30% targets also hold in 7 of 7 years.

Exploratory results for RotatE on half splits with the 20% target:
- **Guaranteed rule:** stays within α in 100% of splits but costs 9.2 hit@10 points (mean β 8.7). The target is certifiable on the grid in only 83% of splits; in the rest the rule falls back to β = 20. The calibration halves hold about 12–61 importers, so the 1/(n+1) term is comparable to α.
- **Grid rule:** stays within α in only 71% of splits, at a cost of 1.5 points.

Exploratory shift results for RotatE (seed 0, mean over years):
- At β = 0.5, risk 0.064, below co-supply's 0.071, with no hit loss (0.733).
- At β = 1, risk 0.051 (−42%) and hit 0.689, still above co-supply's 0.616.

Also exploratory: on Trase, GraphSAGE does better with BCE than with softmax, the opposite of AnonNet. The best loss depends on the data.

## 6. Checks added after the confirmatory run (exploratory, 20 Nov 2026)

**Reverse-link exposure** (`scripts/55_v5_reverse_exposure.py`, `v5_reverse_exposure.json`). In 14% of queries (13.0-16.6% per seed) a correct answer has a visible link from the customer. On the remaining queries, hit@10 is: co-supply 0.670, RotatE 0.822, NBFNet 0.819, GATv2 softmax 0.762, GraphSAGE softmax 0.756. Differences against co-supply: RotatE +15.3, NBFNet +14.9, GATv2 +9.2, GraphSAGE +8.7; GATv2 softmax against binary +11.4 (all 10/10 seeds, p = 0.002). NBFNet against RotatE is -0.3 (2/10, p = 0.16). On the exposed queries: NBFNet 0.909, RotatE 0.835, co-supply 0.475.

**Penalty size** (`scripts/56_v5_penalty_scale.py`, `v5_penalty_scale.json`, first 60 steps on seed 100). At lambda = 1 the penalty is 1.2-2.2% of the base loss under the softmax loss (GATv2) against 11.7-16.8% under BCE. Its gradient is 0.2-3.1% of the base loss's gradient against 8.7-17.8%. lambda is not comparable across losses.
