# Primary tests, declared before the full v4 runs finished

Written 17 November 2026, while scripts 34 and 36 were running. At that point
I had seen only the smoke runs: seed 0 of the AnonNet substitution task
(GraphSAGE, lambda 0 and 1) and seeds 0-2 of the LODHalal frontier log. The
family below follows the implementation plan (02_IMPLEMENTATION_PLAN.md,
work package 7), which was written before any v4 run.

Primary dataset: AnonNet substitution task, tau = 0.1, k = 10, GraphSAGE,
10 seeds. Each test is an exact two-sided Wilcoxon signed-rank test on the 10
paired per-seed means; the five p-values are Holm-corrected as one family.

| # | Comparison | Metric | Question |
|---|---|---|---|
| P1 | learned penalty (lambda = 1) vs plain (lambda = 0) | risk@10 | Does the training-time penalty lower recommended-set risk? |
| P2 | learned penalty (lambda = 1) vs plain (lambda = 0) | hit@10 | What does it cost in substitute recall? |
| P3 | learned penalty (lambda = 1) vs inference-time shift at the same risk | hit@10 | Is training-time learning better than a score shift that needs no retraining? |
| P4 | post-hoc filter (pool 20) vs plain | hit@10 | What does filtering cost in substitute recall? |
| P5 | best learned ranking (fused_sage_lam0) vs best heuristic (co_supply) | hit@10 | Does learning add anything over the strongest non-learned rule? |

P3 detail: for each seed, the shift curve (beta = 0 and the six beta values)
is linearly interpolated in risk@10, and its hit@10 is read off at the learned
model's risk@10. If the learned risk lies outside the curve's range, that seed
is reported as out of range and the test is run on the remaining seeds.

Every other comparison in the v4 results is exploratory and is labelled so.
Intervals: 95% percentile bootstrap, resampling customers within seeds
(10,000 resamples).
