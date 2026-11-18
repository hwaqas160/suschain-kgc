# v4 revision: scripts, outputs and order

Implements `Revision_v4/02_IMPLEMENTATION_PLAN.md`. New result files carry a
`v4_` prefix; no earlier result file is overwritten.

| Script | Work package | What it does | Output |
|---|---|---|---|
| 33_v4_nodeclf_no_cert.py | 1 | LODHalal node classification, full vs certificate-free graph, 10 seeds | v4_nodeclf_lodhalal_no_cert.json |
| 34_v4_disruption_frontier.py | 1, 2 | LODHalal disruption: learned penalty, inference-time shift, filters, certificate rule; full and certificate-free graphs; 10 seeds | v4_disruption_frontier_lodhalal.{json,npz} |
| 35_v4_analysis.py | 2, 7 | Declared primary tests (Holm), clustered bootstrap, equal-risk comparison, figures | v4_analysis.json, results/figures/v4_frontier_*.pdf |
| 36_v4_anonnet_substitution.py | 3, 4 | AnonNet substitution with held-out ground truth; heuristics, GNNs, fusion, sustainability mechanisms | v4_anonnet_substitution_tau{tau}.{json,npz} |
| 37_v4_risk_sensitivity.py | 4 | Missing-risk policies, country-only, sector-only, 20 multiplier perturbations | v4_risk_sensitivity_anonnet.json |
| 39_v4_heuristics_ties.py | 8 | Heuristics with optimistic / realistic / pessimistic tie ranks and accuracy-optimal thresholds | v4_heuristics_ties.json |
| 40_v4_coldstart_control.py | 8 | Is the cold-start margin query information or a learned prior? | v4_coldstart_control.json |
| v4_queue.sh | - | Runs the GPU-bound scripts in order | logs/ |
| 41_v4_fair_benchmark.py | 8 | Corrected link-prediction benchmark: equal-budget Optuna tuning on validation MRR, leak-free directed AnonNet, KGE models with reciprocal relations, realistic tie ranks; phases tune/final/leaky/sweep | configs/model_v4/*.json, v4_benchmark_{phase}_{dataset}.{json,npz} |
| 42_v4_paper_figures.py | 10 | Risk-accuracy trade-off figure | results/figures/v4_fig_tradeoff.{pdf,png} |
| 43_v4_substitution_tuned.py | 8, 11 | Substitution with tuned GNNs and KGE models; risk-budgeted re-ranking (calibrate on half the customers, evaluate on the other half) | v4_substitution_tuned_tau0.1{tag}.json |
| 44_v4_peeris_replication.py | 9 | Replication of Peeris et al. in their stated setting (2:1 negatives, validation-split metrics) | v4_peeris_replication.json |
| 45_v4_paper_tables.py | 10 | Writes the manuscript's LaTeX tables from the result files | Revision_v4/v5/tables/*.tex |
| v4_queue2.sh | - | Runs script 41 phases in order | logs/ |

Library additions
- `data/graph_builder.py`: `relations=` presets `full`, `no_cert`, `food_manufacturer_only`.
- `decision_eval/scenario_trainer.py`: the disruption trainer (from script 20), with
  `penalty_on` ("all" or "negatives") and `mp_edge_index` (needed for AnonNet).
- `decision_eval/disruption_simulation.py`: `per_query_outcomes`, `recommend_score_penalty`,
  `cert_cooccurrence_scores`.
- `decision_eval/substitutes.py`: the substitution task.
- `eval/significance.py`: `exact_wilcoxon`, `paired_bootstrap_ci` (optionally clustered),
  `holm_correction`.
- `eval/metrics.py`: `best_accuracy_threshold`, `ranks_with_ties`.
- `training/train_inductive.py`: optional `coldstart_controls`.

Notes
- `docs/v4/primary_tests.md` declares the primary test family; it was written
  before the full runs finished.
- `docs/v4/penalty_derivation.md` derives the first-order equivalence between the
  training-time penalty and an inference-time logit shift.
- `docs/v4/anonnet_edge_direction.md` records the evidence for reading AnonNet's
  `source` as the supplier.
- GPU runs are not bit-for-bit deterministic: script 34 reproduces script 20's
  seed-0 risk exactly and its recall to within 0.0009 (about 4 of 4,310 queries).
