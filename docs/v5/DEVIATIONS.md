# Deviations from the pre-registration

Recorded as they happen. The pre-registration (`PREREGISTRATION.md`, commit 0c706e8) is unchanged.

1. **Tuning budget for the embedding models (19 Nov 2026, before any evaluation seed).**
   The pre-registration says 25 TPE trials per method. One embedding-model trial took 3 to 4.5 minutes on the shared 5 GB GPU, so 25 trials for six embedding configurations would have taken more than seven hours. The embedding models were tuned with **12 trials each**, and their search space was narrowed (batch size 512 or 1024; 30 or 60 epochs; 256 or 1024 shared negatives for the softmax loss). The GNN variants kept 25 trials, with the wider space in `scripts/46`. NBFNet used a four-point grid (dimension 32 or 64, 2 or 3 layers; 240 s each). The unequal trial counts favour the GNNs; the wall-clock time spent on each method is in the `trials` list of each `configs/model_v5/substitution_*.json`.
2. **Discarded first attempts.** Two earlier embedding-model tuning runs were stopped after a few trials for the reason above; their logs are kept as `logs/v5_tune_kge_first_attempt.log` and `logs/v5_tune_kge_second_attempt.log`. Their trials were not used.
3. **Tuning scenario.** Scenario seed 100 hides a random 10% of links of the same network that the evaluation seeds use. No hidden link of an evaluation seed is trained on, but the tuned settings reflect this network; this is stated in the paper as a limitation.
4. **NBFNet is our implementation** of the published idea (`src/suschain/models/nbfnet.py`), simplified to run on a 5 GB GPU; it is not the authors' code, and it is trained for a fixed wall-clock budget (600 s per seed).
5. **Not run: Neural Common Neighbor and BUDDY.** They score a pair from the neighbourhoods of both nodes. Co-supply, common neighbours and Adamic-Adar are in the study, and the question the models would answer (can message passing count common neighbours?) is addressed by NBFNet's path counting. Stated as a limitation.
