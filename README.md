# suschain-kgc

Reproducible re-benchmark and extension of Peeris, Baryannis & Papadakis
(2026), *Sustainability* 18(6), 2825, "Advancing Sustainable Supply Chains
Through Knowledge Graph Completion and Graph-Based Artificial Intelligence"
(<https://doi.org/10.3390/su18062825>).

The base paper proposes graph neural networks (GraphSAGE, GAT, RGCN) for
supplier-substitution link prediction and sustainability-risk node
classification, evaluated on two supply-chain datasets against synthetic
labels, with no released code and an undisclosed evaluation protocol. This
repository rebuilds that pipeline from scratch and closes the gaps that
follow from an undisclosed protocol: no non-learned baselines, no
knowledge-graph-embedding (KGE) baselines, no stated negative-sampling ratio
or candidate-pool size, and no statistical testing between architectures.

## v4 revision (November 2026): current results

The v4 scripts (`scripts/33`–`45`, `docs/v4/`) support the revised manuscript
"Retrain or re-rank? An evaluation of sustainability-aware graph learning for
supplier substitution under disruption". Every result uses ten seeds. Where
the v4 findings differ from the earlier sections below, the v4 findings
apply. They are summarised in `docs/v4/FINDINGS.md`.

- **Substitution test with held-out ground truth** on AnonNet
  (`scripts/36`, `43`). The failed supplier is never a correct answer.
- **The sustainability penalty in the loss behaves like a score shift**, to
  first order (`docs/v4/penalty_derivation.md`). Applied after training, the
  shift beats the trained penalty at every setting (`scripts/34`–`37`).
- **The choice of ranker matters more than the penalty.** Top-10 hit rate:
  RotatE 0.823; the co-supply rule 0.635; the GNNs 0.45–0.59.
- **Risk-budgeted re-ranking** (`scripts/43`) calibrates the shift to a
  target risk reduction on half the customers and checks it on the other half.
- **Corrected link-prediction benchmark** (`scripts/41`,
  `docs/v4/benchmark_protocol.md`). It uses equal-budget tuning on validation
  data only, a leak-free directed AnonNet split, KGE models with reciprocal
  relations, and tied scores sharing ranks. KGE models lead on both datasets.
- **Replication of the base paper's setting** (`scripts/44`), following the
  first author's description of it.
- The manuscript's tables are generated from the result files by
  `scripts/45_v4_paper_tables.py`.

The earlier claims about the multi-objective loss and the cold-start result
are withdrawn in v4. See `docs/v4/FINDINGS.md`: section 7 and the "Gate A decision" section.

## What this adds over the base paper (v1–v3)

- A disclosed benchmark: heuristic floors (popularity, common neighbors,
  Adamic-Adar, preferential attachment), four KGE models (TransE, DistMult,
  ComplEx, RotatE via PyKEEN), a dual-reported ranking protocol
  (full-catalog filtered and 99-candidate sampled-pool), per-architecture
  hyperparameter tuning, and seven-seed Wilcoxon significance testing.
- Real sustainability grounding: the base paper's synthetic emission labels
  are replaced with cited external data (Our World in Data CO2-per-capita)
  where matching is reliable, and with the datasets' own certification
  records where external matching is measured and found inadequate.
- Sustainability folded into training via a multi-objective loss, evaluated
  both as a Pareto utility-risk tradeoff and as a supplier-disruption
  simulation reporting managerial outcomes (substitute recall, recommended
  risk, certification compliance), all at seven seeds.
- The first genuine inductive/cold-start test for this model class, with a
  destination-popularity control.
- A zero-parameter certificate-co-occurrence baseline for LODHalal link
  prediction that outperforms every learned model tested (see
  `results/metrics/phase1_cert_cooccurrence_full_lodhalal.json` and
  `scripts/22`-`25`): 99.0% of LODHalal certificates that connect more than
  one food connect foods sharing a single manufacturer, so the certificate
  functions as a near-lookup for the prediction target. This is the paper's
  central methodological finding, applied to its own results rather than
  only the base paper's.

The manuscript itself is not part of this repository; see the citation
metadata in `CITATION.cff` and `.zenodo.json` for how to cite this code and
the associated paper once published.

## Datasets

- **LODHalal** (`data/raw/lodhalal/`): food-manufacturer-ingredient-
  certification knowledge graph, sourced from
  <https://github.com/utomogirraz/graph-app-kit> (cited by the base paper).
- **AnonNet** (`data/raw/anonnet/`): anonymised manufacturing supply
  network, sourced from <https://github.com/lx249/anon-network> (Xu, Long &
  Brintrup, 2025).

Both are used exactly as published by their original sources, with data
quality issues found during verification documented in the code (see
`src/suschain/data/`) and in the paper's Materials and Methods section.
`data/processed/lodhalal_food_off_matched.parquet` and
`data/external/owid_co2_data.csv` are the enriched real-sustainability-label
artifacts described above. The Open Food Facts product dump and the
sentence-transformer embedding cache used by `scripts/08` are not included
(1.2 GB and regenerable); running that script rebuilds them from the public
Open Food Facts data export.

## Setup

```bash
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Developed against Python 3.12, PyTorch 2.5.1, PyTorch Geometric 2.6.1 and
CUDA 12.4. A GPU is not required but the multi-seed scripts are slow on CPU.

## Pipeline

Scripts in `scripts/` are numbered in the order they were run to produce
every table and figure in the paper. Each writes its results as JSON (or
CSV, for the consolidated tables) into `results/`, so every reported number
traces back to a specific script, configuration and random seed.

| Range | What it produces |
|---|---|
| `01`-`02` | Descriptive statistics; baseline reproduction |
| `03`-`04` | Heuristic and KGE baselines |
| `05`-`06` | Decoder ablation; hyperparameter search |
| `07` | Seven-seed GNN benchmark (Tables 2-3) |
| `08`-`09` | Sustainability grounding; node classification (Table 4) |
| `10`-`12`, `18`-`19` | Multi-objective lambda sweep, single- and seven-seed (Figure 9) |
| `13` | Inductive/cold-start evaluation (Table 6) |
| `14`, `20`-`21` | Disruption simulation, single- and seven-seed (Table 5) |
| `15`-`17` | Result consolidation; figure generation; qualitative visualisations |
| `22`-`25` | Certificate-co-occurrence shortcut baseline and its ablations |

`results/metrics/` holds one JSON file per experiment; `results/tables/`
holds the consolidated CSVs used to build the manuscript's tables.

## Tests

```bash
pytest tests/
```

## License

CC BY 4.0. See `CITATION.cff` for how to cite this work.
