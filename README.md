# suschain-kgc

Code, configurations and per-seed results for a study of sustainability-aware graph learning for
supplier substitution. The manuscript is under double-anonymised review; author details will be added
after the review.

The study asks whether a sustainability preference should be learned in the training loss of a
supplier-ranking model or applied to its scores afterwards. It builds on Peeris, Baryannis &
Papadakis (2026), *Sustainability* 18(6), 2825 (<https://doi.org/10.3390/su18062825>), who rank
candidate suppliers with graph neural networks and propose building sustainability objectives into
the training loss as future work.

## Current version (v5, November 2026)

Everything reported in the current manuscript comes from the v5 scripts below. Results are
summarised in `docs/v5/FINDINGS.md`.

- **Pre-registrations.** `docs/v5/PREREGISTRATION.md` (commit 0c706e8) fixes the hypotheses,
  selection rules and analysis of the confirmatory round before any evaluation seed was run.
  `docs/v5/PREREGISTRATION_TRASE.md` (commit da186fa) does the same for the Trase study.
  Deviations are listed in `docs/v5/DEVIATIONS.md`.
- **Tuning on substitution queries** (`scripts/46_v5_tune_substitution.py`), scenario seed 100.
  Frozen settings: `configs/model_v5/` (commit 12d7a9c).
- **Confirmatory round** on AnonNet, seeds 10 to 19 (`scripts/47_v5_confirmatory.py`) and its
  pre-registered analysis (`scripts/48_v5_analysis.py`, `scripts/54_v5_h5_penalty.py`).
- **Observed replacements on Trase Brazil soy** (`scripts/50_v5_trase.py`).
- **Exploratory checks added after the confirmatory run:** reverse-link exposure
  (`scripts/55_v5_reverse_exposure.py`), the size of the learned penalty relative to the base
  loss (`scripts/56_v5_penalty_scale.py`) and decision-time risk weights with an importer-level
  Trase check (`scripts/57_v5_control_modes.py`).
- **Figures and tables** of the manuscript: `scripts/52_v5_schematics.py`,
  `scripts/53_v5_results_figures.py`.

Result files: `results/metrics/v5_*.json` and `v5_*_curves.npz` (per-query outcomes).

### Trase data

The Trase Brazil soy supply chain, release 2.6.1 (CC BY 4.0), is not redistributed here because of
its size. Download the composite table from
<https://trase.earth/open-data/datasets/supply-chains-brazil-soy> and save it as
`data/external/brazil_soy_v2_6_1_composite.csv` before running `scripts/50_v5_trase.py`.

## Earlier versions

The v4 scripts (`scripts/33` to `45`, `docs/v4/`) are the first, exploratory round of the current
study and the corrected link-prediction benchmark (`scripts/41`, `docs/v4/benchmark_protocol.md`).
Where they differ from v5, v5 applies.

Scripts `01` to `25` belong to an earlier, unpublished version of this study. Its benchmark had
several problems that are documented in `docs/v4/FINDINGS.md` (section 8) and corrected in v4: test
data used in tuning, reverse-edge leakage on AnonNet, embedding baselines trained on one relation
and tied scores counted in a method's favour. Its conclusions about the multi-objective loss are
superseded. These scripts are kept so that every earlier number can still be reproduced.

## Datasets

- **LODHalal** (`data/raw/lodhalal/`): food-manufacturer-ingredient-
  certification knowledge graph, sourced from
  <https://github.com/utomogirraz/graph-app-kit> (as used by Peeris et al.).
- **AnonNet** (`data/raw/anonnet/`): anonymised manufacturing supply
  network, sourced from <https://github.com/lx249/anon-network> (Xu, Long &
  Brintrup, 2025).

Both are used exactly as published by their original sources, with data
quality issues found during verification documented in the code (see
`src/suschain/data/`) and in the manuscript.
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
