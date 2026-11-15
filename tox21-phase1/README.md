# Tox21 Molecular Toxicity Prediction

This repository contains the Phase 1 classical baselines and the Phase 2 deep-learning follow-on for predicting activity across the 12 Tox21 toxicity assays. Phase 1 compares Logistic Regression, Random Forest, and XGBoost using Morgan fingerprints and RDKit descriptors. Phase 2 adds graph neural networks and fingerprint MLPs under the same missing-label and split contracts.

## Results at a glance

On the scaffold test split, Random Forest with Morgan fingerprints plus descriptors had the highest mean scores in these runs: **ROC AUC 0.776 ± 0.002** and **PR AUC 0.404 ± 0.004**. XGBoost with the same features was close on PR AUC at **0.401 ± 0.005**. These are means across the 12 endpoints, with standard deviations across three model seeds.

| Model | Features | Scaffold ROC AUC | Scaffold PR AUC |
|---|---|---:|---:|
| Logistic Regression | Morgan | 0.716 ± 0.000 | 0.312 ± 0.000 |
| Logistic Regression | Morgan + descriptors | 0.732 ± 0.000 | 0.312 ± 0.000 |
| Random Forest | Morgan | 0.729 ± 0.006 | 0.370 ± 0.002 |
| **Random Forest** | **Morgan + descriptors** | **0.776 ± 0.002** | **0.404 ± 0.004** |
| XGBoost | Morgan | 0.702 ± 0.003 | 0.318 ± 0.002 |
| XGBoost | Morgan + descriptors | 0.761 ± 0.004 | 0.401 ± 0.005 |

The random-split scores are higher for all three models. The two test sets contain different molecules, however, so this is a comparison between evaluation settings, not a controlled measurement of leakage. The random and scaffold results, endpoint details, and prediction files are in `results/` and `tables/`.

## Dataset and evaluation

The cleaned dataset contains **7,709 unique molecules** from Tox21. Cleaning keeps the largest fragment, canonicalizes SMILES, removes invalid structures, deduplicates molecules, and sets conflicting duplicate labels to missing. The cleaned CSV defines the canonical row order; split files and feature arrays use that same order.

Each molecule has up to 12 binary assay labels. A missing value means the molecule was not tested on that endpoint, not that it was non-toxic. Missing labels remain `NaN`; each endpoint model and metric uses labeled rows only. Positive rates range from 2.91% to 15.94%, so the project reports PR AUC as well as ROC AUC.

Both saved splits use an 80/10/10 allocation: **6,167 train / 771 validation / 771 test**.

- **Scaffold split:** groups molecules by Bemis–Murcko scaffold. All 1,753 acyclic molecules share an empty scaffold and land in training, so scaffold validation and test contain only ring systems.
- **Random split:** deterministic row-level split with seed 0.
- The model seeds (0, 1, 2) change model randomness only. There is one fixed split of each type, so the reported seed variation does not measure split-to-split uncertainty.

## Features and models

- **Morgan:** binary radius-2 circular fingerprints with 2,048 bits.
- **Morgan + descriptors:** the same fingerprints plus 213 retained RDKit 2D descriptors, for 2,261 features total. Of 217 raw descriptors, three are constant zero across the dataset and `fr_azide` is near-constant in scaffold training.
- **Descriptor preprocessing:** non-finite values become missing; all-missing columns are dropped; missing values are median-imputed; extremes are clipped to training-set percentiles; near-constant columns are dropped; remaining values are standardized. Every statistic is fit on the training partition only in `src/features.py`.
- **Models:** one binary classifier per endpoint. Logistic Regression and Random Forest use class weights; XGBoost uses endpoint-specific `scale_pos_weight`.
- **Metrics:** per-endpoint ROC AUC and average precision (PR AUC), evaluated on labeled rows. A metric is `NaN` if an evaluation set contains only one class.

The raw descriptors contain 1,224 NaN cells and 6 infinite cells across 107 molecules. The training-only preprocessing prevents validation or test information from affecting descriptor scaling or imputation.

## Quick start

Run commands from `tox21-phase1/`. Python **3.11 or newer** is required; the project was developed and reproduced with Python 3.14. On macOS, install OpenMP for XGBoost with `brew install libomp`.

```bash
python3 -m venv .venv
source .venv/bin/activate
make setup
make features
make test
```

The cleaned dataset and split files are committed, so the raw download is not needed for the quick start. Feature arrays (`.npy`) are generated locally and ignored by Git. Build them with `make features` before running tests, tuning, or evaluation.

`scikit-learn>=1.8` is required because the Logistic Regression grid uses `l1_ratio` to select L1 and L2 penalties. Without `libomp` on macOS, XGBoost cannot load; XGBoost-specific tests may skip, but tuning and final evaluation require it.

## Reproduce the experiment

| Command | Purpose |
|---|---|
| `make features` | Build Morgan fingerprints, raw descriptors, and the feature manifest. |
| `make tune` | Tune all three models on the scaffold validation set; write `results/tuning_val.csv` and `results/best_configs.json`. |
| `make final-test` | Score the frozen configurations on both test splits and all three model seeds; write the 432-row results file and prediction arrays. |
| `make figures` | Regenerate report figures and summary tables from the committed test results. |
| `make test` | Run the pytest suite; build features first. |
| `make eda` | Regenerate dataset and scaffold EDA figures from the committed cleaned dataset. |
| `make data` | Rebuild the clean dataset and splits from the raw Tox21 CSV. |
| `make all` | Run the complete pipeline from raw data through figures; requires the raw CSV. |

The raw data is not committed. To download it and rebuild the cleaned dataset and splits:

```bash
PYTHONPATH=. python scripts/01_clean_and_split.py --download
```

This saves the source file under `data/raw/`. `make data` and `make all` do not download automatically. If you already have the file, place it at `data/raw/tox21.csv.gz` (or `data/raw/tox21.csv`).

### Protecting the test set

1. Run `make tune`. Tuning is limited to eight configurations per model and feature set, selected by mean validation PR AUC on the scaffold split (seed 0).
2. Commit `results/best_configs.json`.
3. Run `make final-test`. The script requires the configuration file to be complete, valid, tracked, clean, and committed before it reads the test split.

Configurations selected on scaffold validation are reused unchanged for the random split. Do not use test results to select hyperparameters.

For parallel tuning or final evaluation, see the examples in [`scripts/03_tune.py`](scripts/03_tune.py) and [`scripts/04_final_test.py`](scripts/04_final_test.py). The scripts use file locks for shared result files; use separate output files for parallel final-test runs and merge them with `scripts/merge_results.py`.

## Phase 2: deep learning

Phase 2 adds graph construction, MPNN and GCN architectures, fingerprint MLPs, masked multi-task training, and a frozen-config test runner under `src/dl/`. The Phase 2 code uses the same saved scaffold and random test indices and preserves missing labels as `NaN`.

Install the additional dependencies after the Phase 1 setup:

```bash
python -m pip install -r requirements_phase2.txt
```

The frozen graph MPNN results and configs are already committed. For Malhar's Phase 2 analysis and QA, run:

```bash
python scripts/07_tanimoto_analysis.py
python scripts/08_member5_validation.py --splits scaffold,random
```

The Tanimoto analysis compares frozen XGBoost, graph MPNN, and H1 (MPNN + descriptors) predictions across similarity quantiles on the scaffold test set. It writes the figure, CSV tables, and a LaTeX table to `report_phase2/`. The validation script checks prediction dimensions, ranges, endpoint counts, and metric agreement against the saved result CSVs; it also records model-seed variation.

Run the architecture ablations and shuffled-label negative control with:

```bash
python scripts/09_run_malhar_ablations.py --dry-run
python scripts/09_run_malhar_ablations.py
```

These test-set runs must use the frozen config and require PyTorch and PyTorch Geometric. They write `results/dl_ablation_test_malhar.csv`; do not report outcomes until that file has been generated. See [`report_phase2/README.md`](report_phase2/README.md) for the report handoff and reproducibility details.

## Repository map

| Path | Contents |
|---|---|
| `src/data.py`, `src/splits.py` | Cleaning, label handling, and canonical split loading/building. |
| `src/features.py` | Morgan and descriptor generation, train-only descriptor preprocessing, feature loading. |
| `src/models/` | Model factories, hyperparameter grids, and model registry. |
| `src/evaluate.py` | Masked metrics, endpoint training, validation tuning, and final scoring. |
| `src/dl/` | Graph featurization, GCN/MPNN/FP-MLP models, training, ensemble, and Phase 2 analysis. |
| `src/similarity.py` | Maximum Morgan Tanimoto similarity to the training set. |
| `scripts/` | Data, feature, tuning, evaluation, EDA, figure, result-merge, and Phase 2 QA entry points. |
| `data/processed/`, `data/splits/` | Committed cleaned data, deduplication audit, and split indices. |
| `features/` | Generated local feature arrays and committed manifest. |
| `results/`, `tables/`, `figures/` | Tuning and test results, predictions, paper tables, and plots. |
| `report/`, `slides/` | LaTeX report, presentation source, and generated deck. |
| `report_phase2/` | Phase 2 report fragments, Tanimoto analysis tables, and figure. |
| `tests/` | Unit, script, and smoke tests. |

See [`CONTRACT.md`](CONTRACT.md) for binding rules about endpoint order, row alignment, missing labels, feature processing, tuning, and test-set use.

## QA and known limitations

The Phase 1 QA record is in [`docs/qa_report.md`](docs/qa_report.md), and [`docs/viva_drill.md`](docs/viva_drill.md) maps presentation questions to evidence in the repository. Phase 2 checks, Tanimoto results, clean-clone test results, and the dependency blocker for ablation training are recorded in [`docs/phase2_member5_qa.md`](docs/phase2_member5_qa.md).

Key limitations to state alongside the results:

- There is one fixed scaffold split and one fixed random split; seed standard deviations reflect model randomness only.
- Scaffold test results have 20–109 positive examples per endpoint and no acyclic molecules.
- Random and scaffold test sets contain different compounds and positive counts; their score difference is not a controlled causal estimate of scaffold leakage.
- No significance test, bootstrap confidence interval, or error-by-scaffold analysis is reported.
- The experiment uses fingerprints and 2D descriptors; it does not establish why descriptors help the tree models.
- Hyperparameters are tuned on scaffold validation and reused unchanged on the random split.

## Team

| Member | Project responsibility |
|---|---|
| Raghav | Data cleaning and deduplication, EDA, splits, Logistic Regression. |
| Ranajeet | Morgan fingerprints, RDKit descriptors, leak-free preprocessing, Random Forest. |
| Parrv | Repository structure and contract, evaluation harness, XGBoost, tuning and final-test scripts, reproducibility. |
| Abhijeet | Literature review, theory, results analysis, Tanimoto study, report and slides. |
| Malhar (Member 5) | Phase 2 prediction alignment and seed-variance QA, Tanimoto-bin comparison, predeclared MPNN ablations and shuffled-label control, clean-clone verification, and report handoff. |
