# Tox21 Molecular Toxicity Prediction

This project builds reproducible classical machine learning baselines for predicting activity across the 12 Tox21 toxicity assays. It compares Logistic Regression, Random Forest, and XGBoost using Morgan fingerprints alone and fingerprints combined with RDKit molecular descriptors. It evaluates both random and Bemis–Murcko scaffold splits.

> **Scope:** This repository contains Phase 1 (classical baselines) and Phase 2 (deep learning in `src/dl/`; see [Phase 2: deep learning](#phase-2-deep-learning)). The results and tables in the Phase 1 sections below are Phase 1 only.

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

Phase 2 adds deep models on the same Phase 1 data, splits, seeds (0, 1, 2) and masked ROC AUC / PR AUC metrics. The code is in `src/dl/`.

| ID | Model |
|---|---|
| D0 | FP-MLP: a multilayer perceptron on Morgan bits and/or RDKit descriptors. |
| D1 | GCN on molecular graphs (atom features only). |
| D2 | MPNN: edge-network messages, GRU update, Set2Set readout. |
| H1 | MPNN with the RDKit descriptor vector added before the classification head. |
| H2 | Rank-average ensemble (weights 0.5 / 0.5, no training) of the Phase 1 XGBoost predictions (morgan+desc) and the deep model with the best validation PR AUC (`fpmlp+both`); a secondary variant pairs XGBoost with H1. |

### Phase 2 setup

Run commands from `tox21-phase1/` with the virtual environment active.

```bash
pip install -r requirements.txt -r requirements_phase2.txt
```

`make setup` installs `requirements.txt` only. `requirements_phase2.txt` adds `torch`, `torch_geometric` and `scipy`.

The feature arrays `features/morgan_2048.npy` and `features/descriptors_raw.npy` are not tracked by Git (`features/*.npy` is ignored). Regenerate them with the Phase 1 command:

```bash
make features
```

The graph cache `data/processed/graphs.pt` is also ignored by Git. `src.dl.run` rebuilds it from the cleaned CSV when it is missing or no longer matches.

### Phase 2 run order

| Step | Command | Reads test split? |
|---|---|---|
| 1. Tune | `make dl-tune` | No (validation only, seed 0, scaffold split, at most 8 configs per model) |
| 2. Freeze | Commit `results/best_configs_dl.json` with a commit message that starts with `FREEZE` | No |
| 3. Final | `make dl-final` | Yes |
| 4. Ablate | `make dl-ablate` | Yes |
| 5. Ensemble | `make dl-ensemble` | Yes |

`dl-final`, `dl-ablate` and `dl-ensemble` first run `dl-guard`, which stops unless `results/best_configs_dl.json` exists, is committed, and is unchanged, and then prints that the configs are frozen. `dl-tune` refuses to run once `results/best_configs_dl.json` exists, because `src.dl.run --mode tune` merges its result into that file. MPNN tuning and finals are slow on a CPU; use a GPU (Kaggle or Colab).

### Freeze rule

The deep-learning configurations were frozen in commit `0aebd87` ("FREEZE: DL configs frozen before any test evaluation"), which adds `results/best_configs_dl.json`. Do not edit that file afterwards and do not use test results to choose hyperparameters. The test split is read only by `--mode final`, `--mode ablate` and `src/dl/ensemble.py`; each needs `--i-know-this-touches-test`, and `--mode final` and `--mode ablate` also exit unless `results/best_configs_dl.json` exists.

### Phase 2 result files

Files are written by `src.dl.run` / `src.dl.ensemble` under default names (`dl_tuning_val.csv`, `best_configs_dl.json`, `dl_results_test.csv`, `dl_ablation_test.csv`) and were renamed with an owner suffix when committed. Ensemble runs (`make dl-ensemble`) append to `results/dl_results_test.csv`, so after each run rename or split that file into the per-owner results file, as was done for `dl_results_test_ranajeet_h2.csv` and `dl_results_test_ranajeet_h2_alt_mpnndesc.csv`.

| File in `results/` | Owner | Contents |
|---|---|---|
| `dl_tuning_val_parrv.csv` | Parrv | MPNN tuning, validation metrics per config and endpoint (scaffold split, seed 0). |
| `dl_tuning_val_parrv_gcn.csv` | Parrv | GCN tuning, same layout. |
| `best_configs_dl_parrv.json`, `best_configs_dl_parrv_gcn.json` | Parrv | Selected configs for `mpnn\|none` and `gcn\|none`. |
| `best_configs_dl_ranajeet.json` | Ranajeet | Selected configs for `fpmlp\|morgan` and `fpmlp\|both`. |
| `best_configs_dl.json` | Parrv | The merged, frozen configs (commit `0aebd87`). |
| `dl_results_test_parrv_mpnn.csv` | Parrv | D2 MPNN test metrics per endpoint, split and seed. |
| `dl_results_test_raghav.csv` | Raghav | D1 GCN test metrics. |
| `dl_results_test_ranajeet_fpmlp.csv` | Ranajeet | D0 FP-MLP (`fpmlp+both`) test metrics. |
| `dl_results_test_ranajeet_h1.csv` | Ranajeet | H1 (`mpnn+desc`) test metrics. |
| `dl_results_test_ranajeet_h2.csv` | Ranajeet | H2 ensemble (`ens_xgb_fpmlp+both`) test metrics. |
| `dl_results_test_ranajeet_h2_alt_mpnndesc.csv` | Ranajeet | Secondary H2 variant (`ens_xgb_mpnn+desc`) test metrics. |
| `dl_ablation_test_raghav.csv` | Raghav | MPNN regularization ablations (`dropedge`, `no_class_weight`, `no_dropout`, `no_mask`, `no_wd`), scaffold split. |
| `dl_ablation_test_ranajeet.csv` | Ranajeet | MPNN feature ablations (`morgan`, `both`, `shuffled_desc`), scaffold split. |
| `dl_ablation_test_malhar.csv` | Malhar | MPNN architecture ablations and shuffled-label control (scaffold split, seeds 0–2). |
| `preds/` | Written by each run | Test-set probability arrays (tracked in git). Phase 2 files are named `{model}_{feature_set}_{split}_seed{k}_{tag}.npy` (`tag` is `final` for finals). |
| `curves/` | Written by each run | Per-epoch training loss and validation PR AUC / ROC AUC as JSON, same naming (tracked in git). |

### Malhar's analysis and validation

Malhar's Phase 2 work covers similarity-stratified evaluation, prediction alignment and seed variability, and the predeclared MPNN architecture ablations. From `tox21-phase1/`, run:

```bash
python scripts/07_tanimoto_analysis.py
python scripts/08_member5_validation.py --splits scaffold,random
python scripts/09_run_malhar_ablations.py --dry-run
python scripts/09_run_malhar_ablations.py
```

Run the ablations on a GPU after installing `requirements.txt` and `requirements_phase2.txt`. They write `results/dl_ablation_test_malhar.csv`; use that CSV as the source for reported ablation metrics. The prediction alignment and provenance audit is in `results/member5_prediction_audit.csv`, and the Phase 2 QA handoff is in `docs/phase2_member5_qa.md`.

### Phase 2 tests

```bash
make phase2-test
```

This runs `tests/test_dl_smoke.py`, `tests/test_gate0_ranajeet.py`, `tests/test_graph_alignment.py` and `tests/test_extra_alignment.py`. `make test` runs the whole suite.

## Repository map

| Path | Contents |
|---|---|
| `src/data.py`, `src/splits.py` | Cleaning, label handling, and canonical split loading/building. |
| `src/features.py` | Morgan and descriptor generation, train-only descriptor preprocessing, feature loading. |
| `src/models/` | Model factories, hyperparameter grids, and model registry. |
| `src/evaluate.py` | Masked metrics, endpoint training, validation tuning, and final scoring. |
| `src/similarity.py` | Maximum Morgan Tanimoto similarity to the training set. |
| `scripts/` | Data, feature, tuning, evaluation, EDA, figure, and result-merge entry points. |
| `data/processed/`, `data/splits/` | Committed cleaned data, deduplication audit, and split indices. |
| `features/` | Generated local feature arrays and committed manifest. |
| `results/`, `tables/`, `figures/` | Tuning and test results, predictions, paper tables, and plots. |
| `report/`, `slides/` | LaTeX report, presentation source, and generated deck. |
| `tests/` | Unit, script, and smoke tests. |

See [`CONTRACT.md`](CONTRACT.md) for binding rules about endpoint order, row alignment, missing labels, feature processing, tuning, and test-set use.

## QA and known limitations

The Phase 1 QA record is in [`docs/qa_report.md`](docs/qa_report.md), and [`docs/viva_drill.md`](docs/viva_drill.md) maps presentation questions to evidence in the repository. Phase 2's Member 5 checks and report handoff are in [`docs/phase2_member5_qa.md`](docs/phase2_member5_qa.md). The Phase 1 QA report documents a passing 200-molecule smoke test and reproduction of the committed 432-row final results. It also reports that notebooks `02_features.ipynb` and `03_analysis.ipynb` ran cleanly, while `01_eda.ipynb` fails as committed because it uses a Google Colab path. A portable-path fix is proposed in the QA report but has not been applied there.

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
| Malhar (Member 5) | Prediction alignment and seed-variance QA, Tanimoto-bin analysis, predeclared MPNN ablations and shuffled-label control, clean-clone verification, README and slide polish, and report handoff. |
