# Tox21 Phase 1: molecular toxicity prediction with fingerprints and classical ML

Predict the outcome of 12 toxicity assays (Tox21) from a molecule's structure. Phase 1 builds the
classical baseline and the evaluation machinery: SMILES -> Morgan fingerprints (+ RDKit descriptors) ->
Logistic Regression, Random Forest and XGBoost, one classifier per assay, scored with masked ROC AUC and
PR AUC on a scaffold split (unseen chemical families) and on a random split. The graph neural network is
Phase 2 and is not part of this repository yet (`phase2/` is a placeholder).

## Data in one paragraph

Tox21 (MoleculeNet / DeepChem CSV) holds SMILES strings and 12 binary labels per molecule; a blank label
means the compound was not tested on that assay, not that it is non-toxic. After cleaning (largest
fragment, canonical SMILES, deduplication, conflicting labels set to missing) there are **7,709 unique
molecules**. Missing labels stay `NaN` and every metric is computed on labeled rows only. Both splits are
80/10/10 and saved as index lists: **6,167 train / 771 validation / 771 test**. In the scaffold split all
1,753 acyclic (empty-scaffold) molecules sit in the training set, so validation and test contain only ring
systems; keep this in mind when reading scaffold results.

## Setup

The project lives in the `tox21-phase1/` subfolder of the repository, and every command below is run from
there. **Python 3.11 or newer** is needed: `scikit-learn>=1.8` does not install on Python 3.9 (the macOS system
`python3`), and 3.10 was not tested. Developed and tested on Python 3.14.

```bash
git clone https://github.com/GabbyMymus/Molecular-Toxicity-Prediction.git
cd Molecular-Toxicity-Prediction/tox21-phase1
python3 -m venv .venv && source .venv/bin/activate   # inside the venv, `python` works
make setup                                           # pip install -r requirements.txt
```

- **scikit-learn >= 1.8 is required.** Logistic Regression selects the L1 or L2 penalty from `l1_ratio`
  only from 1.8; on older versions the two L1 configurations in the grid silently fit as L2. This was
  checked on 1.3.2 to 1.9.1.
- **macOS needs `brew install libomp` for XGBoost.** Without it XGBoost cannot load: `make test` still runs
  (the 4 XGBoost-specific tests are skipped), but `make tune` and `make final-test` fail on the XGBoost model.
- The raw download is not committed (`data/raw/` is ignored). The cleaned data and splits are committed,
  so you do not need the raw file to run the experiments.

## Run

Every stage has one Make target:

| Command | What it does |
|---|---|
| `make features` | Morgan (2048 bits, radius 2) and raw RDKit descriptors into `features/` (about 30 s). The `.npy` files are not committed; build them locally. |
| `make tune` | Validation-only tuning of all models on both feature sets. Writes `results/tuning_val.csv` and `results/best_configs.json`. |
| `make final-test` | **Touches the test set.** Runs `scripts/04_final_test.py --i-know-this-touches-test`. Writes `results/results_test.csv` and `results/preds/`. |
| `make figures` | Report figures from `results/results_test.csv`. |
| `make test` | The pytest suite. Run `make features` first; two tests read the built feature arrays. |
| `make data` | Re-creates the cleaned CSV, deduplication audit, and both canonical splits from `data/raw/tox21.csv` (or `.csv.gz`). |
| `make eda` | Generates dataset and scaffold EDA figures in `figures/`. |
| `make all` | `data features tune final-test figures`. |

### The test-set protocol

1. `make tune` chooses at most 8 configurations per model and feature set by mean validation PR AUC
   across endpoints (seed 0, scaffold split).
2. Commit `results/best_configs.json`.
3. Only then run `make final-test`. The script refuses to run unless the file contains all 6
   `model|feature_set` keys, every config is one of that model's grid entries, and the file is tracked,
   unmodified and present in git history. That commit is the proof that the test set was never used for
   tuning. Configurations tuned on the scaffold split are reused unchanged for the random split.

### Running models in parallel

```bash
python scripts/03_tune.py --models logreg
python scripts/03_tune.py --models rf          # second terminal
python scripts/03_tune.py --models xgb         # third terminal; results merge into best_configs.json

python scripts/04_final_test.py --i-know-this-touches-test --models rf  --out results/parts/rf.csv
python scripts/04_final_test.py --i-know-this-touches-test --models xgb --out results/parts/xgb.csv
python scripts/merge_results.py                # results/parts/*.csv -> results/results_test.csv
```

Tuning progress is logged one line per configuration. Result files are written under a file lock, so
parallel runs do not overwrite each other.

## Experimental design

| Item | Setting |
|---|---|
| Task | 12 binary endpoints, one classifier per endpoint, trained on that endpoint's labeled training rows |
| Feature sets | `morgan` (2048-bit, radius 2) and `morgan+desc` (Morgan + descriptors) |
| Descriptors | 217 raw RDKit descriptors; infinities and NaNs are handled and scaled inside `load_features()`, fit on the training rows only |
| Models | Logistic Regression, Random Forest, XGBoost |
| Class imbalance | LR `class_weight="balanced"`, RF `"balanced_subsample"`, XGBoost `scale_pos_weight = n_neg / n_pos` per endpoint |
| Tuning | 8 configurations per model per feature set, mean validation PR AUC, seed 0, scaffold split |
| Final runs | seeds 0, 1, 2 on both the scaffold and the random split |
| Metrics | per endpoint ROC AUC and PR AUC on labeled rows; NaN when an evaluation set has a single class |
| Grid size | 3 models x 2 feature sets x 2 splits x 3 seeds x 12 endpoints = 432 fitted classifiers (plus tuning) |

## Feature engineering summary

- **Circular Morgan Fingerprints (ECFP4 Equivalent, 2048-bit, Radius 2):** Tox21 molecules exhibit an average active bit density of $29.75 \pm 16.41$ bits per molecule (median 26.0, range $[1, 160]$). All 2,048 bit positions are activated across the dataset (0 unused bits), while the top motif appears in $71.73\%$ of compounds.
- **RDKit 2D Molecular Descriptors (217 raw features):** Empirical analysis revealed 1,224 NaN cells and 6 infinite cells across 107 compounds (caused by undefined partial charges or zero-division topological indices). 3 descriptors are constant zero across all 7,709 molecules (`SMR_VSA8`, `SlogP_VSA9`, `fr_diazo`), and `fr_azide` is near-constant in scaffold training, leaving 213 retained descriptors.
- **Hybrid Modality (`morgan+desc`, 2,261 features):** Combining localized topological fragments (Morgan) with bulk physicochemical parameters (lipophilicity $\log P$, polar surface area, molecular weight) provides complementary inductive biases without expanding dimensionality beyond tabular model feasibility.

The shared rules every script follows are in [`CONTRACT.md`](CONTRACT.md).

## Results

Test-set results on the **scaffold split** (771 test molecules), from `results/results_test.csv` (432 rows:
3 models x 2 feature sets x 2 splits x 3 seeds x 12 endpoints). Each cell is the mean over the 12 endpoints,
then mean ± sample standard deviation over seeds 0, 1, 2; the best value in each column is in bold. Logistic
Regression is deterministic, so its standard deviation over seeds is 0.000. Hyperparameters were chosen on the
validation split and frozen in git (commit `041892c`) before the test split was scored. The random-split
numbers are in the same CSV. Scaffold validation and test contain only ring-containing molecules, because all
acyclic molecules are in the training set.

| Model | Features | ROC AUC | PR AUC |
|---|---|---|---|
| Logistic Regression | morgan | 0.716 ± 0.000 | 0.312 ± 0.000 |
| Logistic Regression | morgan+desc | 0.732 ± 0.000 | 0.312 ± 0.000 |
| Random Forest | morgan | 0.729 ± 0.006 | 0.370 ± 0.002 |
| Random Forest | morgan+desc | **0.776 ± 0.002** | **0.404 ± 0.004** |
| XGBoost | morgan | 0.702 ± 0.003 | 0.318 ± 0.002 |
| XGBoost | morgan+desc | 0.761 ± 0.004 | 0.401 ± 0.005 |

## Repository layout

```
tox21-phase1/
|-- README.md  CONTRACT.md  Makefile  requirements.txt
|-- data/
|   |-- raw/                 tox21.csv.gz (not committed)
|   |-- processed/           tox21_clean.csv (defines row order), dedup_log.txt
|   `-- splits/              scaffold_seed0.json, random_seed0.json (row indices)
|-- features/                morgan_2048.npy, descriptors_raw.npy (built locally), feature_manifest.json
|-- src/
|   |-- config.py            paths, endpoint order, seeds, feature sets
|   |-- data.py  splits.py   cleaning, split functions, load_clean(), load_split()
|   |-- features.py          Morgan, descriptors, train-only DescriptorPreprocessor, load_features()
|   |-- evaluate.py          masked metrics, per-endpoint fit/predict, tune(), final_test()
|   |-- similarity.py        Tanimoto similarity to the training set
|   `-- models/              logreg.py, rf.py, xgb.py (factories + grids), __init__.py (REGISTRY)
|-- scripts/                 01_clean_and_split, 02_build_features, 03_tune, 04_final_test,
|                            05_make_figures, merge_results
|-- tests/                   pytest suite (data, splits, features, evaluation harness, models, scripts)
|-- notebooks/               01_eda, 02_features, 03_analysis
|-- results/                 tuning_val.csv, best_configs.json, results_test.csv, preds/
|-- figures/  tables/        report figures and tables
|-- report/                  main.tex and sections/ (IEEE format)
|-- slides/
`-- phase2/                  placeholder for the graph neural network
```

## Tests

```bash
make eda
make features
make test
```

The harness tests use synthetic data, including throwaway git repositories that exercise every refusal
condition of the final-test script.

## Known issues

- `make features` rewrites the tracked `features/feature_manifest.json` with local absolute paths; run
  `git checkout features/feature_manifest.json` afterwards.

## Team

Roles follow the Phase 1 team handbook. Per-person commit history is in `git log`.

| Member | Responsibility |
|---|---|
| Raghav | Data cleaning and deduplication, EDA notebook, scaffold and random splits, Logistic Regression |
| Ranajeet | Morgan fingerprints, RDKit descriptors, leak-free descriptor preprocessing and `load_features()`, Random Forest, feature notebook |
| Parrv | Repository structure and shared contract, evaluation harness, XGBoost, tuning and final-test scripts, reproducibility |
| Abhijeet | Literature review, theory, results analysis, Tanimoto similarity study, report and slides |
| Fifth member | Slides, README polish, tests, viva preparation (name to be added) |
