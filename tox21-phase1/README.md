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

Run everything from this directory (`tox21-phase1/`). Developed and tested on Python 3.14.

```bash
python -m venv .venv && source .venv/bin/activate
make setup            # pip install -r requirements.txt
```

- **scikit-learn >= 1.8 is required.** Logistic Regression selects the L1 or L2 penalty from `l1_ratio`
  only from 1.8; on older versions the two L1 configurations in the grid silently fit as L2. This was
  checked on 1.3.2 to 1.9.1.
- On macOS, XGBoost needs the OpenMP runtime: `brew install libomp`.
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
| `make data` | Re-creates the cleaned CSV and splits from the raw file (see Known issues before using). |
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

The shared rules every script follows are in [`CONTRACT.md`](CONTRACT.md).

## Results

<!-- RESULTS-TABLE-PLACEHOLDER: to be filled from results/results_test.csv after the final test run.
     Do not add numbers by hand; generate this table from the CSV. -->

_Results pending: no final test run has been made yet._

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
make features
make test
```

The harness tests use synthetic data, including throwaway git repositories that exercise every refusal
condition of the final-test script.

## Known issues

- `make data` regenerates the cleaned CSV and splits from the raw file, but the committed
  `data/splits/scaffold_seed0.json` is not what the current `src/splits.py::scaffold_split` produces
  (it yields a different, unbalanced split). Treat the committed files in `data/processed/` and
  `data/splits/` as canonical and do not overwrite them until the data code is aligned.
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
