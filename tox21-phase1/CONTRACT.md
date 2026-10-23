# Phase 1 contract

- Cleaned row order is the canonical order for every feature matrix and split.
- `mol_id` is a stable zero-based identifier after cleaning.
- Columns are `mol_id`, `smiles`, `scaffold`, followed by the 12 endpoint names
  in `src/config.py::ENDPOINTS`.
- Missing labels are represented by `NaN`; they are excluded from metrics.
- Split JSON files contain integer `train`, `val`, and `test` index arrays.
- Split sizes are approximately 80/10/10, with scaffold split enforcing no
  scaffold overlap between partitions.
- Randomness is controlled by `SEEDS = (0, 1, 2)`.
- `results/results_test.csv` has columns:
  `model,feature_set,split,seed,endpoint,roc_auc,pr_auc,n_test,n_pos_test`.
- Tuning is validation-only. `best_configs.json` is frozen before test runs.

