# Phase 1 shared contract (binding for everyone)

These rules keep the team's results comparable. If you need to change one, tell the whole team first.

1. **Row order is sacred.** `data/processed/tox21_clean.csv` defines row order. Split files store integer row indices
   into it. Feature arrays have exactly the same row order.
2. **Labels.** `Y` is an `(N, 12)` float array with `NaN` for missing. The mask is `~np.isnan(Y)`. Never fill NaN with 0.
3. **Endpoint order** is fixed in `src/config.py`: NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD,
   NR-PPAR-gamma, SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53.
4. **Splits.** Scaffold split (80/10/10, deterministic) and one random split (seed 0, 80/10/10), saved as JSON index
   lists. Nobody makes their own split.
5. **Leakage-prone preprocessing** (imputing, clipping, scaling of descriptors) is fit on the train set only and lives in
   one place: `load_features()` in `src/features.py`.
6. **Dedupe and canonicalize before splitting.**
7. **Training uses only labeled rows** of the endpoint being trained, from the train split.
8. **Class imbalance handling:** LR `class_weight="balanced"`, RF `"balanced_subsample"`, XGBoost
   `scale_pos_weight = n_neg / n_pos` from that endpoint's training labels.
9. **Tuning budget:** at most 8 configurations per model per feature set, chosen by mean validation PR AUC across
   endpoints (seed 0). Same budget for every model.
10. **Seeds:** final runs use seeds 0, 1, 2. (Logistic Regression is nearly deterministic, so its std over seeds will be
    about 0; that is expected, state it in the report.)
11. **Test-set discipline:** the test set is scored only in the final run, after `results/best_configs.json` is committed.
    Never tune on test.
12. **Metrics:** per endpoint, masked ROC AUC and PR AUC. If an eval set has no positives or no negatives for an
    endpoint, the metric is NaN and is skipped in averages.
13. **Feature sets:** `morgan` (2048-bit, radius 2) and `morgan+desc` (Morgan + scaled descriptors).

## Result file formats

`results/tuning_val.csv` - one row per (model, feature_set, config_id, endpoint), seed 0, scaffold split.

```
columns: model, feature_set, split, seed, config_id, params, endpoint, roc_auc, pr_auc, n_test, n_pos_test
(in this file n_test / n_pos_test count the VALIDATION rows)
```

`results/best_configs.json` - frozen before any test run.

```
{"logreg|morgan": {...params...}, "logreg|morgan+desc": {...}, "rf|morgan": ..., ...}
```

`results/results_test.csv` - one row per (model, feature_set, split, seed, endpoint).

```
columns: model, feature_set, split, seed, endpoint, roc_auc, pr_auc, n_test, n_pos_test
```

`results/preds/<model>_<feature_set>_<split>_seed<k>.npy` - test-set probabilities, shape `(n_test_rows, 12)`, rows in the
order of `splits[split]["test"]` (used by the similarity analysis). These files are committed.

## Repo notes (where the repo differs from the handbook)

- The project root is the `tox21-phase1/` subfolder of the git repository; all paths above are relative to it.
- `mol_id` in the cleaned CSV is the original Tox21 string id (for example `TOX3021`), not a zero-based integer.
  Row position, not `mol_id`, is the key.
- `src/config.py` keeps legacy aliases (`CLEAN_PATH`, `SPLITS_PATH`, `FEATURES_PATH`, `RESULTS_PATH`, `RAW_PATH`) for
  already-merged code. New code should use the handbook names.
