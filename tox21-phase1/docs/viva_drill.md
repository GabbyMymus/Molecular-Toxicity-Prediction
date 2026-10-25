# Viva drill: where each handbook section 16 answer lives in the repo

Questions are from the Phase 1 handbook, section 16. The model answers are in the handbook; this page points to the
evidence in this repository so each person can open it and quote a measured number. Numbers below were read from
the committed files (`results/results_test.csv`, `results/best_configs.json`, `tables/`).
The quiz itself has not been run with the team yet.

| Question | Evidence in the repo |
|---|---|
| Why mask missing labels instead of filling with 0? | `src/evaluate.py` (`fit_predict`, `masked_metrics`); `CONTRACT.md`; `tests/test_evaluate.py::test_nan_labels_are_excluded`. 17.05% of label cells are missing. |
| Why a scaffold split? | `src/splits.py::scaffold_split`; all 1,753 acyclic molecules are in train. |
| Why canonicalize and dedupe before splitting? | `src/data.py::clean_dataset`; `data/processed/dedup_log.txt` (7,831 raw rows, 8 invalid, 44 label conflicts set to missing, 114 duplicates removed, 7,709 kept). |
| What is a Morgan bit? What does radius 2 mean? | `src/features.py::build_morgan` (2,048 bits, radius 2); `notebooks/02_features.ipynb`. |
| Why PR AUC next to ROC AUC? | `src/evaluate.py::masked_metrics`; positive rates run from 2.91% (NR-PPAR-gamma) to 15.94% (SR-ARE). |
| Why fit the descriptor scaler on train only? | `src/features.py::DescriptorPreprocessor`, `load_features()`; `tests/test_features.py`. |
| How do the three models differ? | `src/models/logreg.py`, `rf.py`, `xgb.py`; chosen configs in `results/best_configs.json`. |
| How did you handle class imbalance? | LR `class_weight="balanced"`, RF `"balanced_subsample"`, XGBoost `scale_pos_weight`; see `src/models/`. |
| How did you tune, and how do you know the test set was not used? | `scripts/03_tune.py`, `scripts/04_final_test.py`; `git log`: `041892c` (freeze configs, 2026-10-25 19:41:39 +05:30) before `043c2ed` (final results, 19:52:52). |
| Why is the scaffold split harder, with evidence? | `tables/scaffold_penalty.csv`; mean max Tanimoto test-to-train 0.428 (scaffold) vs 0.584 (random), `src/similarity.py`. |
| What are the limitations? | Scaffold test has 20 to 109 positives per endpoint; one fixed split per type; no significance tests. |
| What does the MPNN paper contribute? | `report/sections/related_work.tex`; `report/references.bib` (`gilmer2017neural`). |
| Why is accuracy a bad metric here? | Positive rates above; with 2.91% to 15.94% positives an all-negative predictor already scores high accuracy. |
| Why one model per endpoint? | `src/evaluate.py::fit_predict` fits one classifier per endpoint on that endpoint's labeled rows. |
| What did descriptors add on top of Morgan bits? | `tables/descriptor_benefit.csv`: LR +0.0162 ROC / +0.0006 PR; RF +0.0474 / +0.0333; XGB +0.0586 / +0.0829 (scaffold test, paired by seed). |
