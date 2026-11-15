PYTHON ?= python
export PYTHONPATH := $(CURDIR)

.PHONY: setup data eda features tune final-test figures all test \
	dl-guard dl-tune dl-final dl-ablate dl-ensemble phase2-test

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/01_clean_and_split.py

eda:
	$(PYTHON) scripts/06_dataset_eda.py

features:
	$(PYTHON) scripts/02_build_features.py

# validation only
tune:
	$(PYTHON) scripts/03_tune.py

# test set: run only after results/best_configs.json is committed
final-test:
	$(PYTHON) scripts/04_final_test.py --i-know-this-touches-test

figures:
	$(PYTHON) scripts/05_make_figures.py

all: data eda features tune final-test figures

# pytest owns the name "test"
test:
	$(PYTHON) -m pytest tests


# ---------------------------------------------------------------------------
# Phase 2 (deep learning). Order: dl-tune -> freeze (commit results/best_configs_dl.json)
# -> dl-final -> dl-ablate -> dl-ensemble. Run from tox21-phase1/.
# ---------------------------------------------------------------------------

# Refuses to continue unless the frozen config file exists, is tracked and is unmodified.
dl-guard:
	@test -f results/best_configs_dl.json || { echo "ERROR: results/best_configs_dl.json not found. Tune, then commit (freeze) it before any test run."; exit 1; }
	@git ls-files --error-unmatch results/best_configs_dl.json >/dev/null 2>&1 || { echo "ERROR: results/best_configs_dl.json is not committed. Freeze it first."; exit 1; }
	@git diff --quiet HEAD -- results/best_configs_dl.json || { echo "ERROR: results/best_configs_dl.json differs from the committed version."; exit 1; }
	@echo "Configs are frozen: results/best_configs_dl.json is committed and unchanged. The test split will be read."

# validation only (seed 0, scaffold split, at most 8 configs per model). MPNN is slow: use a GPU (Kaggle/Colab).
# run.py merges its result into results/best_configs_dl.json, so this target refuses to run once that file exists.
dl-tune:
	@test ! -e results/best_configs_dl.json || { echo "ERROR: results/best_configs_dl.json already exists (frozen). dl-tune would modify it; not running."; exit 1; }
	$(PYTHON) -m src.dl.run --mode tune --model gcn
	$(PYTHON) -m src.dl.run --mode tune --model mpnn
	$(PYTHON) -m src.dl.run --mode tune --model fpmlp --extra morgan
	$(PYTHON) -m src.dl.run --mode tune --model fpmlp --extra both

# test set: D1 GCN, D2 MPNN, D0 FP-MLP (morgan+desc), H1 MPNN+descriptors. Scaffold and random splits, seeds 0,1,2.
dl-final: dl-guard
	$(PYTHON) -m src.dl.run --mode final --model gcn --i-know-this-touches-test
	$(PYTHON) -m src.dl.run --mode final --model mpnn --i-know-this-touches-test
	$(PYTHON) -m src.dl.run --mode final --model fpmlp --extra both --i-know-this-touches-test
	$(PYTHON) -m src.dl.run --mode final --model mpnn --extra desc --i-know-this-touches-test

# test set: ablations reconstructed from the tag/variant columns of results/dl_ablation_test_*.csv
# (scaffold split, seeds 0,1,2; the command lines themselves are not recorded in the repo).
dl-ablate: dl-guard
	$(PYTHON) -m src.dl.run --mode ablate --model mpnn --extra none --variant edge_drop=0.2 --tag dropedge --i-know-this-touches-test
	$(PYTHON) -m src.dl.run --mode ablate --model mpnn --extra none --variant use_pos_weight=false --tag no_class_weight --i-know-this-touches-test
	$(PYTHON) -m src.dl.run --mode ablate --model mpnn --extra none --variant dropout=0 --tag no_dropout --i-know-this-touches-test
	$(PYTHON) -m src.dl.run --mode ablate --model mpnn --extra none --variant mask_missing=false --tag no_mask --i-know-this-touches-test
	$(PYTHON) -m src.dl.run --mode ablate --model mpnn --extra none --variant weight_decay=0 --tag no_wd --i-know-this-touches-test
	$(PYTHON) -m src.dl.run --mode ablate --model mpnn --extra morgan --tag morgan --i-know-this-touches-test
	$(PYTHON) -m src.dl.run --mode ablate --model mpnn --extra both --tag both --i-know-this-touches-test
	$(PYTHON) -m src.dl.run --mode ablate --model mpnn --extra desc --shuffle-extra --tag shuffled_desc --i-know-this-touches-test

# test set: H2 rank-average (0.5/0.5) of the Phase 1 XGBoost (morgan+desc) and one deep partner.
# The partner is chosen by validation PR AUC (see report_phase2/sections/hybrid.tex), not on test data:
#   primary H2   = xgb:morgan+desc with fpmlp+both:both  -> ens_xgb_fpmlp+both
#   secondary    = xgb:morgan+desc with mpnn+desc:graph+desc -> ens_xgb_mpnn+desc (documented alternative)
# Needs the Phase 1 XGBoost preds and the FP-MLP / H1 finals (dl-final). Both runs append to results/dl_results_test.csv;
# the committed files results/dl_results_test_ranajeet_h2.csv and ..._h2_alt_mpnndesc.csv hold one run each.
dl-ensemble: dl-guard
	$(PYTHON) -m src.dl.ensemble --a xgb:morgan+desc --b fpmlp+both:both --i-know-this-touches-test
	$(PYTHON) -m src.dl.ensemble --a xgb:morgan+desc --b mpnn+desc:graph+desc --i-know-this-touches-test

phase2-test:
	$(PYTHON) -m pytest tests/test_dl_smoke.py tests/test_gate0_ranajeet.py tests/test_graph_alignment.py tests/test_extra_alignment.py -q
