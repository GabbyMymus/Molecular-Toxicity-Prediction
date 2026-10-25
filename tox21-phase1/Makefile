PYTHON ?= python
export PYTHONPATH := $(CURDIR)

.PHONY: setup data features tune final-test figures all test

setup:
	$(PYTHON) -m pip install -r requirements.txt

data:
	$(PYTHON) scripts/01_clean_and_split.py

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

all: data features tune final-test figures

# pytest owns the name "test"
test:
	$(PYTHON) -m pytest tests
