PYTHON ?= python

.PHONY: clean-split features tune test figures test-code

clean-split:
	$(PYTHON) scripts/01_clean_and_split.py

features:
	$(PYTHON) scripts/02_build_features.py

tune:
	$(PYTHON) scripts/03_tune.py

test:
	$(PYTHON) scripts/04_final_test.py

figures:
	$(PYTHON) scripts/05_make_figures.py

test-code:
	$(PYTHON) -m pytest tests

