# Tox21 Phase 1

Reproducible multi-task toxicity prediction pipeline.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

Each stage has one Make target:

```bash
make clean-split
make features
make tune
make test
make figures
make test-code
```

The raw download is intentionally not committed. Place `tox21.csv.gz` in
`data/raw/` or run `python scripts/01_clean_and_split.py --download`.

## Results

`results/tuning_val.csv` stores validation scores for every configuration,
`results/best_configs.json` freezes the selected configurations, and
`results/results_test.csv` stores final endpoint-level metrics. Test metrics
must only be generated after tuning has frozen the configurations.

