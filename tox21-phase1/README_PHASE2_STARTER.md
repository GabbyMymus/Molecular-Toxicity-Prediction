# Phase 2 starter (deep learning branch)

Drop into the root of the Phase 1 repo so that `src/dl/` sits next to `src/data.py`, `src/evaluate.py`, ...

    unzip phase2_starter.zip -d tox21-phase1/         # adds src/dl/, tests/test_dl_smoke.py, requirements_phase2.txt
    pip install -r requirements_phase2.txt
    pytest tests/test_dl_smoke.py -q                  # 11 tests, ~10 s, no Phase 1 data needed

Needs from Phase 1 (unchanged): src.config {ROOT, RESULTS, ENDPOINTS, SEEDS}, src.data.load_clean,
src.splits.load_split, src.features.load_features, src.evaluate {masked_metrics, score}.

Run (from the repo root):
    python -m src.dl.run --mode tune  --model gcn   [--grid-limit N]      # validation only
    python -m src.dl.run --mode tune  --model mpnn
    python -m src.dl.run --mode tune  --model fpmlp --extra both
    # commit results/best_configs_dl.json, THEN:
    python -m src.dl.run --mode final --model mpnn --i-know-this-touches-test
    python -m src.dl.run --mode final --model mpnn --extra desc --i-know-this-touches-test      # hybrid H1
    python -m src.dl.ensemble --a xgb:morgan+desc --b mpnn+desc:graph+desc --i-know-this-touches-test   # hybrid H2
    python -m src.dl.run --mode ablate --model mpnn --variant steps=1 --tag T1 --i-know-this-touches-test

Add to .gitignore: data/processed/graphs.pt  results/preds/  results/curves/
