"""Validation-only tuning: loop REGISTRY x FEATURE_SETS and freeze the best configs.

Writes ``results/tuning_val.csv`` (via ``evaluate.tune``) and ``results/best_configs.json``
with keys ``"<model>|<feature_set>"``. Commit ``best_configs.json`` BEFORE running
``scripts/04_final_test.py``. This script never touches the test split.

Parallel use: ``--models xgb`` in one terminal and ``--models rf`` in another. With
``--models`` the chosen configs are MERGED into the existing ``best_configs.json``;
without it the file is rewritten from scratch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import evaluate
from src.config import FEATURE_SETS
from src.models import REGISTRY

SPLIT = "scaffold"


def mean_val_pr_auc(model: str, feature_set: str, config_id: int) -> float:
    """Mean validation PR AUC (NaN endpoints skipped) of one tuned config, read from tuning_val.csv."""
    df = pd.read_csv(evaluate.RESULTS / "tuning_val.csv")
    rows = df[
        (df["model"] == model) & (df["feature_set"] == feature_set) & (df["split"] == SPLIT)
        & (df["seed"] == evaluate.TUNE_SEED) & (df["config_id"] == config_id)
    ]
    return float(rows["pr_auc"].mean())


def select_models(arg: str | None, registry: dict) -> list[str]:
    """Parse the ``--models`` comma list; default is every registered model."""
    if arg is None:
        return list(registry)
    names = [m.strip() for m in arg.split(",") if m.strip()]
    unknown = [m for m in names if m not in registry]
    if unknown:
        sys.exit(f"Unknown model(s) {unknown}; choose from {list(registry)}")
    return names


def main(argv: list[str] | None = None, registry: dict | None = None) -> dict[str, dict[str, Any]]:
    """Tune the selected models on validation data and write ``best_configs.json``."""
    registry = REGISTRY if registry is None else registry
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", help="comma list, e.g. xgb,rf; merges into the existing best_configs.json")
    args = ap.parse_args(argv)
    names = select_models(args.models, registry)

    path = evaluate.RESULTS / "best_configs.json"
    best: dict[str, dict[str, Any]] = {}
    if args.models and path.exists():
        best = json.loads(path.read_text())

    for name in names:
        make, grid = registry[name]
        for fs in FEATURE_SETS:
            k = evaluate.tune(name, make, grid, fs, split=SPLIT)
            best[f"{name}|{fs}"] = grid[k]
            print(f"{name}|{fs}: best config_id={k} params={json.dumps(grid[k])} "
                  f"mean val PR AUC={mean_val_pr_auc(name, fs, k):.4f}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(sorted(best.items())), indent=2) + "\n")
    print(f"wrote {path} ({len(best)} keys). Commit it before running 04_final_test.py.")
    return best


if __name__ == "__main__":
    main()
