"""TEST SET. Final evaluation of the frozen configs on every split and seed.

Refuses to run unless ALL of these hold:
  1. ``--i-know-this-touches-test`` is passed;
  2. ``results/best_configs.json`` exists and has a key for every model x feature set;
  3. every config in it is one of that model's grid entries;
  4. the file is tracked, has no uncommitted changes, and has at least one commit,
     so "configs frozen before the test run" is provable from git history.

Configs were tuned on the scaffold split and are reused unchanged for the random split.
Parallel use: ``--models xgb --out results/parts/xgb.csv`` per terminal, then
``python scripts/merge_results.py``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import evaluate
from src.config import FEATURE_SETS, SEEDS, SPLIT_NAMES
from src.models import REGISTRY


def _git(path: Path, *args: str, must_succeed: bool = True) -> str:
    """Run git in the directory of ``path``; refuse if git or the repository is unavailable.

    With ``must_succeed=False`` a non-zero exit returns an empty string instead of refusing.
    """
    try:
        proc = subprocess.run(["git", *args], cwd=path.parent, capture_output=True, text=True)
    except FileNotFoundError:
        sys.exit("REFUSED: git is not available, cannot prove best_configs.json was frozen.")
    if proc.returncode != 0:
        if must_succeed:
            sys.exit(f"REFUSED: git failed for {path}: {proc.stderr.strip()}")
        return ""
    return proc.stdout.strip()


def load_frozen(path: Path, registry: dict) -> dict[str, dict[str, Any]]:
    """Load best_configs.json and check it covers every model x feature set with valid configs."""
    if not path.exists():
        sys.exit(f"REFUSED: {path} is missing. Run scripts/03_tune.py, then commit it.")
    best = json.loads(path.read_text())
    expected = [f"{m}|{fs}" for m in registry for fs in FEATURE_SETS]
    missing = [k for k in expected if k not in best]
    if missing:
        sys.exit(f"REFUSED: {path.name} lacks {len(missing)} of {len(expected)} keys: {missing}")
    for k in expected:
        grid = registry[k.split("|")[0]][1]
        if best[k] not in grid:
            sys.exit(f"REFUSED: config for {k} is not an entry of that model's grid: {best[k]}")
    return best


def assert_frozen_in_git(path: Path) -> str:
    """Refuse unless ``path`` is tracked, clean and committed. Returns the latest commit hash."""
    status = _git(path, "status", "--porcelain", "--", path.name)
    if status:
        sys.exit(
            f"REFUSED: {path.name} is untracked or has uncommitted changes (git status: {status!r}). "
            "Commit it first so the freeze is provable."
        )
    # status succeeded, so this is a repository; a failing log means it has no commits yet
    commit = _git(path, "log", "-n", "1", "--format=%h", "--", path.name, must_succeed=False)
    if not commit:
        sys.exit(f"REFUSED: {path.name} has no commit in git history (ignored or never added).")
    return commit


def main(argv: list[str] | None = None, registry: dict | None = None) -> pd.DataFrame:
    """Run the final test grid and write the results CSV (overwritten, not appended)."""
    registry = REGISTRY if registry is None else registry
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--i-know-this-touches-test", action="store_true")
    ap.add_argument("--models", help="comma list for parallel runs, e.g. xgb,rf")
    ap.add_argument("--out", type=Path, help="output CSV (default results/results_test.csv)")
    args = ap.parse_args(argv)

    if not args.i_know_this_touches_test:
        sys.exit("REFUSED: this touches the test set. Pass --i-know-this-touches-test after freezing best_configs.json.")
    path = evaluate.RESULTS / "best_configs.json"
    best = load_frozen(path, registry)
    commit = assert_frozen_in_git(path)
    print(f"best_configs.json is clean and frozen at commit {commit}")

    names = list(registry) if args.models is None else [m.strip() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in names if m not in registry]
    if unknown:
        sys.exit(f"Unknown model(s) {unknown}; choose from {list(registry)}")

    frames = []
    for name in names:
        make, _ = registry[name]
        for fs in FEATURE_SETS:
            for split in SPLIT_NAMES:
                for seed in SEEDS:
                    t0 = time.perf_counter()
                    frames.append(evaluate.final_test(name, make, best[f"{name}|{fs}"], fs, split, seed))
                    print(f"{name} | {fs} | {split} | seed {seed}: {time.perf_counter() - t0:.1f}s", flush=True)
    out = pd.concat(frames, ignore_index=True)[evaluate.RESULT_COLUMNS]

    out_path = args.out or evaluate.RESULTS / "results_test.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)  # overwrite, never append
    print(f"wrote {out_path} ({len(out)} rows)")
    summary = out.groupby(["model", "feature_set", "split"])[["roc_auc", "pr_auc"]].mean().round(4)
    print("\nMean over endpoints and seeds (NaN endpoints skipped):")
    print(summary.to_string())
    return out


if __name__ == "__main__":
    main()
