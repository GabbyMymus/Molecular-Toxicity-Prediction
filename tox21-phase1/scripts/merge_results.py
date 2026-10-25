"""Concatenate per-model parts from parallel 04_final_test.py runs into results/results_test.csv."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import evaluate
from src.config import ENDPOINTS

KEY = ["model", "feature_set", "split", "seed", "endpoint"]


def main(argv: list[str] | None = None) -> pd.DataFrame:
    """Merge part files (default ``results/parts/*.csv``) and write the combined CSV."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parts", nargs="*", type=Path, help="part CSVs (default: results/parts/*.csv)")
    ap.add_argument("--out", type=Path, help="default results/results_test.csv")
    args = ap.parse_args(argv)

    parts = args.parts or sorted((evaluate.RESULTS / "parts").glob("*.csv"))
    if not parts:
        sys.exit("No part files found.")
    frames = []
    for p in parts:
        df = pd.read_csv(p)
        if list(df.columns) != evaluate.RESULT_COLUMNS:
            sys.exit(f"{p}: columns {list(df.columns)} != {evaluate.RESULT_COLUMNS}")
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    dup = merged.duplicated(KEY, keep=False)
    if dup.any():
        sys.exit(f"Duplicate (model, feature_set, split, seed, endpoint) rows across parts:\n{merged[dup][KEY].head()}")
    merged["endpoint"] = pd.Categorical(merged["endpoint"], categories=ENDPOINTS, ordered=True)
    merged = merged.sort_values(KEY).reset_index(drop=True)
    merged["endpoint"] = merged["endpoint"].astype(str)

    out = args.out or evaluate.RESULTS / "results_test.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out, index=False)
    print(f"merged {len(parts)} part(s) into {out}: {len(merged)} rows; rows per model:")
    print(merged["model"].value_counts().sort_index().to_string())
    return merged


if __name__ == "__main__":
    main()
