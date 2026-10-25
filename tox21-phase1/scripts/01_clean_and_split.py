"""Create the canonical clean dataset, audit log, and both split files."""

import argparse
import json
import urllib.request

import pandas as pd

from src.config import CLEAN_PATH, RAW_PATH, RAW_URL, SPLITS_PATH
from src.data import clean_dataset
from src.splits import random_split, scaffold_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    if args.download:
        RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(RAW_URL, RAW_PATH)
    input_path = RAW_PATH if RAW_PATH.exists() else RAW_PATH.with_suffix("").with_suffix(".csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Place raw data at {RAW_PATH} (or {input_path}) or pass --download")

    clean, _, log = clean_dataset(pd.read_csv(input_path), return_log=True)
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH, index=False)
    (CLEAN_PATH.parent / "dedup_log.txt").write_text(
        "\n".join(f"{key}: {value}" for key, value in log.items())
    )

    SPLITS_PATH.mkdir(parents=True, exist_ok=True)
    splits = {
        "scaffold_seed0.json": scaffold_split(clean["scaffold"].tolist()),
        "random_seed0.json": random_split(len(clean), seed=0),
    }
    for filename, split in splits.items():
        (SPLITS_PATH / filename).write_text(json.dumps(split))
        print(filename, {key: len(value) for key, value in split.items()})
    print("clean rows:", len(clean), "; audit:", log)


if __name__ == "__main__":
    main()
