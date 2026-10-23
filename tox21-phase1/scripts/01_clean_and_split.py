"""Create clean data and both split definitions."""

import argparse
import urllib.request

import pandas as pd

from src.config import CLEAN_PATH, RAW_PATH, RAW_URL, SPLITS_PATH
from src.data import clean_dataset
from src.splits import random_split, scaffold_split

parser = argparse.ArgumentParser()
parser.add_argument("--download", action="store_true")
args = parser.parse_args()
if args.download:
    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(RAW_URL, RAW_PATH)
if not RAW_PATH.exists():
    raise FileNotFoundError(f"Place raw data at {RAW_PATH} or pass --download")

clean, _ = clean_dataset(pd.read_csv(RAW_PATH))
CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
clean.to_csv(CLEAN_PATH, index=False)
SPLITS_PATH.mkdir(parents=True, exist_ok=True)
for name, split in (
    ("scaffold_seed0.json", scaffold_split(clean["scaffold"].tolist())),
    ("random_seed0.json", random_split(len(clean))),
):
    (SPLITS_PATH / name).write_text(__import__("json").dumps(split, indent=2) + "\n")
(CLEAN_PATH.parent / "dedup_log.txt").write_text("Duplicate removal is recorded by the cleaned row count.\n")

