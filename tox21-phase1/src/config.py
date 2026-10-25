# src/config.py
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/tox21.csv.gz"
CLEAN = ROOT / "data/processed/tox21_clean.csv"
SPLITS = ROOT / "data/splits"
FEATS = ROOT / "features"
RESULTS = ROOT / "results"
ENDPOINTS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
             "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]
SEEDS = [0, 1, 2]
FEATURE_SETS = ["morgan", "morgan+desc"]
SPLIT_NAMES = ["scaffold", "random"]

# Compatibility aliases: names already imported by teammates' merged code
# (data.py, splits.py, features.py, scripts/01, scripts/02, tests). Keep until owners migrate.
RAW_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
N_BITS = 2048
RAW_PATH = RAW
CLEAN_PATH = CLEAN
SPLITS_PATH = SPLITS
FEATURES_PATH = FEATS
RESULTS_PATH = RESULTS
