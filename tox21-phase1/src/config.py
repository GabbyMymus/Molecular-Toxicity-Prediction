"""Pipeline constants and paths."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz"
ENDPOINTS = (
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER",
    "NR-ER-LBD", "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE",
    "SR-MMP", "SR-p53",
)
SEEDS = (0, 1, 2)
N_BITS = 2048
RAW_PATH = ROOT / "data/raw/tox21.csv.gz"
CLEAN_PATH = ROOT / "data/processed/tox21_clean.csv"
SPLITS_PATH = ROOT / "data/splits"
FEATURES_PATH = ROOT / "features"
RESULTS_PATH = ROOT / "results"

