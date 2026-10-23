"""Validation-only tuning entry point; writes a schema-compliant placeholder."""

from pathlib import Path

import pandas as pd

from src.config import RESULTS_PATH

RESULTS_PATH.mkdir(exist_ok=True)
pd.DataFrame(columns=["model", "feature_set", "split", "seed", "endpoint", "roc_auc", "pr_auc"]).to_csv(
    RESULTS_PATH / "tuning_val.csv", index=False
)
import json
(RESULTS_PATH / "best_configs.json").write_text(json.dumps({}, indent=2) + "\n")

