"""Final evaluation entry point; requires frozen tuning configurations."""

import json

from src.config import RESULTS_PATH

best = RESULTS_PATH / "best_configs.json"
if not best.exists():
    raise FileNotFoundError("Run `make tune` before final testing")
json.loads(best.read_text())

