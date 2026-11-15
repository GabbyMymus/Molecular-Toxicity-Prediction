"""Print trainable parameter counts of the frozen-config models, built with the repo's own constructors.

Reads results/best_configs_dl.json and the feature dimensions from src.features.load_features, builds each model
with src.dl.run.build_model, and prints sum(p.numel() for p in model.parameters()).

Run from tox21-phase1/:  python scripts/count_params.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dl.run import N_MORGAN, build_model  # noqa: E402
from src.features import load_features  # noqa: E402


def n_params(model) -> int:
    return sum(p.numel() for p in model.parameters())


def main() -> None:
    cfg = json.loads((ROOT / "results" / "best_configs_dl.json").read_text())
    n_total = load_features("morgan+desc", "scaffold")[0].shape[1]      # Morgan bits + descriptors
    n_desc = n_total - N_MORGAN
    rows = [
        ("D2 MPNN (mpnn|none)", "mpnn", cfg["mpnn|none"], 0),
        ("H1 MPNN+desc (mpnn|none config)", "mpnn", cfg["mpnn|none"], n_desc),
        ("D1 GCN (gcn|none)", "gcn", cfg["gcn|none"], 0),
        ("D0 FP-MLP both (fpmlp|both)", "fpmlp", cfg["fpmlp|both"], n_total),
    ]
    print(f"descriptor columns: {n_desc}; Morgan+descriptor inputs: {n_total}")
    for label, name, c, extra_dim in rows:
        print(f"{label}: {n_params(build_model(name, c, extra_dim)):,}")


if __name__ == "__main__":
    main()
