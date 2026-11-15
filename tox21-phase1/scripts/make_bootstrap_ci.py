"""Recompute the paired-bootstrap intervals quoted in report_phase2/sections/results.tex.

Inputs: committed seed-0 scaffold test predictions in results/preds/ and the test labels of the scaffold split.
Method: src.dl.analysis.paired_bootstrap_ci (PR AUC difference, test rows resampled jointly, 95% percentile
interval) with N_RESAMPLES resamples and the fixed RNG seed RNG_SEED below, per endpoint.
Output: results/dl_bootstrap_ci.csv

Run from tox21-phase1/:  python scripts/make_bootstrap_ci.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ENDPOINTS, RESULTS  # noqa: E402
from src.data import load_clean  # noqa: E402
from src.dl import analysis as A  # noqa: E402
from src.dl.split_io import load_split_named  # noqa: E402

RNG_SEED = 0          # passed to paired_bootstrap_ci as `seed`
N_RESAMPLES = 1000
PREDS = {
    "h1": "mpnn+desc_graph+desc_scaffold_seed0_final.npy",
    "fpmlp": "fpmlp+both_both_scaffold_seed0_final.npy",
    "h2_primary": "ens_xgb_fpmlp+both_morgan+desc+both_scaffold_seed0_final.npy",
    "h2_secondary": "ens_xgb_mpnn+desc_morgan+desc+graph+desc_scaffold_seed0_final.npy",
    "xgb": "xgb_morgan+desc_scaffold_seed0.npy",
}
COMPARISONS = [  # (name, a, b): interval of PR AUC(a) - PR AUC(b)
    ("h1_minus_fpmlp", "h1", "fpmlp"),
    ("h2_primary_minus_h1", "h2_primary", "h1"),
    ("h2_secondary_minus_h1", "h2_secondary", "h1"),
    ("h2_primary_minus_xgb", "h2_primary", "xgb"),
    ("h2_secondary_minus_xgb", "h2_secondary", "xgb"),
]


def main() -> None:
    print(f"rng_seed={RNG_SEED} n_resamples={N_RESAMPLES}")
    _, y_all = load_clean()
    y = y_all[load_split_named("scaffold")["test"]]
    preds = {k: np.load(RESULTS / "preds" / v) for k, v in PREDS.items()}
    for k, p in preds.items():
        assert p.shape == y.shape, (k, p.shape, y.shape)
    rows = []
    for name, a, b in COMPARISONS:
        for j, ep in enumerate(ENDPOINTS):
            d, lo, hi = A.paired_bootstrap_ci(y[:, j], preds[a][:, j], preds[b][:, j], "pr_auc",
                                              n_boot=N_RESAMPLES, seed=RNG_SEED)
            rows.append(dict(comparison=name, endpoint=ep, delta_mean=d, ci_low=lo, ci_high=hi,
                             excludes_zero=bool(lo > 0 or hi < 0), n_resamples=N_RESAMPLES, rng_seed=RNG_SEED))
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "dl_bootstrap_ci.csv", index=False)
    print("wrote results/dl_bootstrap_ci.csv", out.shape)
    for name, g in out.groupby("comparison", sort=False):
        pos = int((g.ci_low > 0).sum()); neg = int((g.ci_high < 0).sum())
        print(f"{name}: excludes zero on {int(g.excludes_zero.sum())}/12 (above 0: {pos}, below 0: {neg})")
    print(out[(out.comparison == "h1_minus_fpmlp") & out.excludes_zero].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
