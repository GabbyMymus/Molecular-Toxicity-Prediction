"""Tests for Malhar's prediction QA summaries."""

import numpy as np
import pandas as pd
import pytest

from src.dl.member5_analysis import seed_variance_table


def test_seed_variance_uses_endpoint_average_per_seed():
    frame = pd.DataFrame(
        [
            {"model": "mpnn", "feature_set": "graph", "split": "scaffold", "seed": seed,
             "roc_auc": roc, "pr_auc": pr}
            for seed, roc, pr in [(0, 0.7, 0.3), (0, 0.9, 0.5), (1, 0.6, 0.2), (1, 0.8, 0.4)]
        ]
    )
    out = seed_variance_table(frame).iloc[0]
    assert out["n_seeds"] == 2
    assert out["roc_auc_mean"] == pytest.approx(0.75)
    assert out["pr_auc_mean"] == pytest.approx(0.35)
    assert out["roc_auc_seed_sd"] == pytest.approx(np.std([0.8, 0.7], ddof=1))


def test_seed_variance_requires_all_summary_columns():
    with pytest.raises(ValueError, match="missing required"):
        seed_variance_table(pd.DataFrame({"model": ["mpnn"]}))
