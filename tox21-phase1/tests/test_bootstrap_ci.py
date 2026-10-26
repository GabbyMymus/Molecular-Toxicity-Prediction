"""Tests for scripts/06_bootstrap_ci.py on toy cases with known answers (no repo data needed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "06_bootstrap_ci.py"
spec = importlib.util.spec_from_file_location("script_06_bootstrap_ci", SCRIPT)
boot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(boot)


def test_known_auc_and_ap():
    """The scikit-learn docs example: AUC 0.75, AP 5/6."""
    y = np.array([0, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.8])
    roc, ap = boot.weighted_metrics(y, s, np.ones((1, 4)))
    assert roc[0] == pytest.approx(0.75)
    assert ap[0] == pytest.approx(5 / 6)


def test_weights_equal_duplicated_rows_with_ties():
    """A multiplicity vector must give the same metrics as repeating the rows (ties included)."""
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 40)
    s = np.round(rng.random(40), 1)  # many ties
    w = rng.integers(0, 4, (5, 40)).astype(float)
    roc, ap = boot.weighted_metrics(y, s, w)
    for b in range(5):
        rep = np.repeat(np.arange(40), w[b].astype(int))
        if len(set(y[rep])) < 2:
            continue
        assert roc[b] == pytest.approx(roc_auc_score(y[rep], s[rep]))
        assert ap[b] == pytest.approx(average_precision_score(y[rep], s[rep]))


def test_single_class_sample_is_nan():
    roc, ap = boot.weighted_metrics(np.array([0, 1, 1]), np.array([0.2, 0.5, 0.9]), np.array([[0.0, 1.0, 2.0]]))
    assert np.isnan(roc[0]) and np.isnan(ap[0])


def test_resample_weights_are_seeded_multiplicities():
    a = boot.resample_weights(50, 20, seed=3)
    assert a.shape == (20, 50)
    assert np.all(a.sum(axis=1) == 50)
    assert np.array_equal(a, boot.resample_weights(50, 20, seed=3))
    assert not np.array_equal(a, boot.resample_weights(50, 20, seed=4))


def test_nan_labels_are_ignored_not_zero_filled():
    """Endpoint 0 has NaN labels; its metric must equal the metric on its labeled rows only."""
    rng = np.random.default_rng(2)
    n = 60
    Y = rng.integers(0, 2, (n, 2)).astype(float)
    Y[::3, 0] = np.nan
    P = rng.random((n, 2))
    out = boot.mean_over_endpoints(Y, P, np.ones((1, n)))
    keep = ~np.isnan(Y[:, 0])
    expected = np.mean([roc_auc_score(Y[keep, 0], P[keep, 0]), roc_auc_score(Y[:, 1], P[:, 1])])
    assert out["roc_auc"][0] == pytest.approx(expected)


def test_perfect_model_interval_is_one_and_identical_models_difference_is_zero():
    rng = np.random.default_rng(5)
    n = 80
    Y = rng.integers(0, 2, (n, 3)).astype(float)
    perfect = Y * 0.8 + 0.1
    weights = boot.resample_weights(n, 200, seed=0)
    preds = {(m, fs, s): perfect for m in boot.MODELS for fs in boot.FEATURE_SETS for s in boot.SEEDS}
    stats = boot.pooled_statistics(preds, Y, weights)
    intervals, diffs = boot.build_tables(stats)
    assert np.allclose(intervals[["estimate", "ci_low", "ci_high"]].to_numpy(), 1.0)
    assert np.allclose(diffs[["estimate", "ci_low", "ci_high"]].to_numpy(), 0.0)
    assert diffs["includes_zero"].all()


def test_percentile_ci_known_values():
    lo, hi = boot.percentile_ci(np.arange(1001, dtype=float))
    assert (lo, hi) == (pytest.approx(25.0), pytest.approx(975.0))
