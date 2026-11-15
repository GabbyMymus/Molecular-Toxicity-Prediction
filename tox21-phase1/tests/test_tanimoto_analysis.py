"""Tests for similarity bins and masked metric aggregation (no DL dependencies)."""

import numpy as np
import pandas as pd
import pytest

from src.config import ENDPOINTS
from src.dl.tanimoto import endpoint_bin_metrics, similarity_bins, summarize_bin_metrics


def test_similarity_bins_collapse_duplicate_quantiles_and_keep_ties_together():
    sims = np.array([0.1, 0.1, 0.1, 0.8, 0.9, 0.9])
    ids, edges = similarity_bins(sims, n_bins=4)
    assert len(edges) < 5
    assert len(set(ids[:3])) == 1
    assert len(set(ids[4:])) == 1
    assert np.all(np.diff(ids) >= 0)


def test_similarity_bins_handle_all_values_tied():
    ids, edges = similarity_bins(np.full(5, 0.42), n_bins=4)
    np.testing.assert_array_equal(ids, np.zeros(5, dtype=np.int32))
    np.testing.assert_allclose(edges, [0.42])


@pytest.mark.parametrize(
    "values,n_bins",
    [([], 4), ([0.1, np.nan], 4), ([0.1, 1.1], 4), ([0.1], 0)],
)
def test_similarity_bins_reject_bad_inputs(values, n_bins):
    with pytest.raises(ValueError):
        similarity_bins(np.asarray(values, dtype=float), n_bins=n_bins)


def test_endpoint_bin_metrics_masks_missing_labels_and_averages_seeds():
    y = np.tile(np.array([[0.0], [1.0], [0.0], [1.0]]), (1, len(ENDPOINTS)))
    y[0, 0] = np.nan
    p = np.tile(np.array([[0.1], [0.9], [0.2], [0.8]]), (1, len(ENDPOINTS)))
    sims = np.array([0.1, 0.2, 0.8, 0.9])
    frames = [
        endpoint_bin_metrics(y, p, sims, model="test", split="scaffold", seed=seed, n_bins=2)
        for seed in (0, 1)
    ]
    raw = pd.concat(frames, ignore_index=True)
    summary = summarize_bin_metrics(raw)
    assert len(raw) == 2 * 2 * len(ENDPOINTS)
    assert summary["n_seeds_pr_auc"].eq(2).all()
    assert summary["pr_auc_mean"].eq(1.0).all()
    q1_first_endpoint = raw[(raw.seed == 0) & (raw.bin_id == 0) & (raw.endpoint == ENDPOINTS[0])].iloc[0]
    assert q1_first_endpoint["n_labeled"] == 1
    assert np.isnan(q1_first_endpoint["pr_auc"])


@pytest.mark.parametrize("pred_shape", [(4, 11), (3, 12)])
def test_endpoint_bin_metrics_reject_prediction_alignment_mismatch(pred_shape):
    y = np.zeros((4, len(ENDPOINTS)))
    prediction = np.zeros(pred_shape)
    with pytest.raises(ValueError):
        endpoint_bin_metrics(y, prediction, np.linspace(0, 1, 4), model="x", split="scaffold", seed=0)
