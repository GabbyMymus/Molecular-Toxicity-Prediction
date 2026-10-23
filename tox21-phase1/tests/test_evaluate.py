import numpy as np

from src.evaluate import masked_metrics


def test_nan_labels_are_excluded():
    metrics = masked_metrics([[1, np.nan], [0, 1]], [[0.9, 0.5], [0.1, 0.6]])
    assert metrics[0]["n"] == 2
    assert metrics[1]["n"] == 1
    assert np.isnan(metrics[1]["roc_auc"])
