"""XGBoost factory: one binary classifier per endpoint, fixed 8-configuration grid."""

from __future__ import annotations

from typing import Any

import numpy as np
from xgboost import XGBClassifier

GRID: list[dict[str, Any]] = [
    dict(max_depth=3, learning_rate=0.10, n_estimators=300),
    dict(max_depth=5, learning_rate=0.10, n_estimators=300),
    dict(max_depth=7, learning_rate=0.10, n_estimators=300),
    dict(max_depth=3, learning_rate=0.05, n_estimators=500),
    dict(max_depth=5, learning_rate=0.05, n_estimators=500),
    dict(max_depth=7, learning_rate=0.05, n_estimators=500),
    dict(max_depth=4, learning_rate=0.10, n_estimators=400),
    dict(max_depth=6, learning_rate=0.03, n_estimators=600),
]  # 8 configs = the tuning budget


def make_xgb(params: dict[str, Any], y_train: np.ndarray, seed: int) -> XGBClassifier:
    """Build an XGBClassifier for one endpoint.

    ``scale_pos_weight = n_neg / n_pos`` is computed from this endpoint's labeled training
    labels (``y_train``, 1-D, values 0/1) with ``n_pos`` floored at 1. No early stopping.
    """
    y_train = np.asarray(y_train)
    n_pos = max(int(y_train.sum()), 1)
    n_neg = len(y_train) - int(y_train.sum())
    return XGBClassifier(
        **params,
        subsample=0.8,
        colsample_bytree=0.5,
        scale_pos_weight=n_neg / n_pos,
        tree_method="hist",
        eval_metric="aucpr",
        n_jobs=-1,
        random_state=seed,
        verbosity=0,
    )
