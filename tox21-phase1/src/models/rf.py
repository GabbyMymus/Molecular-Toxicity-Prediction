"""Random Forest classifier for multi-task molecular toxicity prediction.

Features:
- Addresses extreme class imbalance using `class_weight="balanced_subsample"`.
- Uses `max_features="sqrt"`, `n_jobs=-1`, and explicit random seeds for reproducibility.
- Handles multi-task binary toxicity endpoints with missing (NaN) labels by
  training each endpoint only on its observed labels.
- Integrates seamlessly with the shared evaluation harness `src.evaluate.run_model`.
- Fixed 8-configuration hyperparameter tuning grid.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)

# Fixed 8-configuration hyperparameter tuning grid
GRID: list[dict[str, Any]] = (
    [
        {
            "n_estimators": 300,
            "max_depth": depth,
            "min_samples_leaf": leaf,
        }
        for depth in (None, 30)
        for leaf in (1, 3, 5)
    ]
    + [
        {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 3,
        },
        {
            "n_estimators": 500,
            "max_depth": 30,
            "min_samples_leaf": 5,
        },
    ]
)

# Alias for backwards compatibility with existing codebase conventions
PARAM_GRID = GRID


class MultiTaskRandomForest(BaseEstimator, ClassifierMixin):
    """Multi-task wrapper around scikit-learn's RandomForestClassifier.

    Fits independent binary Random Forest classifiers per endpoint while skipping
    missing (NaN) target labels on a per-task basis.
    """

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        seed: int = 0,
        class_weight: str = "balanced_subsample",
        max_features: str = "sqrt",
        n_jobs: int = -1,
        **extra_params: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.seed = seed
        self.class_weight = class_weight
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.extra_params = extra_params
        self.estimators_: list[RandomForestClassifier | None] = []
        self.n_tasks_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> MultiTaskRandomForest:
        """Fit an independent Random Forest model for each endpoint.

        Parameters
        ----------
        X : np.ndarray
            Input features (shape (n_samples, n_features)).
        y : np.ndarray
            Target labels (shape (n_samples, n_tasks) or (n_samples,)).
            Can contain NaN values representing unmeasured endpoints.
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y, dtype=float)

        if y_arr.ndim == 1:
            y_arr = y_arr[:, None]

        self.n_tasks_ = y_arr.shape[1]
        self.estimators_ = []

        for j in range(self.n_tasks_):
            valid_mask = np.isfinite(y_arr[:, j])
            n_valid = int(valid_mask.sum())

            if n_valid == 0:
                logger.warning(f"Task {j} has 0 observed labels; skipping.")
                self.estimators_.append(None)
                continue

            y_sub = y_arr[valid_mask, j].astype(int)
            unique_classes = np.unique(y_sub)

            if unique_classes.size < 2:
                logger.warning(
                    f"Task {j} has insufficient class diversity (unique classes: {unique_classes}); "
                    f"skipping."
                )
                self.estimators_.append(None)
                continue

            rf = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                class_weight=self.class_weight,
                max_features=self.max_features,
                n_jobs=self.n_jobs,
                random_state=self.seed + j,
                **self.extra_params,
            )
            rf.fit(X_arr[valid_mask], y_sub)
            self.estimators_.append(rf)

        return self

    def predict_proba(self, X: np.ndarray) -> list[np.ndarray]:
        """Compute class probabilities for all endpoints.

        Returns a list of length n_tasks, where each element is an array of
        shape (n_samples, 2) giving [P(class=0), P(class=1)].
        This format directly satisfies `evaluate.py:run_model`.
        """
        X_arr = np.asarray(X)
        n_samples = X_arr.shape[0]
        results: list[np.ndarray] = []

        for j, rf in enumerate(self.estimators_):
            if rf is None:
                # Return NaN probabilities if the task could not be trained
                results.append(np.full((n_samples, 2), np.nan, dtype=float))
            else:
                probs = rf.predict_proba(X_arr)
                if probs.shape[1] == 1:
                    # In case only one class is present in predictions
                    cls = rf.classes_[0]
                    if cls == 1:
                        probs = np.column_stack([np.zeros(n_samples), probs[:, 0]])
                    else:
                        probs = np.column_stack([probs[:, 0], np.zeros(n_samples)])
                results.append(probs)

        return results

    @property
    def feature_importances_(self) -> np.ndarray:
        """Average feature importances across all successfully fitted tasks."""
        valid_importances = [
            rf.feature_importances_
            for rf in self.estimators_
            if rf is not None and hasattr(rf, "feature_importances_")
        ]
        if not valid_importances:
            raise ValueError("No fitted estimators available to compute feature importances.")
        return np.mean(valid_importances, axis=0)


def make_rf(
    params: dict[str, Any] | None = None,
    y_train: np.ndarray | None = None,
    seed: int = 0,
) -> MultiTaskRandomForest:
    """Factory function for constructing a MultiTaskRandomForest classifier.

    Parameters
    ----------
    params : dict[str, Any] | None
        Hyperparameters dictionary (e.g. n_estimators, max_depth, min_samples_leaf).
    y_train : np.ndarray | None
        Optional training labels for shape and task inspection.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    MultiTaskRandomForest
        Configured classifier ready for fitting.
    """
    p = dict(params or {})
    return MultiTaskRandomForest(seed=seed, **p)
