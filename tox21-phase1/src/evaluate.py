"""Shared evaluation harness: masked metrics, per-endpoint fitting, tuning, final test.

Every model goes through this code so all metrics are computed the same way, with
missing labels (NaN) excluded. See CONTRACT.md.

Model factories follow ``make_model(params, y_train, seed)`` and return an object with
``fit(X, y)`` and ``predict_proba(X)``. The harness calls it once per endpoint with a 1-D
label vector of labeled training rows only.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import ENDPOINTS, RESULTS, SPLITS
from src.data import load_clean
from src.features import load_features
from src.splits import load_split

logger = logging.getLogger(__name__)

TUNE_SEED = 0
SPLIT_SEED = 0
RESULT_COLUMNS = ["model", "feature_set", "split", "seed", "endpoint", "roc_auc", "pr_auc", "n_test", "n_pos_test"]
TUNING_COLUMNS = [
    "model", "feature_set", "split", "seed", "config_id", "params", "endpoint",
    "roc_auc", "pr_auc", "n_test", "n_pos_test",
]

ModelFactory = Callable[[dict[str, Any], np.ndarray, int], Any]

# (feature_set, split) -> (X_train, X_val, X_test, Y_train, Y_val, Y_test)
_DATA_CACHE: dict[tuple[str, str], tuple[np.ndarray, ...]] = {}


def masked_metrics(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Return ``(roc_auc, pr_auc)`` for one endpoint, ignoring rows with a NaN label.

    Returns ``(nan, nan)`` if no labeled rows remain or only one class is present.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    keep = ~np.isnan(y_true)
    y, s = y_true[keep], y_score[keep]
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return np.nan, np.nan
    if not np.isfinite(s).all():
        logger.warning("non-finite scores on labeled rows; metric set to NaN")
        return np.nan, np.nan
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def _positive_proba(model: Any, X: np.ndarray, fallback: float) -> np.ndarray:
    """Return P(class 1) as a 1-D array, tolerating odd ``predict_proba`` outputs.

    Handles a list of per-task arrays (multi-task wrappers) and a single-column output.
    """
    proba = model.predict_proba(X)
    if isinstance(proba, list):
        proba = proba[0]
    proba = np.asarray(proba, dtype=float)
    if proba.ndim == 1:
        return proba
    if proba.shape[1] >= 2:
        return proba[:, 1]
    classes = getattr(model, "classes_", None)
    if classes is not None and len(classes) == 1:
        return proba[:, 0] if classes[0] == 1 else 1.0 - proba[:, 0]
    logger.warning("predict_proba returned one column and the class is unknown; using positive rate %.4f", fallback)
    return np.full(len(X), fallback)


def fit_predict(
    make_model: ModelFactory,
    params: dict[str, Any],
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_ev: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Train one classifier per endpoint on its labeled train rows and predict ``X_ev``.

    Returns an ``(n_eval, n_endpoints)`` probability array. An endpoint whose labeled
    train rows contain a single class (or none) is not fitted: a warning is logged and its
    column is filled with the train positive rate (NaN if there are no labeled rows).
    """
    P = np.full((X_ev.shape[0], len(ENDPOINTS)), np.nan)
    for j in range(len(ENDPOINTS)):
        keep = ~np.isnan(Y_tr[:, j])
        y = Y_tr[keep, j].astype(int)
        if len(y) == 0:
            logger.warning("endpoint %s: no labeled train rows; predictions left NaN", ENDPOINTS[j])
            continue
        rate = float(y.mean())
        if y.min() == y.max():
            logger.warning(
                "endpoint %s: single class in labeled train rows; filling with positive rate %.4f", ENDPOINTS[j], rate
            )
            P[:, j] = rate
            continue
        model = make_model(params, y, seed)
        model.fit(X_tr[keep], y)
        P[:, j] = _positive_proba(model, X_ev, rate)
    return P


def score(Y_ev: np.ndarray, P: np.ndarray) -> list[dict[str, Any]]:
    """Per-endpoint metrics on labeled rows only: one dict per endpoint."""
    rows = []
    for j, endpoint in enumerate(ENDPOINTS):
        roc, pr = masked_metrics(Y_ev[:, j], P[:, j])
        labeled = ~np.isnan(Y_ev[:, j])
        rows.append(
            dict(
                endpoint=endpoint,
                roc_auc=roc,
                pr_auc=pr,
                n_test=int(labeled.sum()),
                n_pos_test=int(np.nansum(Y_ev[:, j])),
            )
        )
    return rows


def _load_arrays(feature_set: str, split: str) -> tuple[np.ndarray, ...]:
    """Load (and cache per process) features and labels for a feature set and split.

    Rows follow the saved split index order; labels come from ``load_clean`` in clean-CSV order.
    """
    key = (feature_set, split)
    if key not in _DATA_CACHE:
        _, Y = load_clean()
        idx = {k: np.asarray(v, dtype=int) for k, v in load_split(SPLITS / f"{split}_seed{SPLIT_SEED}.json").items()}
        Xtr, Xva, Xte = load_features(feature_set, split, seed=SPLIT_SEED)
        for name, X in (("train", Xtr), ("val", Xva), ("test", Xte)):
            if len(X) != len(idx[name]):
                raise ValueError(f"{feature_set}/{split}: {name} has {len(X)} feature rows, split has {len(idx[name])}")
        _DATA_CACHE[key] = (Xtr, Xva, Xte, Y[idx["train"]], Y[idx["val"]], Y[idx["test"]])
    return _DATA_CACHE[key]


def clear_cache() -> None:
    """Drop cached feature/label arrays."""
    _DATA_CACHE.clear()


def _replace_rows(new: pd.DataFrame, path, keys: Sequence[str]) -> None:
    """Write ``new`` to ``path``, first removing existing rows that share ``keys`` values with it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        old = pd.read_csv(path)
        if set(keys) <= set(old.columns) and len(old):
            combos = new[list(keys)].drop_duplicates()
            merged = old.merge(combos.assign(_drop=1), on=list(keys), how="left")
            old = old[merged["_drop"].isna().to_numpy()]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(path, index=False)


def tune(
    name: str,
    make_model: ModelFactory,
    grid: Sequence[dict[str, Any]],
    feature_set: str,
    split: str = "scaffold",
) -> int:
    """VALIDATION ONLY. Seed 0. Return the ``config_id`` with the best mean validation PR AUC.

    Appends per-endpoint rows to ``results/tuning_val.csv``; rows already stored for the same
    (model, feature_set, split, seed) are replaced, so reruns never duplicate.
    """
    Xtr, Xva, _, Ytr, Yva, _ = _load_arrays(feature_set, split)
    rows = []
    for k, params in enumerate(grid):
        P = fit_predict(make_model, params, Xtr, Ytr, Xva, seed=TUNE_SEED)
        for r in score(Yva, P):
            rows.append(
                dict(model=name, feature_set=feature_set, split=split, seed=TUNE_SEED, config_id=k,
                     params=json.dumps(params), **r)
            )
    df = pd.DataFrame(rows, columns=TUNING_COLUMNS)
    _replace_rows(df, RESULTS / "tuning_val.csv", ["model", "feature_set", "split", "seed"])
    mean_pr = df.groupby("config_id")["pr_auc"].apply(lambda s: np.nanmean(s) if s.notna().any() else np.nan)
    if mean_pr.isna().all():
        logger.warning("%s|%s: every config has NaN validation PR AUC; defaulting to config 0", name, feature_set)
        return 0
    return int(mean_pr.idxmax())


def final_test(
    name: str,
    make_model: ModelFactory,
    params: dict[str, Any],
    feature_set: str,
    split: str,
    seed: int,
) -> pd.DataFrame:
    """TEST SET. Call only from scripts/04_final_test.py after best_configs.json is committed.

    Saves test probabilities to ``results/preds/<model>_<feature_set>_<split>_seed<k>.npy``
    (rows in the order of the split's test indices) and returns per-endpoint metrics.
    """
    Xtr, _, Xte, Ytr, _, Yte = _load_arrays(feature_set, split)
    P = fit_predict(make_model, params, Xtr, Ytr, Xte, seed)
    preds_dir = RESULTS / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / f"{name}_{feature_set}_{split}_seed{seed}.npy", P)
    df = pd.DataFrame(score(Yte, P))
    df = df.assign(model=name, feature_set=feature_set, split=split, seed=seed)
    return df[RESULT_COLUMNS]


# --- Backwards-compatible helpers used by already-merged code (tests/test_features.py) ---


def masked_metrics_matrix(y_true: np.ndarray, y_score: np.ndarray) -> list[dict[str, Any]]:
    """Column-wise masked metrics for a 2-D label matrix (legacy ``run_model`` format)."""
    y_true, y_score = np.asarray(y_true, float), np.asarray(y_score, float)
    out = []
    for j in range(y_true.shape[1]):
        roc, pr = masked_metrics(y_true[:, j], y_score[:, j])
        labeled = ~np.isnan(y_true[:, j])
        out.append(dict(roc_auc=roc, pr_auc=pr, n=int(labeled.sum()), n_pos=int(np.nansum(y_true[:, j]))))
    return out


def run_model(model: Any, x_train: np.ndarray, y_train: np.ndarray, x_eval: np.ndarray, y_eval: np.ndarray):
    """Fit a multi-task model on 2-D labels and return legacy per-column metrics."""
    model.fit(x_train, y_train)
    prediction = model.predict_proba(x_eval)
    if isinstance(prediction, list):
        prediction = np.column_stack([item[:, 1] for item in prediction])
    return masked_metrics_matrix(y_eval, prediction)


def pick_best_config(frame: pd.DataFrame) -> dict[str, Any]:
    """Return the row with the highest ROC AUC (ties broken by PR AUC); legacy helper."""
    return frame.sort_values(["roc_auc", "pr_auc"], ascending=False).iloc[0].to_dict()
