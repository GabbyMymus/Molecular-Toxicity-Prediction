"""Tests for the shared evaluation harness (synthetic data only)."""

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression

from src import evaluate
from src.config import ENDPOINTS
from src.evaluate import fit_predict, final_test, masked_metrics, score, tune

K = len(ENDPOINTS)


def make_dummy(params, y_train, seed):
    return DummyClassifier(strategy="prior")


def make_lr(params, y_train, seed):
    return LogisticRegression(random_state=seed, **params)


def synthetic(n_tr=400, n_ev=2000, rate=0.1, seed=0):
    rng = np.random.default_rng(seed)
    Xtr, Xev = rng.normal(size=(n_tr, 5)), rng.normal(size=(n_ev, 5))
    Ytr = (rng.random((n_tr, K)) < rate).astype(float)
    Yev = (rng.random((n_ev, K)) < rate).astype(float)
    return Xtr, Ytr, Xev, Yev


def test_nan_labels_are_excluded():
    y = np.array([1, 0, 1, 0, np.nan])
    s = np.array([0.9, 0.2, 0.8, 0.1, 0.99])
    # without the NaN row the ranking is perfect; if NaN were treated as 0 the 0.99 negative would break it
    assert masked_metrics(y, s) == (1.0, 1.0)
    assert roc_if_filled_with_zero(y, s) < 1.0


def roc_if_filled_with_zero(y, s):
    from sklearn.metrics import roc_auc_score

    return roc_auc_score(np.nan_to_num(y, nan=0.0), s)


def test_single_class_returns_nan():
    for y in (np.ones(5), np.zeros(5), np.array([1, 1, np.nan, np.nan]), np.full(4, np.nan), np.array([])):
        roc, pr = masked_metrics(y, np.linspace(0, 1, len(y)))
        assert np.isnan(roc) and np.isnan(pr)


def test_perfect_ranking():
    assert masked_metrics(np.array([0, 0, 1, 1, 0]), np.array([0.1, 0.2, 0.9, 0.8, 0.3])) == (1.0, 1.0)


def test_dummy_prior_gives_chance_level():
    Xtr, Ytr, Xev, Yev = synthetic(rate=0.1)
    P = fit_predict(make_dummy, {}, Xtr, Ytr, Xev, seed=0)
    rows = score(Yev, P)
    assert [r["endpoint"] for r in rows] == list(ENDPOINTS)
    for r in rows:
        assert r["roc_auc"] == pytest.approx(0.5, abs=0.02)  # constant scores
        assert r["pr_auc"] == pytest.approx(Yev[:, 0].mean(), abs=0.05)
        assert r["n_test"] == len(Yev)


def test_score_counts_only_labeled_rows():
    Y = np.full((6, K), np.nan)
    Y[:4, 0] = [1, 0, 0, 1]
    rows = score(Y, np.random.default_rng(0).random((6, K)))
    assert rows[0]["n_test"] == 4 and rows[0]["n_pos_test"] == 2
    assert rows[1]["n_test"] == 0 and np.isnan(rows[1]["roc_auc"])


def test_fit_predict_survives_endpoint_without_positives():
    Xtr, Ytr, Xev, _ = synthetic()
    Ytr[:, 0] = 0.0  # no positives in train
    Ytr[:, 1] = np.nan  # no labels at all
    P = fit_predict(make_lr, {"C": 1.0}, Xtr, Ytr, Xev, seed=0)
    assert np.all(P[:, 0] == 0.0)  # filled with the train positive rate
    assert np.isnan(P[:, 1]).all()
    assert np.isfinite(P[:, 2:]).all()


def test_fit_predict_single_class_fills_positive_rate():
    Xtr, Ytr, Xev, _ = synthetic()
    Ytr[:, 3] = 1.0
    P = fit_predict(make_dummy, {}, Xtr, Ytr, Xev, seed=0)
    assert np.all(P[:, 3] == 1.0)


def test_fit_predict_does_not_fill_nan_labels_with_zero():
    seen = []

    class Spy(DummyClassifier):
        def fit(self, X, y, **kw):
            seen.append(len(y))
            return super().fit(X, y, **kw)

    Xtr, Ytr, Xev, _ = synthetic(n_tr=100)
    Ytr[:30, 0] = np.nan
    fit_predict(lambda p, y, s: Spy(strategy="prior"), {}, Xtr, Ytr, Xev, seed=0)
    assert seen[0] == 70


class OneColumn:
    """Model whose predict_proba returns a single column."""

    def fit(self, X, y):
        self.classes_ = np.array([int(y[0])])
        return self

    def predict_proba(self, X):
        return np.ones((len(X), 1))


def test_single_column_predict_proba_is_guarded():
    Xtr, Ytr, Xev, _ = synthetic()
    P = fit_predict(lambda p, y, s: OneColumn(), {}, Xtr, Ytr, Xev, seed=0)
    assert P.shape == (len(Xev), K) and np.isfinite(P).all()


class ListProba:
    """Multi-task style model: predict_proba returns a list of (n, 2) arrays."""

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        return [np.column_stack([np.full(len(X), 0.7), np.full(len(X), 0.3)])]


def test_list_valued_predict_proba_is_handled():
    Xtr, Ytr, Xev, _ = synthetic()
    P = fit_predict(lambda p, y, s: ListProba(), {}, Xtr, Ytr, Xev, seed=0)
    assert np.allclose(P, 0.3)


@pytest.fixture
def fake_data(tmp_path, monkeypatch):
    Xtr, Ytr, Xva, Yva = synthetic(n_tr=300, n_ev=300, rate=0.2, seed=1)
    Xte, Yte = synthetic(n_tr=1, n_ev=200, rate=0.2, seed=2)[2:]
    monkeypatch.setattr(evaluate, "RESULTS", tmp_path)
    monkeypatch.setattr(evaluate, "_DATA_CACHE", {("morgan", "scaffold"): (Xtr, Xva, Xte, Ytr, Yva, Yte)})
    return tmp_path


def test_tune_is_idempotent_and_picks_best(fake_data):
    grid = [{"C": 1e-6}, {"C": 1.0}]
    k1 = tune("lr", make_lr, grid, "morgan")
    first = pd.read_csv(fake_data / "tuning_val.csv")
    k2 = tune("lr", make_lr, grid, "morgan")
    second = pd.read_csv(fake_data / "tuning_val.csv")
    assert k1 == k2 and len(first) == len(second) == len(grid) * K
    assert list(second.columns) == evaluate.TUNING_COLUMNS
    assert set(second["seed"]) == {0} and set(second["split"]) == {"scaffold"}
    tune("dummy", make_dummy, [{}], "morgan")  # a different model keeps its own rows
    assert len(pd.read_csv(fake_data / "tuning_val.csv")) == (len(grid) + 1) * K
    assert all(json.loads(p) in grid for p in second["params"])


def test_final_test_saves_predictions_and_columns(fake_data):
    df = final_test("dummy", make_dummy, {}, "morgan", "scaffold", seed=1)
    assert list(df.columns) == evaluate.RESULT_COLUMNS and len(df) == K
    P = np.load(fake_data / "preds" / "dummy_morgan_scaffold_seed1.npy")
    assert P.shape == (200, K)
    assert (df["seed"] == 1).all() and (df["n_test"] == 200).all()
