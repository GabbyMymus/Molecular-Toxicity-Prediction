"""Registry and factory tests (tiny synthetic fits)."""

import numpy as np
import pytest

from src.models import REGISTRY
from src.models.xgb import GRID as XGB_GRID
from src.models.xgb import make_xgb


def tiny_data(n=120, d=6, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.5 * rng.normal(size=n) > 0.8).astype(int)
    return X, y


def tiny_params(name, grid):
    """Smallest sensible override of the first config so the fit is fast."""
    params = dict(grid[0])
    if name in ("rf", "xgb"):
        params["n_estimators"] = 5
    return params


def test_registry_has_three_models():
    assert set(REGISTRY) == {"logreg", "rf", "xgb"}


@pytest.mark.parametrize("name", ["logreg", "rf", "xgb"])
def test_grid_within_budget(name):
    _, grid = REGISTRY[name]
    assert 1 <= len(grid) <= 8
    assert all(isinstance(p, dict) for p in grid)


@pytest.mark.parametrize("name", ["logreg", "rf", "xgb"])
def test_factory_returns_fit_predict_proba_and_tiny_fit_works(name):
    make, grid = REGISTRY[name]
    X, y = tiny_data()
    model = make(tiny_params(name, grid), y, 0)
    assert hasattr(model, "fit") and hasattr(model, "predict_proba")
    model.fit(X, y)
    proba = model.predict_proba(X)
    proba = proba[0] if isinstance(proba, list) else np.asarray(proba)
    assert proba.shape == (len(X), 2)
    assert np.all((proba >= 0) & (proba <= 1))


def test_xgb_grid_is_the_handbook_grid():
    assert len(XGB_GRID) == 8
    assert XGB_GRID[0] == dict(max_depth=3, learning_rate=0.10, n_estimators=300)
    assert XGB_GRID[-1] == dict(max_depth=6, learning_rate=0.03, n_estimators=600)


def test_xgb_scale_pos_weight_is_neg_over_pos():
    y = np.array([1] * 5 + [0] * 45)
    model = make_xgb(XGB_GRID[0], y, seed=3)
    assert model.get_params()["scale_pos_weight"] == pytest.approx(45 / 5)


def test_xgb_scale_pos_weight_floors_npos_at_one():
    y = np.zeros(30, dtype=int)
    assert make_xgb(XGB_GRID[0], y, seed=0).get_params()["scale_pos_weight"] == pytest.approx(30.0)


def test_xgb_fixed_settings():
    p = make_xgb(XGB_GRID[7], np.array([0, 1, 0, 0]), seed=7).get_params()
    assert p["subsample"] == 0.8 and p["colsample_bytree"] == 0.5
    assert p["tree_method"] == "hist" and p["eval_metric"] == "aucpr"
    assert p["n_jobs"] == -1 and p["random_state"] == 7 and p["verbosity"] == 0
    assert p["max_depth"] == 6 and p["learning_rate"] == 0.03 and p["n_estimators"] == 600
    assert p.get("early_stopping_rounds") is None
