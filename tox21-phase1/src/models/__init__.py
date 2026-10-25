"""Model registry: name -> (factory, grid). Factories follow ``make(params, y_train, seed)``."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np

from src.models import logreg, rf, xgb
from src.models.rf import make_rf
from src.models.xgb import make_xgb


def _as_contract_factory(factory: Callable[..., Any]) -> Callable[[dict[str, Any], np.ndarray, int], Any]:
    """Return ``factory`` unchanged if it already takes (params, y_train, seed), else wrap it.

    Raghav's current ``make_logreg(params)`` takes only ``params``; the wrapper drops
    ``y_train`` and ``seed``. Once his file follows the contract this is a no-op.
    """
    n_params = len(inspect.signature(factory).parameters)
    if n_params >= 3:
        return factory

    def wrapped(params: dict[str, Any], y_train: np.ndarray, seed: int) -> Any:
        return factory(params)

    wrapped.__name__ = factory.__name__
    return wrapped


make_logreg = _as_contract_factory(logreg.make_logreg)
# logreg.GRID is the contract name; fall back to the legacy PARAM_GRID until Raghav renames it.
LOGREG_GRID = getattr(logreg, "GRID", getattr(logreg, "PARAM_GRID", None))

REGISTRY: dict[str, tuple[Callable[..., Any], list[dict[str, Any]]]] = {
    "logreg": (make_logreg, LOGREG_GRID),
    "rf": (make_rf, rf.GRID),
    "xgb": (make_xgb, xgb.GRID),
}
