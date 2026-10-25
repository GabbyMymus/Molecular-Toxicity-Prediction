
"""Logistic Regression baseline for the Tox21 multi-task classification project."""

from sklearn.linear_model import LogisticRegression


# Six L2 configurations and two L1 configurations.
# Hyperparameters are selected using validation PR-AUC by the shared harness.
GRID = (
    [{"C": c, "l1_ratio": 0.0} for c in (0.01, 0.03, 0.1, 0.3, 1, 3)]
    + [
        {"C": 0.1, "l1_ratio": 1.0},
        {"C": 1, "l1_ratio": 1.0},
    ]
)

PARAM_GRID = GRID


def make_logreg(params, y_train=None, seed=0):
    return LogisticRegression(
        solver="liblinear",
        class_weight="balanced",
        max_iter=2000,
        random_state=seed,
        **params,
    )