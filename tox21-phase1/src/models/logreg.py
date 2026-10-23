from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

PARAM_GRID = [{"C": 0.1}, {"C": 1.0}, {"C": 10.0}]


def make_logreg(params):
    return OneVsRestClassifier(LogisticRegression(max_iter=1000, class_weight="balanced", **params))

