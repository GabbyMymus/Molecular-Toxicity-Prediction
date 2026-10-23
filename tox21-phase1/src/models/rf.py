from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier

PARAM_GRID = [{"n_estimators": 200, "max_depth": None}, {"n_estimators": 300, "max_depth": 12}]


def make_rf(params):
    return OneVsRestClassifier(RandomForestClassifier(random_state=0, class_weight="balanced", n_jobs=-1, **params))

