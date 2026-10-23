from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

PARAM_GRID = [{"max_depth": 4, "learning_rate": 0.05}, {"max_depth": 6, "learning_rate": 0.05}]


def make_xgb(params, scale_pos_weight=1.0):
    return OneVsRestClassifier(XGBClassifier(n_estimators=300, random_state=0, eval_metric="logloss", scale_pos_weight=scale_pos_weight, **params))

