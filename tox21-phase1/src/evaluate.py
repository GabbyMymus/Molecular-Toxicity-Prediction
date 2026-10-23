"""Masked endpoint metrics and model selection."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def masked_metrics(y_true, y_score):
    y_true, y_score = np.asarray(y_true, float), np.asarray(y_score, float)
    output = []
    for column in range(y_true.shape[1]):
        mask = np.isfinite(y_true[:, column])
        if mask.sum() == 0 or np.unique(y_true[mask, column]).size < 2:
            auc = pr = float("nan")
        else:
            auc = roc_auc_score(y_true[mask, column], y_score[mask, column])
            pr = average_precision_score(y_true[mask, column], y_score[mask, column])
        output.append({"roc_auc": auc, "pr_auc": pr, "n": int(mask.sum()), "n_pos": int(np.nansum(y_true[mask, column]))})
    return output


def run_model(model, x_train, y_train, x_eval, y_eval):
    model.fit(x_train, y_train)
    prediction = model.predict_proba(x_eval)
    if isinstance(prediction, list):
        prediction = np.column_stack([item[:, 1] for item in prediction])
    return masked_metrics(y_eval, prediction)


def pick_best_config(frame):
    return frame.sort_values(["roc_auc", "pr_auc"], ascending=False).iloc[0].to_dict()

