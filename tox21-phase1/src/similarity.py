"""Tanimoto similarity to the training molecule set."""

import numpy as np


def max_tanimoto_to_train(train_fingerprints, query_fingerprints):
    train = np.asarray(train_fingerprints, dtype=bool)
    query = np.asarray(query_fingerprints, dtype=bool)
    result = []
    for row in query:
        intersection = np.logical_and(train, row).sum(axis=1)
        union = np.logical_or(train, row).sum(axis=1)
        result.append(float(np.max(np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0))))
    return np.asarray(result)

