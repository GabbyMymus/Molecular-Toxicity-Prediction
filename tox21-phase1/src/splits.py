"""Scaffold and random train/validation/test splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import SPLITS_PATH


def _pack(train, val, test):
    return {"train": list(map(int, train)), "val": list(map(int, val)), "test": list(map(int, test))}


def random_split(n: int, seed: int = 0) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n)
    n_test, n_val = max(1, round(n * 0.1)), max(1, round(n * 0.1))
    return _pack(indices[: n - n_val - n_test], indices[n - n_val - n_test : n - n_test], indices[-n_test:])


def scaffold_split(scaffolds: list[str], seed: int = 0) -> dict[str, list[int]]:
    groups = {}
    for index, scaffold in enumerate(scaffolds):
        groups.setdefault(scaffold, []).append(index)
    groups = list(groups.values())
    np.random.default_rng(seed).shuffle(groups)
    target_test = max(1, round(len(scaffolds) * 0.1))
    target_val = max(1, round(len(scaffolds) * 0.1))
    test, val, train = [], [], []
    for group in groups:
        if len(test) < target_test:
            test.extend(group)
        elif len(val) < target_val:
            val.extend(group)
        else:
            train.extend(group)
    return _pack(train, val, test)


def load_split(path: Path) -> dict[str, list[int]]:
    with path.open() as handle:
        split = json.load(handle)
    if set(split) != {"train", "val", "test"}:
        raise ValueError("Split must contain train, val, and test")
    return split
