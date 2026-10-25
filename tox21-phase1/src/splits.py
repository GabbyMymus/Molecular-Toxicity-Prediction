"""Deterministic scaffold and random train/validation/test splits."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import SPLITS_PATH


def _pack(train, val, test):
    return {"train": sorted(map(int, train)), "val": sorted(map(int, val)), "test": sorted(map(int, test))}


def random_split(n: int, seed: int = 0) -> dict[str, list[int]]:
    """Return a deterministic 80/10/10 row split with stable sorted indices."""
    if n < 3:
        raise ValueError("At least three rows are required")
    indices = np.random.default_rng(seed).permutation(n)
    train_end, val_end = int(0.8 * n), int(0.9 * n)
    return _pack(indices[:train_end], indices[train_end:val_end], indices[val_end:])


def scaffold_split(scaffolds: list[str], seed: int = 0) -> dict[str, list[int]]:
    """Greedily place largest whole scaffold groups against 80/10/10 cutoffs.

    Ties are stable by first row index. The empty scaffold is one group, so all
    acyclic molecules remain in the same partition.
    """
    n = len(scaffolds)
    if n < 3:
        raise ValueError("At least three rows are required")
    grouped: dict[str, list[int]] = {}
    for index, scaffold in enumerate(scaffolds):
        grouped.setdefault(scaffold or "", []).append(index)
    groups = sorted(grouped.values(), key=lambda rows: (len(rows), -rows[0]), reverse=True)
    train_cutoff, val_cutoff = 0.8 * n, 0.9 * n
    train, val, test = [], [], []
    for group in groups:
        if len(train) + len(group) <= train_cutoff:
            train.extend(group)
        elif len(train) + len(val) + len(group) <= val_cutoff:
            val.extend(group)
        else:
            test.extend(group)
    return _pack(train, val, test)


def load_split(path: Path) -> dict[str, list[int]]:
    with path.open() as handle:
        split = json.load(handle)
    if set(split) != {"train", "val", "test"}:
        raise ValueError("Split must contain train, val, and test")
    return split
