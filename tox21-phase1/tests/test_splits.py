import numpy as np
import pytest

from src.splits import random_split, scaffold_split


def _assert_partition(split, n):
    assert set(split) == {"train", "val", "test"}
    all_indices = split["train"] + split["val"] + split["test"]
    assert sorted(all_indices) == list(range(n))
    assert len(all_indices) == len(set(all_indices))
    assert all(indices == sorted(indices) for indices in split.values())


def test_random_split_is_reproducible_and_approximately_80_10_10():
    one = random_split(1000, seed=17)
    assert one == random_split(1000, seed=17)
    _assert_partition(one, 1000)
    assert [len(one[k]) for k in ("train", "val", "test")] == [800, 100, 100]
    assert one != random_split(1000, seed=18)


def test_scaffold_split_partitions_all_rows_without_scaffold_leakage():
    scaffolds = [f"scaffold-{i // (1 + i % 4)}" for i in range(101)]
    split = scaffold_split(scaffolds)
    _assert_partition(split, len(scaffolds))
    memberships = {}
    for name, indices in split.items():
        for index in indices:
            memberships.setdefault(scaffolds[index], set()).add(name)
    assert all(len(parts) == 1 for parts in memberships.values())
    assert all(abs(len(split[name]) / len(scaffolds) - target) <= 0.12
               for name, target in zip(("train", "val", "test"), (0.8, 0.1, 0.1)))


def test_empty_scaffolds_are_kept_together():
    split = scaffold_split([""] * 20 + [f"ring-{i}" for i in range(20)])
    memberships = [name for name, indices in split.items() if any(i < 20 for i in indices)]
    assert len(memberships) == 1


def test_split_rejects_too_few_rows():
    with pytest.raises(ValueError):
        random_split(2)
    with pytest.raises(ValueError):
        scaffold_split(["a", "b"])
