from src.splits import random_split, scaffold_split


def test_random_split_covers_all_rows():
    split = random_split(100)
    assert sorted(sum(split.values(), [])) == list(range(100))


def test_scaffold_split_has_three_partitions():
    split = scaffold_split(["a", "a", "b", "c", "d", "e", "f", "g", "h", "i"])
    assert set(split) == {"train", "val", "test"}

