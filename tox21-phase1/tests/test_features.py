import numpy as np

from src.features import build_morgan


def test_morgan_shape(tmp_path):
    values = build_morgan(["C", "CC"], tmp_path / "morgan.npy")
    assert values.shape == (2, 2048)
    assert values.dtype == np.uint8

