import pandas as pd
import pytest

from src.data import clean_dataset


def test_clean_dataset_rejects_invalid_smiles():
    with pytest.raises(ValueError):
        clean_dataset(pd.DataFrame({"not_smiles": ["C"]}))


def test_clean_dataset_has_unique_canonical_smiles():
    frame, labels = clean_dataset(pd.DataFrame({"smiles": ["C", "C"], "NR-AR": [1, None]}))
    assert frame["smiles"].is_unique
    assert labels.shape[1] == 12

