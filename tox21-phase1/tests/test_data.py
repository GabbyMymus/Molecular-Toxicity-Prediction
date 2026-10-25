import numpy as np
import pandas as pd
import pytest

from src.config import ENDPOINTS
from src.data import clean_dataset, load_clean


def test_clean_dataset_requires_smiles():
    with pytest.raises(ValueError, match="smiles"):
        clean_dataset(pd.DataFrame({"structure": ["C"]}))


def test_clean_dataset_canonicalizes_largest_fragment_and_deduplicates():
    frame = pd.DataFrame({
        "mol_id": ["first", "second", "invalid"],
        "smiles": ["O.CC", "CC", "not-a-smiles"],
        "NR-AR": [1, 1, 0],
    })
    clean, labels, log = clean_dataset(frame, return_log=True)
    assert clean["smiles"].tolist() == ["CC"]
    assert clean["mol_id"].tolist() == ["first"]
    assert labels.shape == (1, 12)
    assert labels[0, ENDPOINTS.index("NR-AR")] == 1
    assert log == {
        "raw_rows": 3,
        "invalid_dropped": 1,
        "valid_after_parsing": 2,
        "label_conflicts_set_to_nan": 0,
        "duplicate_rows_removed": 1,
        "clean_rows": 1,
    }


def test_duplicate_conflicts_are_missing_per_endpoint_and_agreements_survive():
    frame = pd.DataFrame({
        "smiles": ["CC", "C(C)"],
        "NR-AR": [0, 1],
        "NR-AhR": [1, np.nan],
    })
    clean, labels, log = clean_dataset(frame, return_log=True)
    assert len(clean) == 1
    assert np.isnan(labels[0, ENDPOINTS.index("NR-AR")])
    assert labels[0, ENDPOINTS.index("NR-AhR")] == 1
    assert log["label_conflicts_set_to_nan"] == 1


def test_clean_dataset_has_unique_canonical_smiles_and_fixed_label_shape():
    clean, labels = clean_dataset(pd.DataFrame({"smiles": ["C", "C"]}))
    assert clean["smiles"].is_unique
    assert labels.shape == (1, 12)
    assert np.isnan(labels).all()


def test_load_clean_restores_empty_acyclic_scaffolds(tmp_path):
    columns = ["mol_id", "smiles", "scaffold", *ENDPOINTS]
    path = tmp_path / "clean.csv"
    pd.DataFrame([["x", "CC", np.nan, *([np.nan] * 12)]], columns=columns).to_csv(path, index=False)
    clean, labels = load_clean(path)
    assert clean.loc[0, "scaffold"] == ""
    assert labels.shape == (1, 12)
