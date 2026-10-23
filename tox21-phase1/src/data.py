"""Dataset cleaning and loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLEAN_PATH, ENDPOINTS


def _scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)


def clean_dataset(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Canonicalize molecules, remove duplicates, and return labels."""
    if "smiles" not in frame.columns:
        raise ValueError("Input must contain a smiles column")
    from rdkit import Chem

    result = frame.copy()
    result["smiles"] = result["smiles"].astype("string").str.strip()
    result = result[result["smiles"].notna() & result["smiles"].ne("")]
    result["smiles"] = result["smiles"].map(
        lambda value: Chem.MolToSmiles(Chem.MolFromSmiles(value))
        if Chem.MolFromSmiles(value) is not None else None
    )
    result = result.dropna(subset=["smiles"]).drop_duplicates("smiles").reset_index(drop=True)
    for endpoint in ENDPOINTS:
        if endpoint not in result:
            result[endpoint] = np.nan
    result["scaffold"] = result["smiles"].map(_scaffold)
    result["mol_id"] = np.arange(len(result))
    result = result[["mol_id", "smiles", "scaffold", *ENDPOINTS]]
    return result, result.loc[:, ENDPOINTS].to_numpy(dtype=float)


def load_clean(path: Path = CLEAN_PATH) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the clean CSV and return `(dataframe, labels)`."""
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    return frame, frame.loc[:, ENDPOINTS].to_numpy(dtype=float)

