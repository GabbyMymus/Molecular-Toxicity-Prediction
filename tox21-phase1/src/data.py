"""Dataset cleaning and loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import CLEAN_PATH, ENDPOINTS


def _largest_fragment_canonical(smiles: object) -> str | None:
    """Return canonical SMILES for the largest parsed fragment, or None."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")

    if pd.isna(smiles):
        return None
    molecule = Chem.MolFromSmiles(str(smiles).strip())
    if molecule is None:
        return None
    molecule = rdMolStandardize.LargestFragmentChooser().choose(molecule)
    return Chem.MolToSmiles(molecule, canonical=True)


def _merge_labels(values: pd.Series) -> float:
    """Keep a known label if duplicates agree; conflicting labels become missing."""
    observed = pd.unique(values.dropna())
    return float(observed[0]) if len(observed) == 1 else np.nan


def _scaffold(smiles: str) -> str:
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid canonical SMILES: {smiles}")
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule, includeChirality=False)


def clean_dataset(frame: pd.DataFrame, *, return_log: bool = False):
    """Clean Tox21 rows and merge duplicate labels by canonical molecule.

    Returns ``(clean_frame, Y)`` by default. Set ``return_log=True`` to also
    receive a dictionary of auditable row and label-conflict counts.
    """
    if "smiles" not in frame.columns:
        raise ValueError("Input must contain a smiles column")
    result = frame.copy()
    for endpoint in ENDPOINTS:
        if endpoint not in result:
            result[endpoint] = np.nan
    if "mol_id" not in result:
        result["mol_id"] = np.arange(len(result)).astype(str)

    log = {"raw_rows": int(len(result))}
    result["smiles"] = result["smiles"].map(_largest_fragment_canonical)
    log["invalid_dropped"] = int(result["smiles"].isna().sum())
    result = result.dropna(subset=["smiles"])
    log["valid_after_parsing"] = int(len(result))

    # Count molecule/endpoint conflicts before resolving them.
    grouped = result.groupby("smiles", sort=False, dropna=False)
    conflicts = int((grouped[ENDPOINTS].nunique(dropna=True) > 1).to_numpy().sum())
    log["label_conflicts_set_to_nan"] = conflicts
    out = grouped.agg({**{endpoint: _merge_labels for endpoint in ENDPOINTS}, "mol_id": "first"}).reset_index()
    log["duplicate_rows_removed"] = int(len(result) - len(out))
    log["clean_rows"] = int(len(out))
    out["scaffold"] = out["smiles"].map(_scaffold)
    out = out[["mol_id", "smiles", "scaffold", *ENDPOINTS]].reset_index(drop=True)
    labels = out.loc[:, ENDPOINTS].to_numpy(dtype=float)
    return (out, labels, log) if return_log else (out, labels)


def load_clean(path: Path = CLEAN_PATH) -> tuple[pd.DataFrame, np.ndarray]:
    """Load the clean CSV and return ``(dataframe, labels)``."""
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["scaffold"] = frame["scaffold"].astype("string").fillna("")
    return frame, frame.loc[:, ENDPOINTS].to_numpy(dtype=float)
