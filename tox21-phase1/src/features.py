"""Molecular fingerprint and descriptor generation."""

from pathlib import Path

import numpy as np

from .config import FEATURES_PATH, N_BITS


def build_morgan(smiles, output: Path = FEATURES_PATH / "morgan_2048.npy"):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    rows = []
    for smile in smiles:
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            raise ValueError(smile)
        bit_vector = AllChem.GetMorganFingerprintAsBitVect(molecule, 2, nBits=N_BITS)
        rows.append(np.asarray(bit_vector, dtype=np.uint8))
    values = np.asarray(rows, dtype=np.uint8)
    output.parent.mkdir(exist_ok=True)
    np.save(output, values)
    return values


def build_descriptors(smiles, output: Path = FEATURES_PATH / "descriptors_raw.npy"):
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    descriptor_names = [name for name, _ in Descriptors.descList]
    values = []
    for smile in smiles:
        molecule = Chem.MolFromSmiles(smile)
        values.append([function(molecule) for _, function in Descriptors.descList])
    values = np.asarray(values, dtype=float)
    output.parent.mkdir(exist_ok=True)
    np.save(output, values)
    (output.parent / "descriptor_names.txt").write_text("\n".join(descriptor_names) + "\n")
    return values, descriptor_names


def load_features(path: Path = FEATURES_PATH):
    return (
        np.load(path / "morgan_2048.npy"),
        np.load(path / "descriptors_raw.npy"),
        (path / "descriptor_names.txt").read_text().splitlines(),
    )

