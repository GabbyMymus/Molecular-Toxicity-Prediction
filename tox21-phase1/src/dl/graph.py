"""SMILES -> PyTorch Geometric graphs (atoms = nodes, bonds = edges).

Row order contract: ``build_graphs(smiles, Y)[i]`` is row ``i`` of ``tox21_clean.csv``.
Missing labels stay NaN in ``data.y`` and are masked in the loss and the metrics.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

ATOM_TYPES = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]          # + "other"
DEGREES = [0, 1, 2, 3, 4, 5]                                           # + "other"
NUM_HS = [0, 1, 2, 3, 4]                                               # + "other"
HYBRIDIZATIONS = [
    Chem.HybridizationType.SP,
    Chem.HybridizationType.SP2,
    Chem.HybridizationType.SP3,
    Chem.HybridizationType.SP3D,
    Chem.HybridizationType.SP3D2,
]                                                                      # + "other"
BOND_TYPES = [
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
]


def _onehot(value, choices: Sequence, with_other: bool = True) -> list[float]:
    out = [float(value == c) for c in choices]
    if with_other:
        out.append(float(value not in choices))
    return out


def atom_features(atom: Chem.Atom) -> list[float]:
    f: list[float] = []
    f += _onehot(atom.GetSymbol(), ATOM_TYPES)                 # 10
    f += _onehot(atom.GetDegree(), DEGREES)                    # 7
    f += _onehot(atom.GetTotalNumHs(), NUM_HS)                 # 6
    f += _onehot(atom.GetHybridization(), HYBRIDIZATIONS)      # 6
    f += [float(np.clip(atom.GetFormalCharge(), -2, 2))]       # 1
    f += [float(atom.GetIsAromatic()), float(atom.IsInRing())]  # 2
    f += [atom.GetMass() / 100.0]                              # 1
    return f


def bond_features(bond: Chem.Bond) -> list[float]:
    f = _onehot(bond.GetBondType(), BOND_TYPES, with_other=False)   # 4
    f += [float(bond.GetIsConjugated()), float(bond.IsInRing())]    # 2
    return f


_dummy = Chem.MolFromSmiles("CC")
ATOM_DIM = len(atom_features(_dummy.GetAtomWithIdx(0)))             # 33
BOND_DIM = len(bond_features(_dummy.GetBondWithIdx(0)))             # 6


def mol_to_graph(mol: Chem.Mol, y: Optional[np.ndarray] = None) -> Data:
    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)
    src: list[int] = []
    dst: list[int] = []
    ea: list[list[float]] = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        src += [i, j]
        dst += [j, i]
        ea += [f, f]
    if ea:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr = torch.tensor(ea, dtype=torch.float)
    else:  # single-atom molecule
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, BOND_DIM), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    if y is not None:
        data.y = torch.tensor(np.asarray(y, dtype=np.float32)).view(1, -1)  # NaN kept
    return data


def smiles_to_graph(smiles: str, y: Optional[np.ndarray] = None) -> Data:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    return mol_to_graph(mol, y)


def build_graphs(smiles: Sequence[str], Y: np.ndarray) -> list[Data]:
    """One graph per row, same order as ``smiles``/``Y``. ``g.idx`` stores the row index."""
    graphs = []
    for i, (s, y) in enumerate(zip(smiles, Y)):
        g = smiles_to_graph(s, y)
        g.idx = torch.tensor([i])
        graphs.append(g)
    return graphs


def save_graphs(graphs: list[Data], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, path)


def load_graphs(path: Path) -> list[Data]:
    # Our own cache file, so full unpickling is fine (torch>=2.6 defaults to weights_only=True).
    return torch.load(path, weights_only=False)


def subset(graphs: list[Data], idx: Sequence[int]) -> list[Data]:
    """Graphs for a split, in the order of the split's index list."""
    return [graphs[int(i)] for i in idx]


def attach_extra(graphs: list[Data], X: np.ndarray) -> list[Data]:
    """Copy graphs and add a per-molecule feature vector ``g.extra`` of shape (1, d).

    ``X`` rows must follow the same order as ``graphs`` (use the split order everywhere).
    Used by the hybrid models (descriptors / fingerprints next to the graph embedding).
    """
    assert len(graphs) == len(X), "graphs and extra features must have the same length"
    out = []
    for g, row in zip(graphs, X):
        g2 = copy.copy(g)
        g2.extra = torch.tensor(np.asarray(row, dtype=np.float32)).view(1, -1)
        out.append(g2)
    return out


def labels_of(graphs: Sequence[Data]) -> np.ndarray:
    return np.vstack([g.y.numpy() for g in graphs])
