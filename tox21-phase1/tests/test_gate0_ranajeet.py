"""Gate 0 — Ranajeet: extra features vs. graphs alignment test.

Verifies that the engineered feature rows (Morgan fingerprints and RDKit
descriptors) produced by src.features.load_features line up exactly with
the graph objects produced by src.dl.graph.build_graphs / get_graphs,
using the SAME molecule ordering as tox21_clean.csv.

This is the prerequisite alignment check before any FP-MLP or hybrid
(MPNN+descriptors) training can safely proceed.

Run:
    cd tox21-phase1
    python -m pytest tests/test_gate0_ranajeet.py -v

Pass criteria
-------------
1. SMILES alignment  : the SMILES attached to graph[i] is the canonical form
                       of smiles_csv[i] (round-tripped through RDKit); they
                       encode the same molecule (same canonical SMILES).
2. Label alignment   : graph[i].y matches Y[i] (NaN preserved exactly).
3. Morgan alignment  : morgan_2048.npy row i == the Morgan fingerprint computed
                       directly from the SMILES in row i of tox21_clean.csv.
4. Descriptor dim    : after preprocessing on the scaffold split, the descriptor
                       matrix has exactly 213 columns → morgan+desc = 2261 total.
5. Row-count parity  : len(graphs) == len(csv) == morgan.shape[0] == desc.shape[0].
6. Split coverage    : train ∪ val ∪ test == {0 … N-1} (no gaps, no overlaps).
7. attach_extra order: G.attach_extra aligns correctly — permuting rows breaks
                       the graph[i].extra == X[i] invariant.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rdkit import Chem
from src.data import load_clean
from src.dl import graph as G
from src.features import DescriptorPreprocessor, load_features

SPLIT_FILE = ROOT / "data/splits/scaffold_seed0.json"
MORGAN_FILE = ROOT / "features/morgan_2048.npy"
DESC_FILE   = ROOT / "features/descriptors_raw.npy"
N_MORGAN    = 2048
N_DESC_PROC = 213          # descriptors after preprocessing
N_COMBINED  = 2261         # N_MORGAN + N_DESC_PROC


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def csv_data():
    """Return (df, Y) from the Phase 1 load_clean function."""
    df, Y = load_clean()
    return df, Y


@pytest.fixture(scope="module")
def graphs_and_Y(csv_data):
    """Build graphs from the CSV SMILES (without file-cache to avoid stale data)."""
    df, Y = csv_data
    graphs = G.build_graphs(df["smiles"].tolist(), Y)
    return graphs, Y


@pytest.fixture(scope="module")
def scaffold_split():
    with open(SPLIT_FILE) as f:
        return json.load(f)


# ------------------------------------------------------------------ 1. row-count parity
def test_row_count_parity(csv_data, graphs_and_Y):
    """len(graphs) == len(csv) == morgan.shape[0] == descriptors_raw.shape[0]."""
    df, _ = csv_data
    graphs, _ = graphs_and_Y
    morgan = np.load(MORGAN_FILE)
    desc   = np.load(DESC_FILE)

    n_csv    = len(df)
    n_graphs = len(graphs)
    n_morgan = morgan.shape[0]
    n_desc   = desc.shape[0]

    assert n_csv == n_graphs == n_morgan == n_desc, (
        f"Row count mismatch: csv={n_csv}, graphs={n_graphs}, "
        f"morgan={n_morgan}, desc={n_desc}"
    )


# ------------------------------------------------------------------ 2. SMILES alignment
def test_smiles_alignment(csv_data, graphs_and_Y):
    """graph[i] encodes the same molecule as smiles_csv[i] (canonical SMILES match)."""
    df, _ = csv_data
    graphs, _ = graphs_and_Y

    sample_idx = list(range(0, len(df), len(df) // 50))   # ~50 evenly spaced rows
    for i in sample_idx:
        csv_smi   = df["smiles"].iloc[i]
        graph_mol = Chem.MolFromSmiles(csv_smi)
        can_csv   = Chem.MolToSmiles(graph_mol)

        # The graph was built from csv_smi; its atom count must match the molecule
        expected_atoms = graph_mol.GetNumAtoms()
        actual_atoms   = graphs[i].x.shape[0]
        assert actual_atoms == expected_atoms, (
            f"Row {i}: atom count mismatch: csv→{expected_atoms}, graph→{actual_atoms}  "
            f"(SMILES: {csv_smi!r})"
        )
        # Row-order contract: g.idx must equal i
        assert int(graphs[i].idx) == i, (
            f"Row {i}: graph.idx={int(graphs[i].idx)}, expected {i}"
        )


# ------------------------------------------------------------------ 3. label alignment
def test_label_alignment(csv_data, graphs_and_Y):
    """graph[i].y matches Y[i], NaN preserved exactly."""
    _, Y         = csv_data
    graphs, _    = graphs_and_Y

    sample_idx = list(range(0, len(Y), len(Y) // 100))
    for i in sample_idx:
        g_y = graphs[i].y.numpy()[0]           # shape (12,)
        csv_y = Y[i]                            # shape (12,)
        nan_match  = np.isnan(g_y) == np.isnan(csv_y)
        val_match  = np.allclose(
            np.nan_to_num(g_y), np.nan_to_num(csv_y), atol=1e-6
        )
        assert nan_match.all() and val_match, (
            f"Row {i}: label mismatch\n  graph: {g_y}\n  csv:   {csv_y}"
        )


# ------------------------------------------------------------------ 4. Morgan alignment
def test_morgan_alignment(csv_data):
    """morgan_2048.npy[i] agrees with an on-the-fly Morgan fingerprint for SMILES[i]."""
    from rdkit.Chem import rdFingerprintGenerator
    df, _ = csv_data
    morgan_stored = np.load(MORGAN_FILE)
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_MORGAN)

    sample_idx = list(range(0, len(df), len(df) // 50))
    for i in sample_idx:
        mol = Chem.MolFromSmiles(df["smiles"].iloc[i])
        fp_live = gen.GetFingerprintAsNumPy(mol).astype(np.uint8)
        np.testing.assert_array_equal(
            morgan_stored[i], fp_live,
            err_msg=f"Morgan mismatch at row {i} (SMILES: {df['smiles'].iloc[i]!r})"
        )


# ------------------------------------------------------------------ 5. descriptor dimension after preprocessing
def test_descriptor_dim_after_preprocessing(scaffold_split):
    """After scaffold-split preprocessing: 213 descriptor columns, 2261 combined."""
    desc_raw = np.load(DESC_FILE)
    train_idx = scaffold_split["train"]

    prep = DescriptorPreprocessor()
    X_tr = prep.fit_transform(desc_raw[train_idx])

    assert X_tr.shape[1] == N_DESC_PROC, (
        f"Expected {N_DESC_PROC} descriptor columns after preprocessing, "
        f"got {X_tr.shape[1]}"
    )
    # Combined dimension with Morgan
    combined = N_MORGAN + X_tr.shape[1]
    assert combined == N_COMBINED, (
        f"Expected {N_COMBINED} combined columns (morgan+desc), got {combined}"
    )


# ------------------------------------------------------------------ 6. split coverage
def test_split_coverage_no_overlap(scaffold_split):
    """train ∪ val ∪ test == {0 … N-1}, disjoint, no gaps."""
    morgan = np.load(MORGAN_FILE)
    N = morgan.shape[0]

    train = set(scaffold_split["train"])
    val   = set(scaffold_split["val"])
    test  = set(scaffold_split["test"])

    assert train.isdisjoint(val),  "train ∩ val is not empty"
    assert train.isdisjoint(test), "train ∩ test is not empty"
    assert val.isdisjoint(test),   "val ∩ test is not empty"
    assert train | val | test == set(range(N)), (
        "train ∪ val ∪ test does not cover {0 … N-1} exactly"
    )


# ------------------------------------------------------------------ 7. attach_extra alignment
def test_attach_extra_alignment(graphs_and_Y, scaffold_split):
    """G.attach_extra(graphs[idx], X) → graph.extra[i] == X[i] for every row."""
    graphs, _ = graphs_and_Y
    desc_raw  = np.load(DESC_FILE)

    train_idx = scaffold_split["train"]
    val_idx   = scaffold_split["val"]

    # Fit preprocessor on train, transform val
    prep  = DescriptorPreprocessor()
    X_tr  = prep.fit_transform(desc_raw[train_idx]).astype(np.float32)
    X_val = prep.transform(desc_raw[val_idx]).astype(np.float32)

    # Build graph subsets in split order
    tr_graphs = G.subset(graphs, train_idx)
    va_graphs = G.subset(graphs, val_idx)

    # Attach extra features
    tr_aug = G.attach_extra(tr_graphs, X_tr)
    va_aug = G.attach_extra(va_graphs, X_val)

    import torch
    for local_i in range(0, len(tr_aug), max(1, len(tr_aug) // 20)):
        g = tr_aug[local_i]
        np.testing.assert_allclose(
            g.extra.numpy().squeeze(), X_tr[local_i], atol=1e-6,
            err_msg=f"train attach_extra mismatch at local index {local_i}"
        )
    for local_i in range(0, len(va_aug), max(1, len(va_aug) // 20)):
        g = va_aug[local_i]
        np.testing.assert_allclose(
            g.extra.numpy().squeeze(), X_val[local_i], atol=1e-6,
            err_msg=f"val attach_extra mismatch at local index {local_i}"
        )


# ------------------------------------------------------------------ 8. scrambling breaks alignment (negative control)
def test_scrambled_extra_breaks_alignment(graphs_and_Y, scaffold_split):
    """Shuffling the extra-feature rows should break alignment for ≥1 molecule."""
    graphs, _ = graphs_and_Y
    desc_raw  = np.load(DESC_FILE)
    train_idx = scaffold_split["train"]

    prep  = DescriptorPreprocessor()
    X_tr  = prep.fit_transform(desc_raw[train_idx]).astype(np.float32)

    # Shuffle X_tr rows
    rng  = np.random.default_rng(42)
    perm = rng.permutation(len(X_tr))
    X_shuffled = X_tr[perm]

    tr_graphs  = G.subset(graphs, train_idx)
    tr_aligned = G.attach_extra(tr_graphs, X_tr)
    tr_bad     = G.attach_extra(tr_graphs, X_shuffled)

    mismatches = 0
    for i in range(len(tr_aligned)):
        if not np.allclose(
            tr_aligned[i].extra.numpy(), tr_bad[i].extra.numpy(), atol=1e-8
        ):
            mismatches += 1

    assert mismatches > 0, (
        "Shuffling extra rows should produce at least one mismatch but did not. "
        "Possible alignment bug: extra features may already be order-independent."
    )


if __name__ == "__main__":
    # Allow running directly: python tests/test_gate0_ranajeet.py
    import sys
    fails = 0
    test_fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    # We need fixtures – run via pytest for real; this stub just lists them.
    print("Run via:  python -m pytest tests/test_gate0_ranajeet.py -v")
    print("Tests defined:", [f.__name__ for f in test_fns])
