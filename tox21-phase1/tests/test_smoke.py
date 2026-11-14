"""End-to-end smoke test on 200 molecules: clean, split, features, one fast model, evaluate.

The 200 molecules are a seeded sample of the committed cleaned CSV (read only). Every output is
written under ``tmp_path``; the real splits, features and ``results/`` are never written, and the
real test split indices are never used.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.config import CLEAN, ENDPOINTS, RESULTS, SPLITS
from src.data import clean_dataset
from src.evaluate import fit_predict, score
from src.features import DescriptorPreprocessor, build_descriptors, build_morgan
from src.models import REGISTRY
from src.splits import scaffold_split

pytestmark = pytest.mark.smoke

N_MOL = 200


def _snapshot(*dirs: Path) -> dict[str, tuple[int, int]]:
    """Map every file under ``dirs`` to (size, mtime_ns) so writes can be detected."""
    return {
        str(p): (p.stat().st_size, p.stat().st_mtime_ns)
        for d in dirs
        if d.exists()
        for p in d.rglob("*")
        if p.is_file()
    }


def test_end_to_end_200_molecules(tmp_path):
    before = _snapshot(RESULTS, SPLITS)

    # 1. 200 molecules from the committed cleaned CSV, in original order; NaN labels stay NaN.
    full = pd.read_csv(CLEAN)
    pick = np.sort(np.random.default_rng(0).choice(len(full), size=N_MOL, replace=False))
    raw = full.iloc[pick][["mol_id", "smiles", *ENDPOINTS]].reset_index(drop=True)
    n_missing_raw = int(raw[ENDPOINTS].isna().sum().sum())
    assert n_missing_raw > 0

    # 2. clean: already canonical, so the row set and label values must survive unchanged.
    clean, Y = clean_dataset(raw)
    assert len(clean) == N_MOL
    assert list(clean["mol_id"]) == list(raw["mol_id"])  # row order preserved
    assert int(np.isnan(Y).sum()) == n_missing_raw  # missing labels never filled
    assert Y.shape == (N_MOL, len(ENDPOINTS))

    # 3. scaffold split: disjoint, complete, 80/10/10 on 200 rows.
    split = scaffold_split(list(clean["scaffold"]))
    tr, va, te = (np.array(split[k]) for k in ("train", "val", "test"))
    assert len(set(tr) | set(va) | set(te)) == N_MOL
    assert len(tr) + len(va) + len(te) == N_MOL
    assert not (set(tr) & set(te)) and not (set(tr) & set(va)) and not (set(va) & set(te))
    assert len(tr) > 0 and len(va) > 0

    # 4. features written to tmp_path only; descriptors cleaned with a train-only preprocessor.
    smiles = list(clean["smiles"])
    X_morgan = build_morgan(smiles, output=tmp_path / "morgan.npy")
    assert X_morgan.shape == (N_MOL, 2048)
    raw_desc, names = build_descriptors(smiles, output=tmp_path / "descriptors_raw.npy")
    assert raw_desc.shape == (N_MOL, len(names))
    assert (tmp_path / "descriptor_names.txt").exists()
    prep = DescriptorPreprocessor().fit(raw_desc[tr])
    desc = prep.transform(raw_desc)
    assert np.isfinite(desc).all()
    X = np.hstack([X_morgan.astype(float), desc])
    assert X.shape[0] == N_MOL and X.shape[1] > 2048

    # 5. one fast model (Logistic Regression) fitted on train, evaluated on the validation rows.
    make_model, grid = REGISTRY["logreg"]
    P = fit_predict(make_model, grid[0], X[tr], Y[tr], X[va], seed=0)
    assert P.shape == (len(va), len(ENDPOINTS))
    assert not np.isnan(P).any()
    assert ((P >= 0) & (P <= 1)).all()

    # 6. evaluate: one row per endpoint; metrics in [0, 1] where defined; counts match the labels.
    rows = score(Y[va], P)
    assert [r["endpoint"] for r in rows] == ENDPOINTS
    for j, r in enumerate(rows):
        assert r["n_test"] == int((~np.isnan(Y[va][:, j])).sum())
        for key in ("roc_auc", "pr_auc"):
            assert np.isnan(r[key]) or 0.0 <= r[key] <= 1.0
    assert sum(not np.isnan(r["roc_auc"]) for r in rows) >= 1

    # Nothing outside tmp_path was written: results/ and the real splits are untouched.
    assert _snapshot(RESULTS, SPLITS) == before
