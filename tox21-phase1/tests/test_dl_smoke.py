"""Phase 2 smoke tests. Run: pytest tests/test_dl_smoke.py -q   (or: python tests/test_dl_smoke.py)

These check the things that silently break DL pipelines: label masking, class weights,
graph/label alignment, permutation invariance, and that a model can overfit a tiny batch.
"""
from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))   # repo root, so `src` imports

import numpy as np
import torch
from rdkit import Chem
from torch_geometric.loader import DataLoader

from src.dl import graph as G
from src.dl.models import FPMLP, GCN, MPNN
from src.dl.train import (compute_pos_weight, fit, masked_bce_with_logits, predict,
                          set_seed)

SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CCO",
    "c1ccccc1", "Oc1ccccc1", "[O-][N+](=O)c1ccccc1", "Nc1ccccc1", "Clc1ccccc1", "CC(=O)C",
    "c1ccncc1", "c1ccc2ccccc2c1", "CC(=O)Nc1ccc(O)cc1", "NCC(=O)O", "C", "O", "CS(C)=O",
    "NC(N)=O", "C1CCCCC1", "Cc1ccccc1", "CC(=O)O", "c1ccsc1", "Fc1ccccc1", "Brc1ccccc1",
    "ClC(Cl)(Cl)C(c1ccc(Cl)cc1)c1ccc(Cl)cc1", "CCOCC", "C=Cc1ccccc1", "CC#N",
    "CCOP(=O)(OCC)OCC", "OC(=O)c1ccccc1", "CN1CCCC1c1cccnc1", "c1c[nH]cn1", "c1ccc2[nH]ccc2c1",
    "c1ccc2ncccc2c1", "c1ccc(cc1)-c1ccccc1", "OCC1OC(O)C(O)C(O)C1O", "OCCO", "CCCCCC",
    "C[N+](C)(C)C", "[Na+]",
]
N, T = len(SMILES), 12


def toy_labels(seed: int = 0, p_missing: float = 0.3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Y = (rng.random((N, T)) < 0.3).astype(np.float32)
    Y[rng.random((N, T)) < p_missing] = np.nan
    return Y


def toy_graphs(extra_dim: int = 0):
    Y = toy_labels()
    graphs = G.build_graphs(SMILES, Y)
    if extra_dim:
        X = np.random.default_rng(1).normal(size=(N, extra_dim)).astype(np.float32)
        graphs = G.attach_extra(graphs, X)
    return graphs, Y


def batch_of(graphs):
    return next(iter(DataLoader(graphs, batch_size=len(graphs), shuffle=False)))


# ------------------------------------------------------------------ featurization
def test_featurization_shapes_and_order():
    graphs, Y = toy_graphs()
    assert G.ATOM_DIM == 33 and G.BOND_DIM == 6
    for i, (s, g) in enumerate(zip(SMILES, graphs)):
        mol = Chem.MolFromSmiles(s)
        assert g.x.shape == (mol.GetNumAtoms(), G.ATOM_DIM)
        assert g.edge_index.shape[1] == 2 * mol.GetNumBonds()
        assert g.edge_attr.shape == (2 * mol.GetNumBonds(), G.BOND_DIM)
        assert int(g.idx) == i                                   # row-order contract
        np.testing.assert_array_equal(np.isnan(g.y.numpy()[0]), np.isnan(Y[i]))  # NaN kept
    assert graphs[SMILES.index("C")].edge_index.shape == (2, 0)   # single atom, no bonds


def test_subset_and_extra_follow_split_order():
    graphs, Y = toy_graphs()
    idx = [5, 2, 9]
    sub = G.subset(graphs, idx)
    assert [int(g.idx) for g in sub] == idx
    X = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    sub = G.attach_extra(sub, X)
    b = batch_of(sub)
    assert b.extra.shape == (3, 4)
    np.testing.assert_array_equal(b.extra.numpy(), X)
    assert not hasattr(graphs[5], "extra")                        # originals untouched
    np.testing.assert_array_equal(G.labels_of(sub), Y[idx])


# ------------------------------------------------------------------ loss + class weights
def test_masked_loss_matches_manual_and_ignores_nan():
    rng = np.random.default_rng(0)
    logits = torch.tensor(rng.normal(size=(8, T)), dtype=torch.float)
    y = torch.tensor(toy_labels()[:8])
    loss = float(masked_bce_with_logits(logits, y))
    p = 1 / (1 + np.exp(-logits.numpy()))
    yy, m = y.numpy(), ~np.isnan(y.numpy())
    manual = -(yy[m] * np.log(p[m]) + (1 - yy[m]) * np.log(1 - p[m])).mean()
    assert abs(loss - manual) < 1e-5
    logits2 = logits.clone()
    logits2[torch.isnan(y)] += 100.0                              # change ONLY missing entries
    assert abs(float(masked_bce_with_logits(logits2, y)) - loss) < 1e-5
    all_nan = torch.full_like(y, float("nan"))
    assert float(masked_bce_with_logits(logits, all_nan)) == 0.0   # no NaN / no crash


def test_unmasked_variant_treats_nan_as_negative():
    """mask_missing=False is the deliberately wrong ablation: NaN is scored as label 0."""
    y2 = torch.full((4, T), float("nan"))
    y2[0, 0] = 1.0                                       # exactly one labeled (positive) entry
    zero = torch.zeros(4, T)
    a = float(masked_bce_with_logits(zero, y2, mask_missing=True))
    b = float(masked_bce_with_logits(zero, y2, mask_missing=False))
    assert abs(a - np.log(2)) < 1e-6 and abs(b - np.log(2)) < 1e-6   # BCE at logit 0 is log 2
    big = torch.full((4, T), 5.0)                        # confidently predicts "toxic" everywhere
    masked = float(masked_bce_with_logits(big, y2, mask_missing=True))      # only the true positive
    unmasked = float(masked_bce_with_logits(big, y2, mask_missing=False))   # 47 fake negatives
    assert masked < 0.01 and unmasked > 4.0


def test_pos_weight_is_per_endpoint_n_neg_over_n_pos():
    Y = np.full((10, 2), np.nan, dtype=np.float32)
    Y[:, 0] = [1, 1, 0, 0, 0, 0, 0, 0, np.nan, np.nan]            # 2 pos, 6 neg -> 3.0
    Y[:4, 1] = [1, 0, 0, 0]                                         # 1 pos, 3 neg -> 3.0
    Y[4, 1] = 0
    pw = compute_pos_weight(Y, cap=None)
    assert abs(float(pw[0]) - 3.0) < 1e-6 and abs(float(pw[1]) - 4.0) < 1e-6
    assert float(compute_pos_weight(Y, cap=3.5)[1]) == 3.5


# ------------------------------------------------------------------ models
def test_forward_shapes_all_models():
    graphs, _ = toy_graphs(extra_dim=16)
    b = batch_of(graphs)
    models = [
        GCN(G.ATOM_DIM, hidden=32, layers=3),
        MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=16, steps=2),
        MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=16, steps=2, readout="sum"),
        MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=16, steps=2, readout="mean"),
        MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=16, steps=2, extra_dim=16),   # hybrid
        MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=16, steps=2, use_edges=False),
        MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=16, steps=2, edge_drop=0.2),
        FPMLP(16, hidden=32),
    ]
    for m in models:
        m.train()
        out = m(b)
        assert out.shape == (N, T) and torch.isfinite(out).all(), type(m).__name__


def test_permutation_invariance_of_atom_order():
    """Same molecule, different atom order => same prediction (graph models must not depend on it)."""
    set_seed(0)
    rng_smiles = []
    for s in SMILES:
        mol = Chem.MolFromSmiles(s)
        rng_smiles.append(Chem.MolToSmiles(mol, doRandom=True))
    Y = toy_labels()
    g1, g2 = G.build_graphs(SMILES, Y), G.build_graphs(rng_smiles, Y)
    X = np.random.default_rng(1).normal(size=(N, 8)).astype(np.float32)
    g1, g2 = G.attach_extra(g1, X), G.attach_extra(g2, X)
    for m in [GCN(G.ATOM_DIM, hidden=32), MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=16, steps=3),
              MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=16, steps=3, extra_dim=8)]:
        m.eval()
        with torch.no_grad():
            a, b = m(batch_of(g1)), m(batch_of(g2))
        assert torch.allclose(a, b, atol=1e-4), (type(m).__name__, float((a - b).abs().max()))


def test_extra_features_change_hybrid_output_only():
    graphs, _ = toy_graphs(extra_dim=8)
    b = batch_of(graphs)
    plain = MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=16, steps=2).eval()
    hyb = MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=16, steps=2, extra_dim=8).eval()
    with torch.no_grad():
        o1 = plain(b)
        b2 = b.clone(); b2.extra = b2.extra + 1.0
        o2 = plain(b2)                      # plain model must ignore .extra
        h1, h2 = hyb(b), hyb(b2)
    assert torch.allclose(o1, o2)
    assert not torch.allclose(h1, h2)


# ------------------------------------------------------------------ learning behaviour
def test_can_overfit_a_tiny_batch():
    set_seed(0)
    graphs, _ = toy_graphs()
    graphs = graphs[:24]
    b = batch_of(graphs)
    model = MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=32, steps=2, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    model.train()
    first = last = None
    for step in range(250):
        opt.zero_grad()
        loss = masked_bce_with_logits(model(b), b.y)
        loss.backward(); opt.step()
        first = float(loss.detach()) if first is None else first
        last = float(loss.detach())
    assert last < 0.25 * first, (first, last)


def test_fit_runs_restores_best_and_early_stops():
    graphs, _ = toy_graphs()
    train, val = graphs[:28], graphs[28:]
    model = MPNN(G.ATOM_DIM, G.BOND_DIM, hidden=16, steps=2)
    info = fit(model, train, val, seed=0, max_epochs=6, patience=2, batch_size=8, lr=1e-3)
    assert 1 <= len(info["history"]) <= 6
    assert 0 <= info["best_epoch"] < len(info["history"])
    P = predict(model, val, "cpu")
    assert P.shape == (len(val), T) and ((P >= 0) & (P <= 1)).all()


def test_same_seed_same_result():
    graphs, _ = toy_graphs()
    train, val = graphs[:28], graphs[28:]
    outs = []
    for _ in range(2):
        set_seed(3)
        m = GCN(G.ATOM_DIM, hidden=16, layers=2)
        fit(m, train, val, seed=3, max_epochs=3, patience=5, batch_size=8)
        outs.append(predict(m, val, "cpu"))
    np.testing.assert_allclose(outs[0], outs[1], atol=1e-6)


if __name__ == "__main__":
    import sys
    fails = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as e:  # noqa: BLE001
                fails += 1; print("FAIL", name, "->", repr(e))
    sys.exit(1 if fails else 0)
