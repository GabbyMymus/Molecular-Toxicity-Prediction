import numpy as np
from src.dl.graph import attach_extra, subset
from src.dl.run import get_graphs
from src.dl.split_io import load_split_named
from src.features import load_features


def test_extra_rows_align_with_graphs():
    graphs = get_graphs()
    for split_name in ["scaffold", "random"]:
        sp = load_split_named(split_name)
        Xtr, Xva, Xte = load_features("morgan+desc", split_name)
        assert Xtr.shape == (len(sp["train"]), 2261)
        desc_tr = Xtr[:, 2048:]
        sub = attach_extra(subset(graphs, sp["train"]), desc_tr)
        for k in [0, 10, len(sub) // 2, len(sub) - 1]:
            np.testing.assert_allclose(sub[k].extra.numpy()[0], desc_tr[k], rtol=1e-5)
            assert int(sub[k].idx) == int(sp["train"][k])


def test_features_finite_and_roughly_standardized_on_train():
    Xtr, Xva, Xte = load_features("morgan+desc", "scaffold")
    for X in (Xtr, Xva, Xte):
        assert np.isfinite(X).all()
    d = Xtr[:, 2048:]
    assert np.median(np.abs(d.mean(0))) < 0.5   # soft check: columns centred on train
