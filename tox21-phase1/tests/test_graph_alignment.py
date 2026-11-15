import numpy as np
from rdkit import Chem

from src.data import load_clean
from src.dl.graph import labels_of, subset
from src.dl.run import get_graphs
from src.dl.split_io import load_split_named


def test_graphs_match_clean_csv():
    df, Y = load_clean()
    graphs = get_graphs()

    assert len(graphs) == len(df) == 7709
    assert all(int(g.idx) == i for i, g in enumerate(graphs))
    graph_labels = labels_of(graphs)
    np.testing.assert_array_equal(np.isnan(graph_labels), np.isnan(Y))
    np.testing.assert_allclose(np.nan_to_num(graph_labels), np.nan_to_num(Y))


def test_split_subsets_keep_labels_and_row_order():
    _, Y = load_clean()
    graphs = get_graphs()

    for split_name in ("scaffold", "random"):
        split = load_split_named(split_name)
        for part in ("train", "val", "test"):
            indices = split[part]
            subset_graphs = subset(graphs, indices)
            assert [int(g.idx) for g in subset_graphs] == [int(i) for i in indices]
            np.testing.assert_allclose(
                np.nan_to_num(labels_of(subset_graphs)),
                np.nan_to_num(Y[indices]),
            )
            np.testing.assert_array_equal(
                np.isnan(labels_of(subset_graphs)), np.isnan(Y[indices])
            )


def test_atom_counts_match_rdkit_sample():
    df, _ = load_clean()
    graphs = get_graphs()
    smiles_col = "smiles" if "smiles" in df.columns else df.columns[0]

    for i in np.random.default_rng(0).choice(len(df), 200, replace=False):
        mol = Chem.MolFromSmiles(df[smiles_col].iloc[i])
        assert mol is not None
        assert graphs[i].x.shape[0] == mol.GetNumAtoms()
