"""Late-fusion hybrid: rank-average a Phase 1 classical model with a DL model (no training, no tuning).

Why rank-average: both models use class weights, so their probabilities live on different scales,
while ROC AUC / PR AUC only depend on the ranking within each endpoint. Equal weights are fixed
a priori (no weight search on val or test), so the ensemble cannot leak information.

  python -m src.dl.ensemble --a xgb:morgan+desc --b mpnn+desc:graph+desc --i-know-this-touches-test
Prediction files follow the naming conventions of the two harnesses:
  Phase 1 : results/preds/{model}_{feature_set}_{split}_seed{k}.npy
  Phase 2 : results/preds/{model}_{feature_set}_{split}_seed{k}_final.npy
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import RESULTS, SEEDS
from src.data import load_clean
from src.evaluate import score
from src.dl.split_io import load_split_named as load_split


def rank_norm(P: np.ndarray) -> np.ndarray:
    """Per-endpoint ranks scaled to (0, 1]."""
    return np.column_stack([rankdata(P[:, j]) / len(P) for j in range(P.shape[1])])


def fuse(Pa: np.ndarray, Pb: np.ndarray) -> np.ndarray:
    assert Pa.shape == Pb.shape, (Pa.shape, Pb.shape)
    return 0.5 * (rank_norm(Pa) + rank_norm(Pb))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Phase 1 model as model:feature_set, e.g. xgb:morgan+desc")
    ap.add_argument("--b", required=True, help="Phase 2 model as model:feature_set, e.g. mpnn+desc:graph+desc")
    ap.add_argument("--splits", default="scaffold,random")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--i-know-this-touches-test", action="store_true")
    args = ap.parse_args()
    if not args.i_know_this_touches_test:
        sys.exit("pass --i-know-this-touches-test (this scores the test set)")
    # model names may themselves contain ':'-free '+' (e.g. mpnn+desc); split on the LAST ':'
    a_model, a_fs = args.a.rsplit(":", 1)
    b_model, b_fs = args.b.rsplit(":", 1)
    _, Y = load_clean()
    name = f"ens_{a_model}_{b_model}"
    rows = []
    for split in args.splits.split(","):
        Yte = Y[load_split(split)["test"]]
        for seed in map(int, args.seeds.split(",")):
            Pa = np.load(RESULTS / "preds" / f"{a_model}_{a_fs}_{split}_seed{seed}.npy")
            Pb = np.load(RESULTS / "preds" / f"{b_model}_{b_fs}_{split}_seed{seed}_final.npy")
            P = fuse(Pa, Pb)
            np.save(RESULTS / "preds" / f"{name}_{a_fs}+{b_fs}_{split}_seed{seed}_final.npy", P)
            df = pd.DataFrame(score(Yte, P)).assign(
                model=name, feature_set=f"{a_fs}+{b_fs}", split=split, seed=seed, tag="final",
                variant="", best_epoch=np.nan)
            rows.append(df)
            print(f"{name} {split} seed {seed}: test PR {df.pr_auc.mean():.4f} ROC {df.roc_auc.mean():.4f}")
    out = RESULTS / "dl_results_test.csv"
    pd.concat(rows).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
