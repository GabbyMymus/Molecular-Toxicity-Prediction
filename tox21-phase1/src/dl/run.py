"""CLI that plugs the DL models into the Phase 1 harness (same splits, metrics, seeds, CSV schema).

Examples (run from the repo root):
  python -m src.dl.run --mode tune  --model mpnn  --extra none
  python -m src.dl.run --mode final --model mpnn  --extra none  --i-know-this-touches-test
  python -m src.dl.run --mode final --model mpnn  --extra desc  --i-know-this-touches-test
  python -m src.dl.run --mode ablate --model mpnn --extra none --variant steps=1 --tag T1 --i-know-this-touches-test

Phase 1 modules used: src.config (ROOT, ENDPOINTS, SEEDS, RESULTS), src.data.load_clean,
src.splits.load_split, src.features.load_features, src.evaluate.score.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import ENDPOINTS, RESULTS, ROOT, SEEDS
from src.data import load_clean
from src.dl import graph as G
from src.dl.models import FPMLP, GCN, MPNN
from src.dl.train import evaluate, fit, predict
from src.evaluate import score
from src.features import load_features
from src.dl.split_io import load_split_named as load_split

N_MORGAN = 2048                      # first 2048 columns of load_features("morgan+desc") are Morgan bits
CACHE = ROOT / "data/processed/graphs.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tuning grids: at most 8 configs each (same budget rule as Phase 1).
GRIDS: dict[str, list[dict]] = {
    "mpnn": [   # cost per epoch grows ~hidden^2 (edge network): h32 ~5s, h64 ~20s, h96 ~45s on a 2-core CPU
        dict(hidden=64, steps=3, dropout=0.2, weight_decay=1e-5, lr=1e-3, edge_drop=0.0),
        dict(hidden=32, steps=3, dropout=0.2, weight_decay=1e-5, lr=1e-3, edge_drop=0.0),
        dict(hidden=64, steps=3, dropout=0.4, weight_decay=1e-5, lr=1e-3, edge_drop=0.0),
        dict(hidden=64, steps=3, dropout=0.3, weight_decay=1e-4, lr=1e-3, edge_drop=0.1),
        dict(hidden=64, steps=4, dropout=0.2, weight_decay=1e-5, lr=5e-4, edge_drop=0.0),
        dict(hidden=32, steps=4, dropout=0.3, weight_decay=1e-4, lr=1e-3, edge_drop=0.1),
        dict(hidden=64, steps=2, dropout=0.2, weight_decay=1e-5, lr=1e-3, edge_drop=0.0),
        dict(hidden=96, steps=3, dropout=0.3, weight_decay=1e-5, lr=5e-4, edge_drop=0.0),  # GPU only
    ],
    "gcn": [
        dict(hidden=128, layers=3, dropout=0.2, weight_decay=1e-5, lr=1e-3),
        dict(hidden=256, layers=3, dropout=0.2, weight_decay=1e-5, lr=1e-3),
        dict(hidden=128, layers=3, dropout=0.4, weight_decay=1e-4, lr=1e-3),
        dict(hidden=256, layers=3, dropout=0.4, weight_decay=1e-4, lr=1e-3),
        dict(hidden=128, layers=4, dropout=0.2, weight_decay=1e-5, lr=5e-4),
        dict(hidden=256, layers=4, dropout=0.3, weight_decay=1e-4, lr=5e-4),
        dict(hidden=128, layers=2, dropout=0.2, weight_decay=1e-5, lr=1e-3),
        dict(hidden=256, layers=2, dropout=0.3, weight_decay=1e-4, lr=1e-3),
    ],
    "fpmlp": [
        dict(hidden=512, dropout=0.3, weight_decay=1e-5, lr=1e-3),
        dict(hidden=1024, dropout=0.3, weight_decay=1e-5, lr=1e-3),
        dict(hidden=512, dropout=0.5, weight_decay=1e-5, lr=1e-3),
        dict(hidden=1024, dropout=0.5, weight_decay=1e-4, lr=1e-3),
        dict(hidden=512, dropout=0.3, weight_decay=1e-4, lr=5e-4),
        dict(hidden=1024, dropout=0.5, weight_decay=1e-5, lr=5e-4),
        dict(hidden=256, dropout=0.3, weight_decay=1e-5, lr=1e-3),
        dict(hidden=256, dropout=0.5, weight_decay=1e-4, lr=1e-3),
    ],
}
TRAIN_KEYS = {"lr", "weight_decay", "batch_size", "max_epochs", "patience", "grad_clip",
              "use_pos_weight", "pos_weight_cap", "mask_missing"}


def get_graphs() -> list:
    """Graphs in clean-CSV row order. The cache is rebuilt if it no longer matches the clean CSV."""
    df, Y = load_clean()
    if CACHE.exists():
        graphs = G.load_graphs(CACHE)
        if len(graphs) == len(df) and np.allclose(G.labels_of(graphs), Y, equal_nan=True):
            return graphs
        print("graph cache is stale -> rebuilding", flush=True)
    graphs = G.build_graphs(df["smiles"].tolist(), Y)
    G.save_graphs(graphs, CACHE)
    return graphs


def feature_name(model: str, extra: str) -> str:
    if model == "fpmlp":
        return extra                           # morgan / desc / both
    return "graph" if extra == "none" else f"graph+{extra}"


def prepare(graphs: list, split: str, extra: str, shuffle_extra: bool = False,
            shuffle_labels: bool = False):
    """Graph subsets for train/val/test in split order, with ``extra`` features attached.

    Validation controls (never used for the headline results):
      shuffle_extra  - permute the extra feature rows inside each split: same parameters and the same
                       marginal feature distributions, but no molecule-specific information.
      shuffle_labels - permute the TRAIN labels across molecules: test AUC must collapse to about 0.5.
    """
    idx = load_split(split)
    parts = {k: G.subset(graphs, idx[k]) for k in ("train", "val", "test")}
    extra_dim = 0
    if extra != "none":
        Xtr, Xva, Xte = load_features("morgan+desc", split)      # rows follow split order
        sl = {"morgan": slice(0, N_MORGAN), "desc": slice(N_MORGAN, None), "both": slice(0, None)}[extra]
        rng = np.random.default_rng(123)
        for k, X in zip(("train", "val", "test"), (Xtr, Xva, Xte)):
            X = X[:, sl]
            if shuffle_extra:
                X = X[rng.permutation(len(X))]
            parts[k] = G.attach_extra(parts[k], X)
        extra_dim = Xtr[:, sl].shape[1]
    if shuffle_labels:
        rng = np.random.default_rng(999)
        ys = [g.y.clone() for g in parts["train"]]
        shuffled = []
        for g, k in zip(parts["train"], rng.permutation(len(ys))):
            g2 = copy.copy(g)
            g2.y = ys[int(k)].clone()
            shuffled.append(g2)
        parts["train"] = shuffled
    return parts, extra_dim


def build_model(name: str, cfg: dict, extra_dim: int):
    kw = {k: v for k, v in cfg.items() if k not in TRAIN_KEYS}
    if name == "gcn":
        return GCN(G.ATOM_DIM, **kw)
    if name == "mpnn":
        return MPNN(G.ATOM_DIM, G.BOND_DIM, extra_dim=extra_dim, **kw)
    if name == "fpmlp":
        assert extra_dim > 0, "fpmlp needs --extra morgan|desc|both"
        return FPMLP(extra_dim, **kw)
    raise ValueError(name)


def run_one(name: str, cfg: dict, parts: dict, extra_dim: int, seed: int):
    model = build_model(name, cfg, extra_dim)
    train_cfg = {k: v for k, v in cfg.items() if k in TRAIN_KEYS}
    info = fit(model, parts["train"], parts["val"], seed=seed, device=DEVICE, **train_cfg)
    return model, info


def _append(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, mode="a", header=not path.exists(), index=False)


def _parse_variant(s: str) -> dict:
    out = {}
    for item in filter(None, s.split(",")):
        k, v = item.split("=")
        out[k] = json.loads(v) if v[0] in "0123456789-[{tfn" else v
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tune", "final", "ablate"], required=True)
    ap.add_argument("--model", choices=["gcn", "mpnn", "fpmlp"], required=True)
    ap.add_argument("--extra", choices=["none", "morgan", "desc", "both"], default="none")
    ap.add_argument("--split", default="scaffold")
    ap.add_argument("--splits", default="scaffold,random", help="final mode: comma list")
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--variant", default="", help='ablate mode, e.g. "steps=1" or "readout=sum,use_edges=false"')
    ap.add_argument("--config-key", default="", help="best_configs_dl.json key to use (default model|extra)")
    ap.add_argument("--tag", default="", help="ablation name written to the CSV")
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--grid-limit", type=int, default=None, help="tune only the first N configs (time cap)")
    ap.add_argument("--shuffle-extra", action="store_true", help="control: permute descriptor/FP rows")
    ap.add_argument("--shuffle-labels", action="store_true", help="control: permute TRAIN labels")
    ap.add_argument("--i-know-this-touches-test", action="store_true")
    args = ap.parse_args()

    graphs = get_graphs()
    best_path = RESULTS / "best_configs_dl.json"
    key = args.config_key or f"{args.model}|{args.extra}"
    fs = feature_name(args.model, args.extra)

    if args.mode == "tune":                              # VALIDATION ONLY, seed 0, scaffold split
        parts, extra_dim = prepare(graphs, "scaffold", args.extra)
        Yva = G.labels_of(parts["val"])
        rows, scores = [], []
        grid = GRIDS[args.model][: args.grid_limit]
        for k, cfg in enumerate(grid):
            cfg = dict(cfg, **({"max_epochs": args.max_epochs} if args.max_epochs else {}))
            model, info = run_one(args.model, cfg, parts, extra_dim, seed=0)
            roc, pr = evaluate(Yva, predict(model, parts["val"], DEVICE))
            scores.append(float(np.nanmean(pr)))
            for j, ep in enumerate(ENDPOINTS):
                rows.append(dict(model=args.model, feature_set=fs, split="scaffold", seed=0,
                                 config_id=k, params=json.dumps(cfg), endpoint=ep,
                                 roc_auc=roc[j], pr_auc=pr[j], best_epoch=info["best_epoch"]))
            print(f"config {k}: val PR {scores[-1]:.4f}  ({cfg})", flush=True)
        _append(pd.DataFrame(rows), RESULTS / "dl_tuning_val.csv")
        best = json.loads(best_path.read_text()) if best_path.exists() else {}
        best[key] = grid[int(np.argmax(scores))]
        best_path.write_text(json.dumps(best, indent=2))
        print("best config:", best[key])
        return

    # final / ablate touch the TEST split
    if not args.i_know_this_touches_test or not best_path.exists():
        sys.exit("Freeze and commit results/best_configs_dl.json first, then pass --i-know-this-touches-test")
    best = json.loads(best_path.read_text())
    cfg = best.get(key) or best[f"{args.model}|none"]
    out_path = RESULTS / ("dl_results_test.csv" if args.mode == "final" else "dl_ablation_test.csv")
    variant = _parse_variant(args.variant)
    cfg = {**cfg, **variant}
    if args.max_epochs:
        cfg["max_epochs"] = args.max_epochs
    splits = args.splits.split(",") if args.mode == "final" else [args.split]
    for split in splits:
        parts, extra_dim = prepare(graphs, split, args.extra, args.shuffle_extra, args.shuffle_labels)
        Yte = G.labels_of(parts["test"])
        for seed in map(int, args.seeds.split(",")):
            model, info = run_one(args.model, cfg, parts, extra_dim, seed)
            P = predict(model, parts["test"], DEVICE)
            name = args.model + ("" if args.extra == "none" else f"+{args.extra}")
            (RESULTS / "preds").mkdir(parents=True, exist_ok=True)
            tag = args.tag or "final"
            np.save(RESULTS / "preds" / f"{name}_{fs}_{split}_seed{seed}_{tag}.npy", P)
            (RESULTS / "curves").mkdir(parents=True, exist_ok=True)       # learning curves for the report
            (RESULTS / "curves" / f"{name}_{fs}_{split}_seed{seed}_{tag}.json").write_text(
                json.dumps(info["history"]))
            df = pd.DataFrame(score(Yte, P)).assign(
                model=name, feature_set=fs, split=split, seed=seed, tag=tag,
                variant=args.variant, best_epoch=info["best_epoch"])
            _append(df, out_path)
            print(f"{name} {split} seed {seed}: test PR {df.pr_auc.mean():.4f} "
                  f"ROC {df.roc_auc.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
