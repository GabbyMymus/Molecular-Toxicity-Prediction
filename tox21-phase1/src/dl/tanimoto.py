"""Tanimoto-stratified evaluation of frozen Phase 1 and Phase 2 predictions.

The bins are defined from the scaffold test-set maximum similarity to the scaffold
training set.  Similarity is computed from Morgan fingerprints only; labels and
predictions are used after the bins have been fixed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import ENDPOINTS, RESULTS, SEEDS
from src.data import load_clean
from src.dl.split_io import load_split_named
from src.evaluate import masked_metrics
from src.features import load_features
from src.similarity import max_tanimoto


MODEL_FILES = {
    "XGBoost (Morgan + descriptors)": (
        "xgb_morgan+desc_{split}_seed{seed}.npy", "xgb", "morgan+desc"
    ),
    "MPNN (graph)": (
        "mpnn_graph_{split}_seed{seed}_final.npy", "mpnn", "graph"
    ),
    "H1 (MPNN + descriptors)": (
        "mpnn+desc_graph+desc_{split}_seed{seed}_final.npy", "mpnn+desc", "graph+desc"
    ),
}


def similarity_bins(similarity: np.ndarray, n_bins: int = 4) -> tuple[np.ndarray, np.ndarray]:
    """Return quantile-bin IDs and unique quantile edges.

    Duplicate quantile edges are collapsed so identical similarity values stay in
    the same bin.  The all-tied case is represented as one bin.
    """
    values = np.asarray(similarity, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("similarity must be a non-empty one-dimensional array")
    if not np.isfinite(values).all():
        raise ValueError("similarity values must be finite")
    if np.any((values < -1e-7) | (values > 1.0 + 1e-7)):
        raise ValueError("Tanimoto values must be in [0, 1]")
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError("n_bins must be a positive integer")

    edges = np.unique(np.quantile(values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size == 1:
        return np.zeros(values.size, dtype=np.int32), edges
    # Boundary ties are assigned to the lower interval; the values are never split.
    ids = np.searchsorted(edges[1:-1], values, side="left").astype(np.int32)
    return ids, edges


def _bin_label(bin_id: int, edges: np.ndarray) -> str:
    if len(edges) == 1:
        return f"Q1 [{edges[0]:.3f}, {edges[0]:.3f}]"
    return f"Q{bin_id + 1} [{edges[bin_id]:.3f}, {edges[bin_id + 1]:.3f}]"


def endpoint_bin_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    similarity: np.ndarray,
    *,
    model: str,
    split: str,
    seed: int,
    n_bins: int = 4,
) -> pd.DataFrame:
    """Compute masked ROC/PR AUC for each endpoint, bin, model and seed."""
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(predictions, dtype=float)
    sims = np.asarray(similarity, dtype=float)
    if y.ndim != 2 or pred.shape != y.shape:
        raise ValueError(f"y_true and predictions must have the same 2-D shape; got {y.shape}, {pred.shape}")
    if y.shape[1] != len(ENDPOINTS):
        raise ValueError(f"expected {len(ENDPOINTS)} endpoint columns, got {y.shape[1]}")
    if sims.shape != (len(y),):
        raise ValueError(f"similarity must have shape ({len(y)},), got {sims.shape}")
    if not np.isfinite(pred).all() or np.any((pred < -1e-6) | (pred > 1.0 + 1e-6)):
        raise ValueError("predictions must be finite probabilities in [0, 1]")
    bin_ids, edges = similarity_bins(sims, n_bins=n_bins)

    rows: list[dict] = []
    for bin_id in range(len(edges) - 1 if len(edges) > 1 else 1):
        in_bin = bin_ids == bin_id
        for endpoint_id, endpoint in enumerate(ENDPOINTS):
            labels = y[in_bin, endpoint_id]
            observed = ~np.isnan(labels)
            roc, pr = masked_metrics(labels, pred[in_bin, endpoint_id])
            rows.append({
                "model": model,
                "split": split,
                "seed": int(seed),
                "bin_id": bin_id,
                "similarity_bin": _bin_label(bin_id, edges),
                "tanimoto_min": float(edges[bin_id]),
                "tanimoto_max": float(edges[bin_id + 1]) if len(edges) > 1 else float(edges[0]),
                "endpoint": endpoint,
                "n_molecules": int(in_bin.sum()),
                "n_labeled": int(observed.sum()),
                "n_positive": int(np.nansum(labels)),
                "roc_auc": roc,
                "pr_auc": pr,
            })
    return pd.DataFrame(rows)


def summarize_bin_metrics(by_endpoint: pd.DataFrame) -> pd.DataFrame:
    """Average endpoints within seed, then report the mean and SD across seeds."""
    seed_keys = ["model", "split", "bin_id", "similarity_bin", "tanimoto_min", "tanimoto_max", "seed"]
    seed_rows = []
    for key, group in by_endpoint.groupby(seed_keys, sort=True, dropna=False):
        record = dict(zip(seed_keys, key))
        for metric in ("roc_auc", "pr_auc"):
            valid = group[metric].dropna()
            record[f"{metric}_mean"] = float(valid.mean()) if len(valid) else np.nan
            record[f"n_endpoints_{metric}"] = int(len(valid))
        record["n_molecules"] = int(group["n_molecules"].iloc[0])
        record["n_labeled_mean"] = float(group["n_labeled"].mean())
        record["n_positive_mean"] = float(group["n_positive"].mean())
        seed_rows.append(record)
    per_seed = pd.DataFrame(seed_rows)
    aggregate_keys = ["model", "split", "bin_id", "similarity_bin", "tanimoto_min", "tanimoto_max"]
    output = []
    for key, group in per_seed.groupby(aggregate_keys, sort=True, dropna=False):
        record = dict(zip(aggregate_keys, key))
        for metric in ("roc_auc", "pr_auc"):
            values = group[f"{metric}_mean"].dropna()
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            record[f"{metric}_seed_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            record[f"n_seeds_{metric}"] = int(len(values))
        for column in ("n_molecules", "n_labeled_mean", "n_positive_mean"):
            record[column] = float(group[column].mean())
        output.append(record)
    return pd.DataFrame(output)


def load_scaffold_similarity(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Return scaffold-test labels and max Tanimoto to scaffold-train molecules."""
    split = load_split_named("scaffold", seed=seed)
    X_train, _, X_test = load_features("morgan", "scaffold", seed=seed)
    _, labels = load_clean()
    y_test = labels[np.asarray(split["test"], dtype=int)]
    if len(X_test) != len(y_test) or len(X_train) != len(split["train"]):
        raise ValueError("feature rows do not align with the saved scaffold split")
    return y_test, max_tanimoto(X_test, X_train)


def load_model_predictions(
    model: str,
    split: str = "scaffold",
    seed: int = 0,
    results_dir: Path = RESULTS,
) -> np.ndarray:
    """Load one frozen model prediction artifact using the repository naming contract."""
    if model not in MODEL_FILES:
        raise ValueError(f"unknown model {model!r}; choose one of {tuple(MODEL_FILES)}")
    filename, _, _ = MODEL_FILES[model]
    path = Path(results_dir) / "preds" / filename.format(split=split, seed=seed)
    if not path.exists():
        raise FileNotFoundError(path)
    return np.load(path, allow_pickle=False)


def run_tanimoto_analysis(
    *,
    split: str = "scaffold",
    seeds: Iterable[int] = SEEDS,
    n_bins: int = 4,
    results_dir: Path = RESULTS,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Analyze available frozen predictions and write raw/summary tables plus PDF."""
    if split != "scaffold":
        raise ValueError("Tanimoto bins are defined against the scaffold train/test split")
    results_dir = Path(results_dir)
    seeds = tuple(int(seed) for seed in seeds)
    output_dir = Path(output_dir) if output_dir else results_dir.parent / "report_phase2"
    y_true, similarity = load_scaffold_similarity(seed=0)
    raw_frames: list[pd.DataFrame] = []
    missing: list[str] = []
    for model in MODEL_FILES:
        for seed in seeds:
            path = results_dir / "preds" / MODEL_FILES[model][0].format(split=split, seed=seed)
            if not path.exists():
                missing.append(str(path.relative_to(results_dir.parent)))
                continue
            pred = np.load(path, allow_pickle=False)
            raw_frames.append(endpoint_bin_metrics(
                y_true, pred, similarity, model=model, split=split, seed=int(seed), n_bins=n_bins
            ))
    if not raw_frames:
        raise FileNotFoundError("no compatible prediction files were found")
    raw = pd.concat(raw_frames, ignore_index=True)
    summary = summarize_bin_metrics(raw)
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    raw.to_csv(tables / "tanimoto_by_endpoint.csv", index=False)
    summary.to_csv(tables / "tanimoto_binned_summary.csv", index=False)
    _write_latex_summary(summary, tables / "tanimoto_summary.tex")
    plot_tanimoto_summary(summary, figures / "tanimoto.pdf")
    return raw, summary, missing


def plot_tanimoto_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Save mean endpoint PR AUC by similarity bin, with across-seed SD bars."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.3, 4.8), constrained_layout=True)
    for model, group in summary.groupby("model", sort=False):
        group = group.sort_values("bin_id")
        ax.errorbar(
            group["similarity_bin"], group["pr_auc_mean"], yerr=group["pr_auc_seed_sd"].fillna(0),
            marker="o", linewidth=1.8, capsize=3, label=model,
        )
    ax.set_xlabel("Maximum Morgan Tanimoto similarity to scaffold training set")
    ax.set_ylabel("Mean endpoint PR AUC (mean ± seed SD)")
    ax.set_title("Scaffold test performance by chemical similarity")
    ax.tick_params(axis="x", labelrotation=18)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _write_latex_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a compact table ready to include from the Phase 2 LaTeX report."""
    order = {name: rank for rank, name in enumerate(MODEL_FILES)}
    labels = {
        "XGBoost (Morgan + descriptors)": "XGBoost",
        "MPNN (graph)": "MPNN",
        "H1 (MPNN + descriptors)": "H1 (MPNN + desc.)",
    }
    rows = summary.assign(_order=summary["model"].map(order)).sort_values(["_order", "bin_id"])
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Scaffold-test performance by maximum Tanimoto similarity to the training set. PR AUC is averaged across endpoints within seed; uncertainty is the sample SD across three model seeds.}",
        r"\label{tab:tanimoto-binned}",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Model & Similarity quantile & Molecules & Mean PR AUC & Seed SD \\",
        r"\midrule",
    ]
    for row in rows.itertuples(index=False):
        model = labels.get(row.model, row.model)
        bin_label = row.similarity_bin.replace("&", r"\&")
        lines.append(
            f"{model} & {bin_label} & {int(round(row.n_molecules))} & "
            f"{row.pr_auc_mean:.3f} & {row.pr_auc_seed_sd:.3f} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
