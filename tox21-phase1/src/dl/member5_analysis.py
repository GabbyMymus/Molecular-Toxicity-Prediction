"""Prediction integrity and seed-variance summaries for the Phase 2 QA owner."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import ENDPOINTS, RESULTS, ROOT, SEEDS
from src.data import load_clean
from src.dl.split_io import load_split_named
from src.evaluate import score


PREDICTION_SPECS = {
    "mpnn": ("mpnn_graph_{split}_seed{seed}_final.npy", "mpnn", "graph"),
    "xgb": ("xgb_morgan+desc_{split}_seed{seed}.npy", "xgb", "morgan+desc"),
    "h1": ("mpnn+desc_graph+desc_{split}_seed{seed}_final.npy", "mpnn+desc", "graph+desc"),
}
REFERENCE_CSVS = {
    "mpnn": (RESULTS / "dl_results_test_parrv_mpnn.csv", "mpnn", "graph"),
    "xgb": (RESULTS / "results_test.csv", "xgb", "morgan+desc"),
    "h1": (RESULTS / "dl_results_test_ranajeet_h1.csv", "mpnn+desc", "graph+desc"),
}


def seed_variance_table(per_endpoint: pd.DataFrame) -> pd.DataFrame:
    """Summarize endpoint-mean test metrics across model seeds.

    The reported seed SD is computed after first averaging over endpoints, matching
    the seed-level convention used in ``src.dl.analysis.summary_table``.
    """
    required = {"model", "feature_set", "split", "seed", "roc_auc", "pr_auc"}
    missing = required - set(per_endpoint.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")
    per_seed = per_endpoint.groupby(
        ["model", "feature_set", "split", "seed"], as_index=False
    )[["roc_auc", "pr_auc"]].mean()
    summary = per_seed.groupby(["model", "feature_set", "split"]).agg(
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_seed_sd=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_seed_sd=("pr_auc", "std"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    return summary


def write_latex_seed_variance(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a compact Phase 2 table from the machine-readable seed summary."""
    labels = {
        ("mpnn", "graph"): "MPNN (graph)",
        ("mpnn+desc", "graph+desc"): "H1 (MPNN + descriptors)",
        ("xgb", "morgan+desc"): "XGBoost (Morgan + descriptors)",
    }
    rows = summary.sort_values(["split", "model", "feature_set"])
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Test-set performance and model-seed variability. Values are endpoint means with sample SD across the three model seeds.}",
        r"\label{tab:member5-seed-variance}",
        r"\begin{tabular}{llrr}",
        r"\hline",
        r"Model & Split & ROC AUC & PR AUC \\",
        r"\hline",
    ]
    for row in rows.itertuples(index=False):
        model = labels.get((row.model, row.feature_set), f"{row.model} ({row.feature_set})")
        lines.append(
            f"{model} & {row.split} & {row.roc_auc_mean:.3f} $\\pm$ {row.roc_auc_seed_sd:.3f} & "
            f"{row.pr_auc_mean:.3f} $\\pm$ {row.pr_auc_seed_sd:.3f} \\\\")
    lines.extend([r"\hline", r"\end{tabular}", r"\end{table}"])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reference_frame(key: str, model: str, feature_set: str, split: str, seed: int) -> pd.DataFrame | None:
    csv_path, ref_model, ref_feature_set = REFERENCE_CSVS[key]
    if not csv_path.exists():
        return None
    frame = pd.read_csv(csv_path)
    filters = {
        "model": ref_model,
        "feature_set": ref_feature_set,
        "split": split,
        "seed": seed,
    }
    for column, value in filters.items():
        if column not in frame:
            return None
        frame = frame[frame[column] == value]
    if "tag" in frame:
        frame = frame[frame.tag == "final"]
    return frame


def _relative_path(path: Path) -> str:
    """Return a repository-relative artifact path for portable audit CSVs."""
    try:
        return path.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def audit_saved_predictions(
    *,
    split: str = "scaffold",
    seeds: Iterable[int] = SEEDS,
    results_dir: Path = RESULTS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Check saved probability matrices and compare metrics with committed result CSVs.

    Prediction rows follow the split's sorted test-index order by the runner contract.
    The metric comparison catches shape, endpoint order, and test-label alignment errors
    when the corresponding score CSV is available.
    """
    if split not in ("scaffold", "random"):
        raise ValueError("split must be 'scaffold' or 'random'")
    results_dir = Path(results_dir)
    _, labels = load_clean()
    split_idx = load_split_named(split, seed=0)
    test_labels = labels[np.asarray(split_idx["test"], dtype=int)]
    expected_shape = (len(test_labels), len(ENDPOINTS))
    summaries: list[dict] = []
    metric_rows: list[dict] = []

    for key, (pattern, model, feature_set) in PREDICTION_SPECS.items():
        for seed in seeds:
            path = results_dir / "preds" / pattern.format(split=split, seed=int(seed))
            if not path.exists():
                summaries.append({
                    "model": model, "feature_set": feature_set, "split": split,
                    "seed": int(seed), "path": _relative_path(path), "status": "missing",
                    "metrics_match_reference": np.nan,
                })
                continue
            prediction = np.load(path, allow_pickle=False)
            if prediction.shape != expected_shape:
                raise ValueError(f"{path}: expected shape {expected_shape}, got {prediction.shape}")
            if not np.isfinite(prediction).all():
                raise ValueError(f"{path}: predictions contain NaN or infinity")
            if np.any((prediction < -1e-6) | (prediction > 1.0 + 1e-6)):
                raise ValueError(f"{path}: prediction probabilities fall outside [0, 1]")

            rows = pd.DataFrame(score(test_labels, prediction)).assign(
                model=model, feature_set=feature_set, split=split, seed=int(seed), tag="final"
            )
            for record in rows.to_dict(orient="records"):
                metric_rows.append(record)

            reference = _reference_frame(key, model, feature_set, split, int(seed))
            matched = np.nan
            if reference is not None and len(reference):
                ref = reference.set_index("endpoint").reindex(ENDPOINTS)
                got = rows.set_index("endpoint").reindex(ENDPOINTS)
                if not ref.index.equals(got.index) or ref.isna().all(axis=None):
                    matched = False
                else:
                    counts_ok = np.array_equal(ref["n_test"].to_numpy(), got["n_test"].to_numpy())
                    positives_ok = np.array_equal(ref["n_pos_test"].to_numpy(), got["n_pos_test"].to_numpy())
                    roc_ok = np.allclose(ref["roc_auc"], got["roc_auc"], rtol=0, atol=2e-6, equal_nan=True)
                    pr_ok = np.allclose(ref["pr_auc"], got["pr_auc"], rtol=0, atol=2e-6, equal_nan=True)
                    matched = bool(counts_ok and positives_ok and roc_ok and pr_ok)
            summaries.append({
                "model": model,
                "feature_set": feature_set,
                "split": split,
                "seed": int(seed),
                "path": _relative_path(path),
                "status": "ok",
                "n_test_molecules": len(prediction),
                "n_endpoints": prediction.shape[1],
                "metrics_match_reference": matched,
            })
    return pd.DataFrame(summaries), pd.DataFrame(metric_rows)


def write_member5_audits(
    *,
    results_dir: Path = RESULTS,
    output_dir: Path | None = None,
    split: str = "scaffold",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Write alignment audit, per-endpoint scores and seed-variance summary."""
    results_dir = Path(results_dir)
    output_dir = Path(output_dir) if output_dir else results_dir
    audit, metrics = audit_saved_predictions(split=split, results_dir=results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "member5_prediction_audit.csv", index=False)
    metrics.to_csv(output_dir / "member5_prediction_metrics.csv", index=False)
    variance = seed_variance_table(metrics)
    variance.to_csv(output_dir / "member5_seed_variance.csv", index=False)
    return audit, metrics, variance
