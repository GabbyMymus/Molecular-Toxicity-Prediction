"""Percentile bootstrap CIs for the scaffold-split test results (no refitting).

Reads only the saved test probabilities ``results/preds/*.npy``, the committed labels and the
committed split file. No model is fitted and nothing is chosen from these numbers: this is a
reporting step after ``scripts/04_final_test.py``.

Method (scaffold split, seeds 0, 1, 2):
  * Draw ``N_BOOT`` bootstrap resamples of the test molecules (seeded). One draw is shared by
    every model, feature set, seed and endpoint, so differences between models are paired.
  * For each endpoint only the labeled rows of the resample count (NaN labels are never used);
    an endpoint with a single class in a resample is NaN and skipped by the mean, exactly as in
    ``evaluate.masked_metrics``.
  * Per resample: ROC AUC / PR AUC per endpoint -> mean over endpoints -> mean over the 3 seeds.
  * 95% percentile interval = 2.5th and 97.5th percentile of that statistic.
  * Paired differences (RF - XGBoost within a feature set, morgan+desc - morgan within a model)
    use the same resamples; the interval is the percentile interval of the per-resample difference.

The bootstrap resamples molecules only. It does not capture variation from retraining, from
choosing hyperparameters, or from a different split; the 3 seeds are pooled by averaging.

Writes ``results/bootstrap_ci.csv`` and ``tables/bootstrap_ci.tex``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ENDPOINTS, FEATURE_SETS, RESULTS, ROOT, SEEDS, SPLITS  # noqa: E402
from src.data import load_clean  # noqa: E402
from src.splits import load_split  # noqa: E402

N_BOOT = 2000
BOOT_SEED = 0
SPLIT = "scaffold"
MODELS = ["logreg", "rf", "xgb"]
METRICS = ["roc_auc", "pr_auc"]
MODEL_LABELS = {"logreg": "Logistic Regression", "rf": "Random Forest", "xgb": "XGBoost"}
FEAT_LABELS = {"morgan": "Morgan", "morgan+desc": "Morgan + Desc"}
METRIC_LABELS = {"roc_auc": "ROC AUC", "pr_auc": "PR AUC"}


def weighted_metrics(y: np.ndarray, score: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """ROC AUC and average precision of one endpoint for many weighted samples at once.

    Args:
        y: labels in {0, 1}, shape ``(n,)`` (labeled rows only).
        score: predicted scores, shape ``(n,)``.
        weights: non-negative row weights, shape ``(B, n)``; a bootstrap resample is its
            multiplicity vector, so weight ``k`` equals the row appearing ``k`` times.

    Returns:
        ``(roc_auc, pr_auc)``, each of shape ``(B,)``; NaN where a sample has a single class.
        Ties are handled like scikit-learn: ROC counts a tied positive-negative pair as 0.5 and
        average precision steps once per distinct score.
    """
    y = np.asarray(y, dtype=float)
    order = np.argsort(score, kind="stable")
    s, y, w = np.asarray(score)[order], y[order], np.asarray(weights, dtype=float)[:, order]
    starts = np.concatenate([[0], np.nonzero(np.diff(s))[0] + 1])
    pos = np.add.reduceat(w * y, starts, axis=1)
    neg = np.add.reduceat(w * (1.0 - y), starts, axis=1)
    p_tot, n_tot = pos.sum(axis=1), neg.sum(axis=1)
    valid = (p_tot > 0) & (n_tot > 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        below = np.cumsum(neg, axis=1) - neg  # negatives with a strictly lower score
        roc = (pos * (below + 0.5 * neg)).sum(axis=1) / (p_tot * n_tot)
        tp = np.cumsum(pos[:, ::-1], axis=1)
        fp = np.cumsum(neg[:, ::-1], axis=1)
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        ap = (prec * pos[:, ::-1]).sum(axis=1) / p_tot
    return np.where(valid, roc, np.nan), np.where(valid, ap, np.nan)


def resample_weights(n: int, n_boot: int, seed: int) -> np.ndarray:
    """Return a ``(n_boot, n)`` matrix of bootstrap multiplicities (each row sums to ``n``)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    counts = np.zeros((n_boot, n))
    rows = np.repeat(np.arange(n_boot), n)
    np.add.at(counts, (rows, idx.ravel()), 1.0)
    return counts


def mean_over_endpoints(Y: np.ndarray, P: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    """Per-sample mean over endpoints of each metric, labeled rows only; shape ``(B,)`` each."""
    out = {m: [] for m in METRICS}
    for j in range(Y.shape[1]):
        keep = ~np.isnan(Y[:, j])
        roc, ap = weighted_metrics(Y[keep, j], P[keep, j], weights[:, keep])
        out["roc_auc"].append(roc)
        out["pr_auc"].append(ap)
    return {m: np.nanmean(np.vstack(v), axis=0) for m, v in out.items()}


def percentile_ci(values: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Central percentile interval of ``values``."""
    tail = (1.0 - level) / 2.0 * 100.0
    lo, hi = np.percentile(values, [tail, 100.0 - tail])
    return float(lo), float(hi)


def pooled_statistics(preds: dict[tuple[str, str, int], np.ndarray], Y: np.ndarray, weights: np.ndarray) -> dict:
    """Bootstrap and point statistics per ``(model, feature_set, metric)`` pooled over seeds.

    Returns ``{(model, fs, metric): (point, boot)}`` where ``boot`` has shape ``(B,)``.
    """
    ones = np.ones((1, len(Y)))
    stats = {}
    for model in MODELS:
        for fs in FEATURE_SETS:
            boot = {m: [] for m in METRICS}
            point = {m: [] for m in METRICS}
            for seed in SEEDS:
                P = preds[(model, fs, seed)]
                b = mean_over_endpoints(Y, P, weights)
                p = mean_over_endpoints(Y, P, ones)
                for m in METRICS:
                    boot[m].append(b[m])
                    point[m].append(p[m][0])
            for m in METRICS:
                stats[(model, fs, m)] = (float(np.mean(point[m])), np.mean(boot[m], axis=0))
    return stats


def build_tables(stats: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(intervals, differences)`` frames from :func:`pooled_statistics` output."""
    rows = []
    for (model, fs, metric), (point, boot) in stats.items():
        lo, hi = percentile_ci(boot)
        rows.append({"kind": "interval", "comparison": "", "model": model, "feature_set": fs,
                     "metric": metric, "estimate": point, "ci_low": lo, "ci_high": hi,
                     "includes_zero": ""})
    diffs = []

    def add(comparison: str, model: str, fs: str, metric: str, a: tuple, b: tuple) -> None:
        d = a[1] - b[1]
        lo, hi = percentile_ci(d)
        diffs.append({"kind": "difference", "comparison": comparison, "model": model, "feature_set": fs,
                      "metric": metric, "estimate": a[0] - b[0], "ci_low": lo, "ci_high": hi,
                      "includes_zero": bool(lo <= 0.0 <= hi)})

    for fs in FEATURE_SETS:
        for metric in METRICS:
            add("rf_minus_xgb", "rf-xgb", fs, metric, stats[("rf", fs, metric)], stats[("xgb", fs, metric)])
    for model in MODELS:
        for metric in METRICS:
            add("desc_minus_morgan", model, "morgan+desc-morgan", metric,
                stats[(model, "morgan+desc", metric)], stats[(model, "morgan", metric)])
    return pd.DataFrame(rows), pd.DataFrame(diffs)


def _fmt_ci(est: float, lo: float, hi: float) -> str:
    return f"{est:.3f} [{lo:.3f}, {hi:.3f}]"


def _fmt_diff(est: float, lo: float, hi: float) -> str:
    return f"{est:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def write_latex(intervals: pd.DataFrame, diffs: pd.DataFrame, path: Path) -> None:
    """Write both tables as booktabs ``tabular`` environments (inside two ``table`` floats)."""
    lines = [
        r"\begin{table}[t]",
        r"\caption{Scaffold test split: mean over 12 endpoints and 3 seeds with 95\% percentile "
        rf"bootstrap intervals ({N_BOOT} resamples of the test molecules).}}",
        r"\label{tab:bootstrap_ci}",
        r"\centering\small",
        r"\begin{tabular}{@{}llcc@{}}",
        r"\toprule",
        r"Model & Features & ROC AUC [95\% CI] & PR AUC [95\% CI] \\",
        r"\midrule",
    ]
    for model in MODELS:
        for fs in FEATURE_SETS:
            cells = []
            for metric in METRICS:
                r = intervals[(intervals.model == model) & (intervals.feature_set == fs)
                              & (intervals.metric == metric)].iloc[0]
                cells.append(_fmt_ci(r.estimate, r.ci_low, r.ci_high))
            lines.append(f"{MODEL_LABELS[model]} & {FEAT_LABELS[fs]} & {cells[0]} & {cells[1]} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", "",
              r"\begin{table}[t]",
              r"\caption{Paired bootstrap differences on the scaffold test split (same resamples for both "
              r"sides). ``Yes'' means the 95\% interval contains 0.}",
              r"\label{tab:bootstrap_diff}",
              r"\centering\small",
              r"\begin{tabular}{@{}llccc@{}}",
              r"\toprule",
              r"Difference & Context & Metric & Estimate [95\% CI] & Contains 0 \\",
              r"\midrule"]
    for _, r in diffs.iterrows():
        if r.comparison == "rf_minus_xgb":
            name, ctx = "RF $-$ XGBoost", FEAT_LABELS[r.feature_set]
        else:
            name, ctx = "Morgan+Desc $-$ Morgan", MODEL_LABELS[r.model]
        lines.append(f"{name} & {ctx} & {METRIC_LABELS[r.metric]} & {_fmt_diff(r.estimate, r.ci_low, r.ci_high)}"
                     f" & {'Yes' if r.includes_zero else 'No'} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def load_inputs() -> tuple[np.ndarray, dict[tuple[str, str, int], np.ndarray]]:
    """Load test labels (committed order, scaffold test indices) and saved test probabilities."""
    _, labels = load_clean()
    test_idx = np.asarray(load_split(SPLITS / f"{SPLIT}_seed0.json")["test"], dtype=int)
    Y = labels[test_idx]
    preds = {}
    for model in MODELS:
        for fs in FEATURE_SETS:
            for seed in SEEDS:
                P = np.load(RESULTS / "preds" / f"{model}_{fs}_{SPLIT}_seed{seed}.npy")
                if P.shape != Y.shape:
                    raise ValueError(f"{model}/{fs}/seed{seed}: preds {P.shape} vs labels {Y.shape}")
                preds[(model, fs, seed)] = P
    return Y, preds


def main(argv: list[str] | None = None) -> None:
    """Compute the intervals and write ``results/bootstrap_ci.csv`` and ``tables/bootstrap_ci.tex``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=BOOT_SEED)
    args = parser.parse_args(argv)
    Y, preds = load_inputs()
    weights = resample_weights(len(Y), args.n_boot, args.seed)
    stats = pooled_statistics(preds, Y, weights)
    intervals, diffs = build_tables(stats)
    out = pd.concat([intervals, diffs], ignore_index=True)
    out.insert(0, "n_boot", args.n_boot)
    out.insert(1, "boot_seed", args.seed)
    out.round(6).to_csv(RESULTS / "bootstrap_ci.csv", index=False)
    (ROOT / "tables").mkdir(exist_ok=True)
    write_latex(intervals, diffs, ROOT / "tables" / "bootstrap_ci.tex")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
