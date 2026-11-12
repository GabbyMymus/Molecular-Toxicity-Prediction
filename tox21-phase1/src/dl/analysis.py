"""Phase 2 analysis helpers (tables, significance tests, bootstrap CIs) for notebooks/04_dl_analysis.ipynb."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.evaluate import masked_metrics


def _read_all(results_dir, pattern: str) -> pd.DataFrame:
    """Concatenate every CSV matching the pattern (each teammate may commit their own file)."""
    files = sorted(results_dir.glob(pattern))
    assert files, f"no files match {pattern} in {results_dir}"
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def load_all(results_dir) -> pd.DataFrame:
    """Phase 1 + Phase 2 test results in one frame. Phase 2 files: dl_results_test*.csv (any suffix)."""
    p1 = pd.read_csv(results_dir / "results_test.csv")
    if "tag" not in p1.columns:
        p1["tag"] = "final"
    p2 = _read_all(results_dir, "dl_results_test*.csv").drop_duplicates(
        subset=["model", "feature_set", "split", "seed", "endpoint", "tag"])
    return pd.concat([p1, p2], ignore_index=True)


def load_ablations(results_dir) -> pd.DataFrame:
    return _read_all(results_dir, "dl_ablation_test*.csv").drop_duplicates(
        subset=["model", "feature_set", "split", "seed", "endpoint", "tag"])


def summary_table(df: pd.DataFrame, split: str = "scaffold") -> pd.DataFrame:
    """Mean and std over seeds of the per-seed endpoint-average, per model x feature_set."""
    d = df[df.split == split]
    per_seed = d.groupby(["model", "feature_set", "seed"])[["roc_auc", "pr_auc"]].mean()
    out = per_seed.groupby(level=[0, 1]).agg(["mean", "std"]).round(4)
    out.columns = [f"{m}_{s}" for m, s in out.columns]
    return out.sort_values("pr_auc_mean", ascending=False)


def endpoint_table(df: pd.DataFrame, split: str, metric: str) -> pd.DataFrame:
    d = df[df.split == split]
    return d.pivot_table(index="endpoint", columns=["model", "feature_set"], values=metric, aggfunc="mean")


def paired_endpoint_test(df: pd.DataFrame, a: tuple, b: tuple, split: str = "scaffold",
                         metric: str = "pr_auc") -> dict:
    """Wilcoxon signed-rank over the 12 endpoint-wise (seed-averaged) scores of model a vs b.

    a, b are (model, feature_set). With 12 pairs the test is low-powered: report it next to the
    effect size and be honest about it.
    """
    t = endpoint_table(df, split, metric)
    x, y = t[a].to_numpy(), t[b].to_numpy()
    keep = ~(np.isnan(x) | np.isnan(y))
    diff = x[keep] - y[keep]
    stat = wilcoxon(diff) if np.any(diff != 0) else None
    return dict(mean_diff=float(diff.mean()), wins=int((diff > 0).sum()), losses=int((diff < 0).sum()),
                n=int(keep.sum()), p_value=float(stat.pvalue) if stat is not None else float("nan"))


def paired_bootstrap_ci(y: np.ndarray, pa: np.ndarray, pb: np.ndarray, metric: str = "pr_auc",
                        n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """95% CI of metric(a) - metric(b) for ONE endpoint, resampling test rows jointly (paired)."""
    rng = np.random.default_rng(seed)
    col = 1 if metric == "pr_auc" else 0
    labeled = np.flatnonzero(~np.isnan(y))
    diffs = []
    for _ in range(n_boot):
        s = rng.choice(labeled, size=len(labeled), replace=True)
        ma, mb = masked_metrics(y[s], pa[s])[col], masked_metrics(y[s], pb[s])[col]
        if not (np.isnan(ma) or np.isnan(mb)):
            diffs.append(ma - mb)
    d = np.array(diffs)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def generalization_gap(df: pd.DataFrame) -> pd.DataFrame:
    """random - scaffold, averaged over endpoints and seeds, per model x feature_set."""
    g = df.groupby(["model", "feature_set", "split"])[["roc_auc", "pr_auc"]].mean().unstack("split")
    return pd.DataFrame({"roc_gap": g[("roc_auc", "random")] - g[("roc_auc", "scaffold")],
                         "pr_gap": g[("pr_auc", "random")] - g[("pr_auc", "scaffold")]}).round(4)


def ablation_table(abl: pd.DataFrame, base: pd.DataFrame, base_model: str, base_fs: str) -> pd.DataFrame:
    """Ablation tag -> mean PR/ROC AUC (scaffold), next to the unablated reference run."""
    ref = base[(base.model == base_model) & (base.feature_set == base_fs) & (base.split == "scaffold")]
    rows = [dict(tag="reference", **ref.groupby("seed")[["roc_auc", "pr_auc"]].mean().mean().to_dict())]
    for tag, d in abl.groupby("tag"):
        rows.append(dict(tag=tag, **d.groupby("seed")[["roc_auc", "pr_auc"]].mean().mean().to_dict()))
    return pd.DataFrame(rows).round(4)
