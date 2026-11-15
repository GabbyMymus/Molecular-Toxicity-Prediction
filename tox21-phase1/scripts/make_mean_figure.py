"""Regenerate report_phase2/figures/mean_pr_roc_scaffold.{pdf,png} from the committed CSVs only.

Same 12 rows and same statistics as tables/main_results.tex (via scripts/make_results_table.py):
scaffold split, mean over the 12 endpoints per seed, then mean and sample SD (ddof=1) over seeds 0, 1, 2.
Inputs: results/results_test.csv and results/dl_results_test_*.csv. No number is typed in by hand.

Run from tox21-phase1/:  python scripts/make_mean_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_results_table import GROUPS, ROOT, load, summarize  # noqa: E402

FIG_DIR = ROOT / "report_phase2" / "figures"
SPLIT = "scaffold"
BLUE, GREEN = "#3274A1", "#55A868"  # deep / hybrid models, Phase 1 baselines (as in the earlier figure)


def main() -> None:
    data = load()
    rows = []  # (label, is_deep, stats)
    for gname, members in GROUPS:
        for label, model, fs, src in members:
            rows.append((label, gname != GROUPS[0][0], summarize(data[src], model, fs, SPLIT)))
    assert len(rows) == 12

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    for ax, key, title in ((axes[0], "pr_auc", "Scaffold test PR AUC"), (axes[1], "roc_auc", "Scaffold test ROC AUC")):
        ordered = sorted(rows, key=lambda r: r[2][key][0])
        means = [r[2][key][0] for r in ordered]
        sds = [r[2][key][1] for r in ordered]
        ax.barh([r[0] for r in ordered], means, xerr=sds,
                color=[BLUE if r[1] else GREEN for r in ordered], alpha=0.9, capsize=4,
                edgecolor="black", linewidth=0.8)
        lo = min(m - s for m, s in zip(means, sds))
        hi = max(m + s for m, s in zip(means, sds))
        pad = 0.15 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)  # axis does not start at zero; differences are small
        ax.set_xlabel(f"Mean {key.split('_')[0].upper()} AUC ± seed SD (axis truncated)")
        ax.set_title(title, pad=10, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "mean_pr_roc_scaffold.pdf")
    fig.savefig(FIG_DIR / "mean_pr_roc_scaffold.png", dpi=300)
    plt.close(fig)
    for label, _, s in sorted(rows, key=lambda r: -r[2]["pr_auc"][0]):
        print(f"{label:22s} PR {s['pr_auc'][0]:.4f} ± {s['pr_auc'][1]:.4f}   ROC {s['roc_auc'][0]:.4f} ± {s['roc_auc'][1]:.4f}")
    print("wrote", (FIG_DIR / "mean_pr_roc_scaffold.pdf").relative_to(ROOT), "and .png")


if __name__ == "__main__":
    main()
