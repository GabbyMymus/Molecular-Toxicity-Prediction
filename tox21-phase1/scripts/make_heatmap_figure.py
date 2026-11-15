"""Regenerate report_phase2/figures/per_endpoint_heatmap_comparison.{pdf,png} from the committed CSVs only.

Scaffold split, per-endpoint PR AUC averaged over seeds 0, 1, 2. Rows are those of tables/main_results.tex
(via scripts/make_results_table.py) except the two Phase 1 logistic-regression rows (10 rows).
Inputs: results/results_test.csv and results/dl_results_test_*.csv.

Run from tox21-phase1/:  python scripts/make_heatmap_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_results_table import GROUPS, ROOT, load  # noqa: E402

FIG_DIR = ROOT / "report_phase2" / "figures"
SPLIT = "scaffold"
ENDPOINTS = ["NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
             "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"]


def build() -> pd.DataFrame:
    data = load()
    rows = {}
    for _, members in GROUPS:
        for label, model, fs, src in members:
            if model == "logreg":
                continue
            d = data[src]
            d = d[(d.model == model) & (d.feature_set == fs) & (d.split == SPLIT)]
            if sorted(d.seed.unique()) != [0, 1, 2]:
                raise SystemExit(f"{label}: expected seeds [0, 1, 2]")
            rows[label] = d.groupby("endpoint")["pr_auc"].mean()
    return pd.DataFrame(rows).T[ENDPOINTS]


def main() -> None:
    hm = build()
    fig, ax = plt.subplots(figsize=(13, 5.6))
    sns.heatmap(hm, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Scaffold test PR AUC"},
                linewidths=0.5, ax=ax)
    ax.set_title("Per-endpoint PR AUC (scaffold test, mean over seeds 0-2)", pad=12, fontweight="bold")
    ax.set_xlabel("Endpoint")
    ax.set_ylabel("")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "per_endpoint_heatmap_comparison.pdf")
    fig.savefig(FIG_DIR / "per_endpoint_heatmap_comparison.png", dpi=300)
    plt.close(fig)
    pd.set_option("display.width", 250)
    print(hm.round(3).to_string())
    print("\nmean over rows:\n", hm.mean().round(3).to_string())
    print("\nmin/max over rows:\n", pd.DataFrame({"min": hm.min(), "max": hm.max()}).round(3).to_string())


if __name__ == "__main__":
    main()
