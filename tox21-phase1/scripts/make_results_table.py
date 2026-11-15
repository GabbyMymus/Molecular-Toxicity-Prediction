"""Build the main results table (report_phase2/tables/main_results.tex) from the committed CSVs only.

Phase 1 rows come from results/results_test.csv, deep-learning rows from results/dl_results_test_*.csv.
For every (model, feature set, split): mean over the 12 endpoints per seed, then mean and sample standard
deviation (ddof=1) over seeds 0, 1, 2. No number is typed in by hand.

Run from tox21-phase1/:  python scripts/make_results_table.py
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report_phase2" / "tables" / "main_results.tex"

# (label, model, feature_set, source) ; source is "p1" (results_test.csv) or "dl" (dl_results_test_*.csv)
GROUPS: list[tuple[str, list[tuple[str, str, str, str]]]] = [
    ("Phase 1 baselines", [
        ("LogReg, M", "logreg", "morgan", "p1"),
        ("LogReg, M+D", "logreg", "morgan+desc", "p1"),
        ("RF, M", "rf", "morgan", "p1"),
        ("RF, M+D", "rf", "morgan+desc", "p1"),
        ("XGBoost, M", "xgb", "morgan", "p1"),
        ("XGBoost, M+D", "xgb", "morgan+desc", "p1"),
    ]),
    ("Deep and hybrid models", [
        ("D0 FP-MLP, M+D", "fpmlp+both", "both", "dl"),
        ("D1 GCN", "gcn", "graph", "dl"),
        ("D2 MPNN", "mpnn", "graph", "dl"),
        ("H1 MPNN+D", "mpnn+desc", "graph+desc", "dl"),
        ("H2 XGBoost+FP-MLP", "ens_xgb_fpmlp+both", "morgan+desc+both", "dl"),
        ("H2 alt. XGBoost+H1", "ens_xgb_mpnn+desc", "morgan+desc+graph+desc", "dl"),
    ]),
]
SPLITS = [("scaffold", "Scaffold test"), ("random", "Random test")]


def load() -> dict[str, pd.DataFrame]:
    p1 = pd.read_csv(RESULTS / "results_test.csv")
    files = sorted(glob.glob(str(RESULTS / "dl_results_test_*.csv")))
    dl = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return {"p1": p1, "dl": dl}


def summarize(df: pd.DataFrame, model: str, fs: str, split: str) -> dict[str, tuple[float, float]]:
    d = df[(df.model == model) & (df.feature_set == fs) & (df.split == split)]
    seeds = sorted(d.seed.unique())
    if seeds != [0, 1, 2]:
        raise SystemExit(f"{model}/{fs}/{split}: expected seeds [0, 1, 2], found {seeds}")
    per_seed = d.groupby("seed")[["pr_auc", "roc_auc"]].mean()
    return {c: (per_seed[c].mean(), per_seed[c].std(ddof=1)) for c in ("pr_auc", "roc_auc")}


def fmt(mean: float, sd: float, bold: bool) -> str:
    s = f"{mean:.4f} \\pm {sd:.4f}"
    return f"$\\mathbf{{{s}}}$" if bold else f"${s}$"


def main() -> None:
    data = load()
    stats: dict[tuple[str, str], dict[str, tuple[float, float]]] = {}
    for split, _ in SPLITS:
        for _, rows in GROUPS:
            for label, model, fs, src in rows:
                stats[(label, split)] = summarize(data[src], model, fs, split)
    best = {(split, c): max(stats[(lab, split)][c][0] for _, rows in GROUPS for lab, *_ in rows)
            for split, _ in SPLITS for c in ("pr_auc", "roc_auc")}

    lines = [
        "\\begin{table}[t]",
        "\\caption{Test performance (mean $\\pm$ sample SD over seeds 0, 1, 2 of the per-seed mean over the 12 endpoints). "
        "M: Morgan fingerprints, D: RDKit descriptors. Best value per column in bold.}",
        "\\label{tab:main_results}",
        "\\centering\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\begin{tabular}{@{}lcc@{}}",
        "\\toprule",
        "Model & PR~AUC & ROC~AUC \\\\",
    ]
    for si, (split, title) in enumerate(SPLITS):
        lines += ["\\midrule", f"\\multicolumn{{3}}{{@{{}}l}}{{\\textit{{{title}}}}} \\\\"]
        for gname, rows in GROUPS:
            for label, *_ in rows:
                s = stats[(label, split)]
                cells = [fmt(*s[c], abs(s[c][0] - best[(split, c)]) < 1e-12) for c in ("pr_auc", "roc_auc")]
                lines.append(f"{label} & {cells[0]} & {cells[1]} \\\\")
            if gname == GROUPS[0][0]:
                lines.append("\\addlinespace[2pt]")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}", ""]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT.relative_to(ROOT)} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
