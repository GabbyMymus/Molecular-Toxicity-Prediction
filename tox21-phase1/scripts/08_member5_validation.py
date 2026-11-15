"""Check saved prediction alignment and summarize test-set seed variability."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dl.member5_analysis import (  # noqa: E402
    audit_saved_predictions,
    seed_variance_table,
    write_latex_seed_variance,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--splits", default="scaffold,random")
    args = parser.parse_args()

    audits, metrics = [], []
    for split in args.splits.split(","):
        audit, per_endpoint = audit_saved_predictions(split=split, results_dir=args.results_dir)
        audits.append(audit)
        metrics.append(per_endpoint)
    audit_frame = pd.concat(audits, ignore_index=True)
    metric_frame = pd.concat(metrics, ignore_index=True)
    variance = seed_variance_table(metric_frame)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.results_dir / "member5_prediction_audit.csv"
    metrics_path = args.results_dir / "member5_prediction_metrics.csv"
    variance_path = args.results_dir / "member5_seed_variance.csv"
    latex_path = ROOT / "report_phase2/tables/member5_seed_variance.tex"
    audit_frame.to_csv(audit_path, index=False)
    metric_frame.to_csv(metrics_path, index=False)
    variance.to_csv(variance_path, index=False)
    write_latex_seed_variance(variance, latex_path)
    print(audit_frame.to_string(index=False))
    print("\nSeed variance (endpoint-average metric per seed):")
    print(variance.to_string(index=False))
    failed = audit_frame[
        (audit_frame.status == "ok")
        & audit_frame.metrics_match_reference.eq(False)
    ]
    if len(failed):
        raise SystemExit(f"{len(failed)} prediction artifact(s) disagree with saved metric CSVs")
    print(f"\nWrote {audit_path}")
    print(f"Wrote {metrics_path}")
    print(f"Wrote {variance_path}")
    print(f"Wrote {latex_path}")


if __name__ == "__main__":
    main()
