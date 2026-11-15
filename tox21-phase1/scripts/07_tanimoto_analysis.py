"""Reproduce the Phase 2 Tanimoto-bin comparison on the frozen scaffold test set."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dl.tanimoto import run_tanimoto_analysis  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "report_phase2")
    parser.add_argument("--bins", type=int, default=4)
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()

    _, summary, missing = run_tanimoto_analysis(
        split="scaffold",
        seeds=[int(seed) for seed in args.seeds.split(",")],
        n_bins=args.bins,
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )
    print(summary.to_string(index=False))
    if missing:
        print("Missing optional prediction artifacts:", *missing, sep="\n  ")
    print(f"Wrote {args.output_dir / 'tables/tanimoto_by_endpoint.csv'}")
    print(f"Wrote {args.output_dir / 'tables/tanimoto_binned_summary.csv'}")
    print(f"Wrote {args.output_dir / 'figures/tanimoto.pdf'}")


if __name__ == "__main__":
    main()
