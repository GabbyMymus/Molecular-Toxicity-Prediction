"""Run Malhar's predeclared MPNN ablations and shuffled-label control.

Run from the repository root after installing requirements_phase2.txt and freezing
results/best_configs_dl.json.  Results append to a dedicated CSV and predictions
are separated by tag.  Use --dry-run to inspect the exact commands without training.
"""
from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

EXPERIMENTS = {
    "A1_depth1": ("steps=1", False),
    "A2_depth2": ("steps=2", False),
    "A3_no_bond": ("use_edges=false", False),
    "P2_depth4": ("steps=4", False),
    "P2_readout_sum": ("readout=sum", False),
    "P2_readout_mean": ("readout=mean", False),
    "C1_shuffled_labels": ("", True),
}


def command(tag: str, variant: str, shuffle_labels: bool) -> list[str]:
    cmd = [
        sys.executable, "-m", "src.dl.run",
        "--mode", "ablate", "--model", "mpnn", "--extra", "none",
        "--split", "scaffold", "--seeds", "0,1,2", "--tag", tag,
        "--output-name", "dl_ablation_test_malhar.csv",
        "--i-know-this-touches-test",
    ]
    if variant:
        cmd.extend(["--variant", variant])
    if shuffle_labels:
        cmd.append("--shuffle-labels")
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", nargs="*", choices=EXPERIMENTS, default=list(EXPERIMENTS))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run:
        missing = [name for name in ("torch", "torch_geometric") if importlib.util.find_spec(name) is None]
        if missing:
            raise SystemExit(
                "Missing Phase 2 dependencies: " + ", ".join(missing)
                + ". Install requirements_phase2.txt, then rerun this script."
            )
        if not (ROOT / "results/best_configs_dl.json").exists():
            raise SystemExit("Freeze results/best_configs_dl.json before running test ablations")

    for tag in args.only:
        variant, shuffle_labels = EXPERIMENTS[tag]
        cmd = command(tag, variant, shuffle_labels)
        print("$", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
