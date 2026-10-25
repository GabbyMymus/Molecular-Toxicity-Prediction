"""Generate dataset EDA figures and a compact, reproducible statistics summary."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

from src.config import CLEAN_PATH, ENDPOINTS, ROOT
from src.splits import load_split


def main():
    frame = pd.read_csv(CLEAN_PATH).fillna({"scaffold": ""})
    y = frame[ENDPOINTS]
    figure_dir = ROOT / "figures"
    figure_dir.mkdir(exist_ok=True)
    plt.rcParams.update({"figure.dpi": 140, "savefig.dpi": 220, "font.size": 8})

    # Each row is one molecule; the plot is downsampled only for legibility.
    missing = y.isna().to_numpy()
    stride = max(1, len(frame) // 1200)
    fig, ax = plt.subplots(figsize=(8.2, 4.1))
    ax.imshow(missing[::stride].T, aspect="auto", interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    ax.set_yticks(range(len(ENDPOINTS)), ENDPOINTS)
    ax.set_xlabel(f"Molecules (every {stride}th row shown)")
    ax.set_title("Missing assay labels")
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_missingness.png", bbox_inches="tight")
    plt.close(fig)

    rates = pd.DataFrame({"missing": y.isna().mean(), "active": y.eq(1).sum(), "inactive": y.eq(0).sum()})
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    rates["missing"].sort_values().plot.barh(ax=axes[0], color="#457b9d")
    axes[0].set(xlabel="Fraction missing", title="Missing-label rate")
    rates[["inactive", "active"]].plot.bar(ax=axes[1], stacked=True, color=["#a8dadc", "#e76f51"])
    axes[1].set(ylabel="Molecules", title="Known labels by endpoint")
    axes[1].tick_params(axis="x", rotation=70)
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_labels.png", bbox_inches="tight")
    plt.close(fig)

    scaffold_counts = frame.groupby("scaffold", dropna=False).size().sort_values(ascending=False)
    split = {key: np.asarray(value, dtype=int) for key, value in load_split(ROOT / "data/splits/scaffold_seed0.json").items()}
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))
    axes[0].hist(scaffold_counts.to_numpy(), bins=np.logspace(0, np.log10(max(2, scaffold_counts.max())), 28), color="#2a9d8f")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set(xlabel="Molecules per scaffold (log)", ylabel="Number of scaffolds (log)", title="Scaffold group sizes")
    axes[1].bar(["Train", "Validation", "Test"], [len(split[k]) for k in ("train", "val", "test")], color=["#457b9d", "#e9c46a", "#e76f51"])
    axes[1].set(ylabel="Molecules", title="Scaffold split sizes")
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_scaffolds.png", bbox_inches="tight")
    plt.close(fig)

    mols = [Chem.MolFromSmiles(s) for s in frame.smiles]
    distributions = {
        "Molecular weight (Da)": [Descriptors.MolWt(m) for m in mols],
        "Heavy atoms": [m.GetNumHeavyAtoms() for m in mols],
        "Ring count": [rdMolDescriptors.CalcNumRings(m) for m in mols],
    }
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    for ax, (title, values) in zip(axes, distributions.items()):
        if title == "Ring count":
            values = np.asarray(values)
            values = values[values <= np.quantile(values, 0.99)]
            bins = np.arange(-0.5, values.max() + 1.5, 1)
        else:
            bins = 35
        ax.hist(values, bins=bins, color="#6d597a", alpha=0.88)
        ax.set(title=title, ylabel="Molecules")
    fig.tight_layout()
    fig.savefig(figure_dir / "eda_molecule_properties.png", bbox_inches="tight")
    plt.close(fig)

    with (ROOT / "data/processed/dedup_log.txt").open() as handle:
        audit = handle.read().strip()
    print(audit)
    print("rows", len(frame), "unique_scaffolds", len(scaffold_counts),
          "acyclic_rows", int(frame.scaffold.eq("").sum()),
          "top10_scaffold_rows", scaffold_counts.head(10).to_list())
    print("endpoint stats (known, active, missing):")
    for endpoint in ENDPOINTS:
        print(endpoint, int(y[endpoint].notna().sum()), int(y[endpoint].eq(1).sum()), int(y[endpoint].isna().sum()))


if __name__ == "__main__":
    main()
