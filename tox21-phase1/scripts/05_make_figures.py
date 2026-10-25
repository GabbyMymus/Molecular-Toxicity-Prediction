"""Generate publication-ready report figures and LaTeX/CSV tables from final test results.

Produces:
  Figures (figures/):
    - model_roc_auc.png
    - endpoint_bars_scaffold_morgan_desc.png
    - endpoint_bars_scaffold_morgan.png
    - scaffold_vs_random_split.png
    - descriptor_benefit.png
    - tanimoto_similarity_distributions.png

  Tables (tables/):
    - results_summary_scaffold.csv / .tex
    - results_summary_random.csv / .tex
    - descriptor_benefit.csv / .tex
    - scaffold_penalty.csv / .tex
    - endpoint_results_scaffold.csv / .tex
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import ENDPOINTS, RESULTS_PATH, ROOT

MODEL_LABELS = {
    "logreg": "Logistic Regression",
    "rf": "Random Forest",
    "xgb": "XGBoost",
}

FEAT_LABELS = {
    "morgan": "Morgan Only",
    "morgan+desc": "Morgan + Desc",
}

PALETTE = {
    "Logistic Regression": "#4c72b0",
    "Random Forest": "#55a868",
    "XGBoost": "#c44e52",
}


def make_endpoint_barchart(df, figure_dir, feature_set="morgan+desc", split="scaffold"):
    """Plot 2x1 subplots of ROC-AUC and PR-AUC per endpoint with std error bars."""
    s = df[(df["split"] == split) & (df["feature_set"] == feature_set)].copy()
    feat_title = FEAT_LABELS.get(feature_set, feature_set)
    split_title = split.capitalize()

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for ax, met, title in zip(axes, ["roc_auc", "pr_auc"], ["ROC AUC", "PR AUC"]):
        sns.barplot(
            data=s,
            x="endpoint",
            y=met,
            hue="model_label",
            errorbar="sd",
            palette=PALETTE,
            ax=ax,
            order=ENDPOINTS,
            hue_order=list(PALETTE.keys()),
        )
        ax.set_ylabel(title, fontsize=12)
        ax.set_xlabel("")
        ax.set_ylim(0, 1.0)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.legend(title="Model", loc="upper right", fontsize=9)
        ax.set_title(f"{title} per Endpoint — {split_title} split, {feat_title}", fontsize=12)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.setp(axes[-1].get_xticklabels(), rotation=35, ha="right")
    axes[-1].set_xlabel("Toxicity Endpoint", fontsize=12)
    plt.tight_layout()

    fs_slug = feature_set.replace("+", "_")
    fname = figure_dir / f"endpoint_bars_{split}_{fs_slug}.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {fname.name}")


def make_scaffold_vs_random_plot(df, figure_dir):
    """Plot mean ROC AUC and PR AUC comparison across splits."""
    avg = (
        df.groupby(["model_label", "feat_label", "split", "seed"])[["roc_auc", "pr_auc"]]
        .mean()
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    split_palette = {"scaffold": "#d62728", "random": "#1f77b4"}

    for ax, met, title in zip(axes, ["roc_auc", "pr_auc"], ["Mean ROC AUC", "Mean PR AUC"]):
        sns.barplot(
            data=avg[avg["feat_label"] == "Morgan + Desc"],
            x="model_label",
            y=met,
            hue="split",
            errorbar="sd",
            palette=split_palette,
            ax=ax,
            order=list(PALETTE.keys()),
        )
        ax.set_ylabel(title, fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{title}: Scaffold vs Random\n(Morgan+Desc, mean ± sd over seeds)", fontsize=11)
        ax.legend(title="Split")
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fname = figure_dir / "scaffold_vs_random_split.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {fname.name}")


def make_descriptor_benefit_plot(df, figure_dir):
    """Plot Δ ROC AUC and Δ PR AUC from adding descriptors."""
    by_seed = (
        df[df["split"] == "scaffold"]
        .groupby(["model", "feature_set", "seed"])[["roc_auc", "pr_auc"]]
        .mean()
        .reset_index()
    )

    m_only = by_seed[by_seed["feature_set"] == "morgan"].set_index(["model", "seed"])
    m_desc = by_seed[by_seed["feature_set"] == "morgan+desc"].set_index(["model", "seed"])
    delta = (m_desc[["roc_auc", "pr_auc"]] - m_only[["roc_auc", "pr_auc"]]).reset_index()
    delta["model_label"] = delta["model"].map(MODEL_LABELS)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, met, title in zip(axes, ["roc_auc", "pr_auc"], ["Δ ROC AUC", "Δ PR AUC"]):
        sns.barplot(
            data=delta,
            x="model_label",
            y=met,
            hue="model_label",
            legend=False,
            errorbar="sd",
            palette=PALETTE,
            ax=ax,
            order=list(PALETTE.keys()),
        )
        ax.axhline(0, color="black", linewidth=1)
        ax.set_ylabel(title, fontsize=12)
        ax.set_xlabel("Model", fontsize=12)
        ax.set_title(f"{title} from Adding Descriptors\n(Scaffold split, mean ± sd over seeds)", fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    fname = figure_dir / "descriptor_benefit.png"
    plt.savefig(fname, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved figure -> {fname.name}")


def make_tanimoto_plot(figure_dir):
    """Plot max Tanimoto similarity distribution for scaffold vs random test sets."""
    try:
        from src.features import load_features
        from src.similarity import max_tanimoto

        X_tr_sc, _, X_te_sc = load_features("morgan", "scaffold")
        X_tr_rd, _, X_te_rd = load_features("morgan", "random")

        sim_scaffold = max_tanimoto(X_te_sc, X_tr_sc)
        sim_random = max_tanimoto(X_te_rd, X_tr_rd)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(
            sim_random,
            bins=40,
            alpha=0.65,
            color="#1f77b4",
            label=f"Random split (mean = {sim_random.mean():.3f})",
        )
        ax.hist(
            sim_scaffold,
            bins=40,
            alpha=0.65,
            color="#d62728",
            label=f"Scaffold split (mean = {sim_scaffold.mean():.3f})",
        )
        ax.set_xlabel("Max Tanimoto similarity to training set", fontsize=12)
        ax.set_ylabel("Number of test molecules", fontsize=12)
        ax.set_title("Test-Set Chemical Novelty: Scaffold vs Random Split", fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(axis="y", linestyle=":", alpha=0.6)
        plt.tight_layout()

        fname = figure_dir / "tanimoto_similarity_distributions.png"
        plt.savefig(fname, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved figure -> {fname.name}")
    except Exception as e:
        print(f"Skipping Tanimoto plot: {e}")


def export_tables(df, tables_dir):
    """Generate and export publication-ready summary tables (CSV and LaTeX)."""
    tables_dir.mkdir(exist_ok=True)

    # 1. Main summary table per split
    for split in ["scaffold", "random"]:
        s = df[df["split"] == split]
        # per-seed mean across endpoints
        seed_means = (
            s.groupby(["model", "feature_set", "seed"])[["roc_auc", "pr_auc"]]
            .mean()
            .reset_index()
        )
        summary = (
            seed_means.groupby(["model", "feature_set"])[["roc_auc", "pr_auc"]]
            .agg(["mean", "std"])
            .round(4)
        )
        summary.columns = ["ROC_AUC_mean", "ROC_AUC_std", "PR_AUC_mean", "PR_AUC_std"]
        summary = summary.reset_index()
        summary["model"] = summary["model"].map(MODEL_LABELS)
        summary["feature_set"] = summary["feature_set"].map(FEAT_LABELS)

        summary.to_csv(tables_dir / f"results_summary_{split}.csv", index=False)
        summary.to_latex(
            tables_dir / f"results_summary_{split}.tex",
            index=False,
            caption=f"Mean test performance on {split.capitalize()} split (averaged across 12 endpoints and 3 seeds)",
            label=f"tab:results_{split}",
        )
        print(f"Saved table -> results_summary_{split}.csv / .tex")

    # 2. Descriptor benefit table (scaffold split)
    by_seed = (
        df[df["split"] == "scaffold"]
        .groupby(["model", "feature_set", "seed"])[["roc_auc", "pr_auc"]]
        .mean()
        .reset_index()
    )
    m_only = by_seed[by_seed["feature_set"] == "morgan"].set_index(["model", "seed"])
    m_desc = by_seed[by_seed["feature_set"] == "morgan+desc"].set_index(["model", "seed"])
    delta = (m_desc[["roc_auc", "pr_auc"]] - m_only[["roc_auc", "pr_auc"]]).reset_index()
    delta["model_label"] = delta["model"].map(MODEL_LABELS)

    desc_table = (
        delta.groupby("model_label")[["roc_auc", "pr_auc"]].agg(["mean", "std"]).round(4)
    )
    desc_table.columns = ["Delta_ROC_mean", "Delta_ROC_std", "Delta_PR_mean", "Delta_PR_std"]
    desc_table = desc_table.reset_index()
    desc_table.to_csv(tables_dir / "descriptor_benefit.csv", index=False)
    desc_table.to_latex(
        tables_dir / "descriptor_benefit.tex",
        index=False,
        caption="Benefit of adding RDKit descriptors to Morgan fingerprints (scaffold split)",
        label="tab:descriptor_benefit",
    )
    print("Saved table -> descriptor_benefit.csv / .tex")

    # 3. Scaffold penalty table (Morgan + Desc)
    per_seed_mplus = (
        df[df["feature_set"] == "morgan+desc"]
        .groupby(["model_label", "split", "seed"])[["roc_auc", "pr_auc"]]
        .mean()
        .reset_index()
    )
    s_agg = (
        per_seed_mplus[per_seed_mplus["split"] == "scaffold"]
        .groupby("model_label")[["roc_auc", "pr_auc"]]
        .mean()
        .rename(columns={"roc_auc": "Scaffold_ROC", "pr_auc": "Scaffold_PR"})
    )
    r_agg = (
        per_seed_mplus[per_seed_mplus["split"] == "random"]
        .groupby("model_label")[["roc_auc", "pr_auc"]]
        .mean()
        .rename(columns={"roc_auc": "Random_ROC", "pr_auc": "Random_PR"})
    )
    gap = s_agg.join(r_agg)
    gap["Delta_ROC_scaf_minus_rand"] = (gap["Scaffold_ROC"] - gap["Random_ROC"]).round(4)
    gap["Delta_PR_scaf_minus_rand"] = (gap["Scaffold_PR"] - gap["Random_PR"]).round(4)
    gap = gap.reset_index()
    gap.to_csv(tables_dir / "scaffold_penalty.csv", index=False)
    gap.to_latex(
        tables_dir / "scaffold_penalty.tex",
        index=False,
        caption="Scaffold split penalty: generalization drop compared to random split (Morgan+Desc)",
        label="tab:scaffold_penalty",
    )
    print("Saved table -> scaffold_penalty.csv / .tex")

    # 4. Detailed per-endpoint table (scaffold split, Morgan + Desc)
    sc_mdesc = df[(df["split"] == "scaffold") & (df["feature_set"] == "morgan+desc")]
    ep_table = (
        sc_mdesc.groupby(["endpoint", "model_label"])[["roc_auc", "pr_auc"]]
        .agg(["mean", "std"])
        .round(3)
    )
    ep_table.columns = ["ROC_mean", "ROC_std", "PR_mean", "PR_std"]
    ep_table = ep_table.reset_index()
    ep_table.to_csv(tables_dir / "endpoint_results_scaffold.csv", index=False)
    ep_table.to_latex(
        tables_dir / "endpoint_results_scaffold.tex",
        index=False,
        caption="Per-endpoint test performance on scaffold split (Morgan+Desc)",
        label="tab:endpoint_results_scaffold",
    )
    print("Saved table -> endpoint_results_scaffold.csv / .tex")


def main():
    sns.set_theme(style="whitegrid", font="sans-serif")

    results_file = RESULTS_PATH / "results_test.csv"
    figure_dir = ROOT / "figures"
    tables_dir = ROOT / "tables"
    figure_dir.mkdir(exist_ok=True)
    tables_dir.mkdir(exist_ok=True)

    if not results_file.exists() or results_file.stat().st_size <= 1:
        print(f"Results file not found or empty: {results_file}")
        return

    df = pd.read_csv(results_file)
    df["model_label"] = df["model"].map(MODEL_LABELS)
    df["feat_label"] = df["feature_set"].map(FEAT_LABELS)

    # 1. Baseline model bar chart
    fig, ax = plt.subplots(figsize=(6, 4))
    df.groupby("model_label")["roc_auc"].mean().plot.bar(
        color=["#4c72b0", "#55a868", "#c44e52"], ax=ax
    )
    ax.set_ylabel("Mean ROC-AUC", fontsize=11)
    ax.set_xlabel("Model", fontsize=11)
    ax.set_title("Overall Mean ROC-AUC Across All Configurations", fontsize=12)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figure_dir / "model_roc_auc.png", dpi=300)
    plt.close()
    print("Saved figure -> model_roc_auc.png")

    # 2. Endpoint bars
    make_endpoint_barchart(df, figure_dir, feature_set="morgan+desc", split="scaffold")
    make_endpoint_barchart(df, figure_dir, feature_set="morgan", split="scaffold")

    # 3. Scaffold vs Random
    make_scaffold_vs_random_plot(df, figure_dir)

    # 4. Descriptor benefit
    make_descriptor_benefit_plot(df, figure_dir)

    # 5. Tanimoto similarity distribution
    make_tanimoto_plot(figure_dir)

    # 6. Export all summary tables
    export_tables(df, tables_dir)

    print("\nAll publication figures and tables generated successfully.")


if __name__ == "__main__":
    main()
