#!/usr/bin/env python3
"""Execute full analysis for Deliverable A3 and generate publication figures in report_phase2/figures/.

Author: Abhijeet (Research and Analysis Owner)
"""

import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RESULTS, ENDPOINTS, SEEDS
from src.data import load_clean
from src.dl import analysis as A
from src.dl.split_io import load_split_named

fig_dir = ROOT / "report_phase2" / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

print("Loading test results and ablations from:", RESULTS)
df = A.load_all(RESULTS)
abl = A.load_ablations(RESULTS)

# 1. Summary tables
scaffold_summary = A.summary_table(df, "scaffold")
random_summary = A.summary_table(df, "random")
gap_df = A.generalization_gap(df)

print("\n--- Scaffold Summary Table ---")
print(scaffold_summary)

print("\n--- Generalization Gap (Random - Scaffold) ---")
print(gap_df)

# 2. Paired tests
print("\n--- Paired Wilcoxon Signed-Rank Tests (12 Endpoints) ---")
test_h1_vs_mlp = A.paired_endpoint_test(df, ("mpnn+desc", "graph+desc"), ("fpmlp+both", "both"))
print("H1 vs FP-MLP:", test_h1_vs_mlp)

test_h2_vs_h1 = A.paired_endpoint_test(df, ("ens_xgb_mpnn+desc", "morgan+desc+graph+desc"), ("mpnn+desc", "graph+desc"))
print("H2 vs H1:", test_h2_vs_h1)

test_h2_vs_xgb = A.paired_endpoint_test(df, ("ens_xgb_mpnn+desc", "morgan+desc+graph+desc"), ("xgb", "morgan+desc"))
print("H2 vs XGBoost:", test_h2_vs_xgb)

test_h1_vs_xgb = A.paired_endpoint_test(df, ("mpnn+desc", "graph+desc"), ("xgb", "morgan+desc"))
print("H1 vs XGBoost:", test_h1_vs_xgb)

# 3. Bootstrap CIs
print("\n--- Joint Bootstrap 95% CIs on Scaffold Test Predictions (Seed 0) ---")
_, Y_all = load_clean()
sp_scaffold = load_split_named("scaffold")
Yte = Y_all[sp_scaffold["test"]]

p_h1 = np.load(RESULTS / "preds" / "mpnn+desc_graph+desc_scaffold_seed0_final.npy")
p_h2 = np.load(RESULTS / "preds" / "ens_xgb_mpnn+desc_morgan+desc+graph+desc_scaffold_seed0_final.npy")
p_mlp = np.load(RESULTS / "preds" / "fpmlp+both_both_scaffold_seed0_final.npy")
p_xgb = np.load(RESULTS / "preds" / "xgb_morgan+desc_scaffold_seed0.npy")

boot_rows = []
for j, ep in enumerate(ENDPOINTS):
    d_h1_mlp, l_h1_mlp, h_h1_mlp = A.paired_bootstrap_ci(Yte[:, j], p_h1[:, j], p_mlp[:, j], "pr_auc")
    d_h2_h1, l_h2_h1, h_h2_h1 = A.paired_bootstrap_ci(Yte[:, j], p_h2[:, j], p_h1[:, j], "pr_auc")
    d_h2_xgb, l_h2_xgb, h_h2_xgb = A.paired_bootstrap_ci(Yte[:, j], p_h2[:, j], p_xgb[:, j], "pr_auc")
    boot_rows.append({
        "endpoint": ep,
        "h1_minus_mlp_diff": d_h1_mlp, "h1_minus_mlp_ci_low": l_h1_mlp, "h1_minus_mlp_ci_high": h_h1_mlp,
        "h2_minus_h1_diff": d_h2_h1, "h2_minus_h1_ci_low": l_h2_h1, "h2_minus_h1_ci_high": h_h2_h1,
        "h2_minus_xgb_diff": d_h2_xgb, "h2_minus_xgb_ci_low": l_h2_xgb, "h2_minus_xgb_ci_high": h_h2_xgb,
    })
boot_df = pd.DataFrame(boot_rows)
print(boot_df[["endpoint", "h1_minus_mlp_diff", "h2_minus_h1_diff", "h2_minus_xgb_diff"]])

# 4. Ablations
abl_table = A.ablation_table(abl, df, "mpnn+desc", "graph+desc")
print("\n--- Hybrid Ablation Table ---")
print(abl_table)

# ----------------- PLOTTING FIGURES -----------------
sns.set_theme(style="whitegrid", font_scale=1.05)

# FIGURE 1: Mean PR & ROC AUC Bar Chart
print("\nPlotting Figure 1: mean_pr_roc_scaffold...")
display_names = {
    ("rf", "morgan+desc"): "RF (Morgan+Desc)",
    ("xgb", "morgan+desc"): "XGBoost (Morgan+Desc)",
    ("ens_xgb_mpnn+desc", "morgan+desc+graph+desc"): "H2: Ensemble (XGB+H1)",
    ("mpnn+desc", "graph+desc"): "H1: MPNN + Descriptors",
    ("fpmlp+both", "both"): "FP-MLP Control",
    ("rf", "morgan"): "RF (Morgan)",
    ("xgb", "morgan"): "XGBoost (Morgan)",
    ("logreg", "morgan+desc"): "LogReg (Morgan+Desc)",
    ("logreg", "morgan"): "LogReg (Morgan)",
}

summary_scaff = scaffold_summary.copy()
summary_scaff["display_name"] = [display_names.get(idx, f"{idx[0]}_{idx[1]}") for idx in summary_scaff.index]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.2))
summary_scaff_pr = summary_scaff.sort_values("pr_auc_mean", ascending=True)
colors_pr = ["#3274A1" if "H1" in n or "H2" in n or "FP-MLP" in n else "#55A868" for n in summary_scaff_pr["display_name"]]
ax1.barh(summary_scaff_pr["display_name"], summary_scaff_pr["pr_auc_mean"], xerr=summary_scaff_pr["pr_auc_std"],
         color=colors_pr, alpha=0.9, capsize=4, edgecolor="black", linewidth=0.8)
ax1.set_xlabel("Mean PR AUC ± Seed Std")
ax1.set_title("Scaffold Test PR AUC (Primary Precision Metric)", pad=10, fontweight="bold")
ax1.set_xlim(0.25, 0.45)
ax1.grid(axis="x", linestyle="--", alpha=0.5)

summary_scaff_roc = summary_scaff.sort_values("roc_auc_mean", ascending=True)
colors_roc = ["#3274A1" if "H1" in n or "H2" in n or "FP-MLP" in n else "#55A868" for n in summary_scaff_roc["display_name"]]
ax2.barh(summary_scaff_roc["display_name"], summary_scaff_roc["roc_auc_mean"], xerr=summary_scaff_roc["roc_auc_std"],
         color=colors_roc, alpha=0.9, capsize=4, edgecolor="black", linewidth=0.8)
ax2.set_xlabel("Mean ROC AUC ± Seed Std")
ax2.set_title("Scaffold Test ROC AUC", pad=10, fontweight="bold")
ax2.set_xlim(0.65, 0.82)
ax2.grid(axis="x", linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig(fig_dir / "mean_pr_roc_scaffold.pdf")
fig.savefig(fig_dir / "mean_pr_roc_scaffold.png", dpi=300)
plt.close(fig)
print("Saved mean_pr_roc_scaffold to", fig_dir)

# FIGURE 2: Per-Endpoint PR AUC Heatmap
print("Plotting Figure 2: per_endpoint_heatmap_comparison...")
scaffold_df = df[df["split"] == "scaffold"].copy()
scaffold_means = scaffold_df.groupby(["model", "feature_set", "endpoint"])["pr_auc"].mean().reset_index()

target_models = [
    ("xgb", "morgan+desc", "XGBoost (Morgan+Desc)"),
    ("rf", "morgan+desc", "RF (Morgan+Desc)"),
    ("fpmlp+both", "both", "FP-MLP Control"),
    ("mpnn+desc", "graph+desc", "H1 (MPNN+Desc)"),
    ("ens_xgb_mpnn+desc", "morgan+desc+graph+desc", "H2 (Ensemble)"),
]

heatmap_rows = []
for m, fs, label in target_models:
    sub = scaffold_means[(scaffold_means["model"] == m) & (scaffold_means["feature_set"] == fs)]
    ep_map = dict(zip(sub["endpoint"], sub["pr_auc"]))
    ep_map["Model"] = label
    heatmap_rows.append(ep_map)

hm_df = pd.DataFrame(heatmap_rows).set_index("Model")[ENDPOINTS]

fig, ax = plt.subplots(figsize=(13, 5))
sns.heatmap(hm_df, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={"label": "Scaffold Test PR AUC"}, linewidths=0.5, ax=ax)
ax.set_title("Per-Endpoint PR AUC across Tox21 Assays (Scaffold Split, 3-Seed Average)", pad=12, fontweight="bold")
ax.set_xlabel("Assay Endpoint")
ax.set_ylabel("")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()
fig.savefig(fig_dir / "per_endpoint_heatmap_comparison.pdf")
fig.savefig(fig_dir / "per_endpoint_heatmap_comparison.png", dpi=300)
plt.close(fig)
print("Saved per_endpoint_heatmap_comparison to", fig_dir)

# FIGURE 3: Generalization Gap
print("Plotting Figure 3: generalization_gap...")
gap_plot = gap_df.copy()
gap_plot["display_name"] = [display_names.get(idx, f"{idx[0]}_{idx[1]}") for idx in gap_plot.index]
gap_plot = gap_plot.sort_values("pr_gap", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(gap_plot))
width = 0.35

ax.bar(x - width/2, gap_plot["pr_gap"], width, label="PR AUC Gap (Random - Scaffold)", color="#C44E52", alpha=0.9, edgecolor="black")
ax.bar(x + width/2, gap_plot["roc_gap"], width, label="ROC AUC Gap (Random - Scaffold)", color="#4C72B0", alpha=0.9, edgecolor="black")

ax.set_ylabel("Generalization Penalty (Δ AUC)")
ax.set_title("Scaffold Generalization Gap (Performance Drop on Unseen Chemotypes)", pad=12, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(gap_plot["display_name"], rotation=25, ha="right")
ax.legend(frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig(fig_dir / "generalization_gap.pdf")
fig.savefig(fig_dir / "generalization_gap.png", dpi=300)
plt.close(fig)
print("Saved generalization_gap to", fig_dir)

# FIGURE 4: Ablation Study
print("Plotting Figure 4: ablation_study...")
fig, ax = plt.subplots(figsize=(8, 4.5))

labels = [
    "Reference H1\n(MPNN+Desc)",
    "Ablation A6c\n(Both: Graph+FP+Desc)",
    "Ablation A6b\n(Morgan: Graph+FP)",
    "Ablation A6a: Control\n(Shuffled Descriptors)"
]
scores = abl_table["pr_auc"].tolist()
colors = ["#2CA02C", "#1F77B4", "#FF7F0E", "#D62728"]

bars = ax.bar(labels, scores, color=colors, alpha=0.88, edgecolor="black", linewidth=1.0, width=0.52)
for b in bars:
    h = b.get_height()
    ax.text(b.get_x() + b.get_width()/2., h + 0.007, f"{h:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=10)

ax.set_ylabel("Scaffold Test PR AUC")
ax.set_title("Ablation Study: Real Molecular Descriptors vs. Representation Controls", pad=12, fontweight="bold")
ax.set_ylim(0.20, 0.40)
ax.grid(axis="y", linestyle="--", alpha=0.5)

plt.tight_layout()
fig.savefig(fig_dir / "ablation_study.pdf")
fig.savefig(fig_dir / "ablation_study.png", dpi=300)
plt.close(fig)
print("Saved ablation_study to", fig_dir)

# FIGURE 5: Learning curves
print("Plotting Figure 5: dl_learning_curves...")
curve_dir = RESULTS / "curves"

with open(curve_dir / "mpnn+desc_graph+desc_scaffold_seed0_final.json") as f:
    c_h1 = json.load(f)
with open(curve_dir / "fpmlp+both_both_scaffold_seed0_final.json") as f:
    c_mlp = json.load(f)
with open(curve_dir / "mpnn+desc_graph+desc_scaffold_seed0_shuffled_desc.json") as f:
    c_shuf = json.load(f)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

# Train Loss
ax1.plot([e["epoch"] for e in c_h1], [e["train_loss"] for e in c_h1], label="H1 (MPNN+Desc)", color="#1F77B4", linewidth=2)
ax1.plot([e["epoch"] for e in c_mlp], [e["train_loss"] for e in c_mlp], label="FP-MLP Control", color="#2CA02C", linewidth=2, linestyle="--")
ax1.plot([e["epoch"] for e in c_shuf], [e["train_loss"] for e in c_shuf], label="Shuffled Desc Control", color="#D62728", linewidth=1.8, linestyle=":")
ax1.set_xlabel("Training Epoch")
ax1.set_ylabel("Masked BCE Loss")
ax1.set_title("Training Loss Convergence", fontweight="bold")
ax1.grid(True, linestyle="--", alpha=0.5)
ax1.legend(frameon=True)

# Val PR AUC
ax2.plot([e["epoch"] for e in c_h1], [e["val_pr"] for e in c_h1], label="H1 (MPNN+Desc)", color="#1F77B4", linewidth=2)
ax2.plot([e["epoch"] for e in c_mlp], [e["val_pr"] for e in c_mlp], label="FP-MLP Control", color="#2CA02C", linewidth=2, linestyle="--")
ax2.plot([e["epoch"] for e in c_shuf], [e["val_pr"] for e in c_shuf], label="Shuffled Desc Control", color="#D62728", linewidth=1.8, linestyle=":")
ax2.set_xlabel("Training Epoch")
ax2.set_ylabel("Mean Validation PR AUC")
ax2.set_title("Validation PR AUC (Early Stopping Metric)", fontweight="bold")
ax2.grid(True, linestyle="--", alpha=0.5)
ax2.legend(frameon=True)

plt.tight_layout()
fig.savefig(fig_dir / "dl_learning_curves.pdf")
fig.savefig(fig_dir / "dl_learning_curves.png", dpi=300)
plt.close(fig)
print("Saved dl_learning_curves to", fig_dir)
print("\nAll analysis calculations and publication figures completed successfully!")
