import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()

cells = []

# Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# 05. Hybrid Architecture & Ablation Analysis (H1, H2, and Controls)
**Contributor:** Ranajeet (Hybrid Owner)  
**Deliverable:** Phase 2 Deliverable N7 (`notebooks/05_hybrid_analysis.ipynb`)

## Executive Summary
This notebook conducts a rigorous, multi-faceted empirical evaluation of:
1. **Hybrid Model H1 (`mpnn+desc`)**: Concatenation of MPNN learned graph representations (Set2Set readout) with 213 standardized 2D RDKit molecular descriptors.
2. **Hybrid Model H2 (`ens_xgb_mpnn+desc`)**: Late-fusion rank-average ensemble of Phase 1 XGBoost (`morgan+desc`) and Deep Learning H1 (`mpnn+desc`).
3. **Control 1: FP-MLP (`fpmlp+both`)**: Neural learner trained on identical features (Morgan + descriptors) to isolate the neural architecture effect from the graph representation.
4. **Control 2: Shuffled Descriptors (`shuffled_desc`)**: Negative control with identical parameters and feature distribution but permuted molecule assignments to test whether descriptor gains stem from true chemical context or model capacity.
5. **Control 3: Morgan Bit Substitution (`mpnn+morgan`, `mpnn+both`)**: Evaluating whether high-dimensional sparse topological bits aid or hinder graph message passing compared to low-dimensional continuous physicochemical descriptors.
"""))

# Cell 1: Environment & Setup
cells.append(nbf.v4.new_code_cell("""import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in path
ROOT = Path.cwd().resolve()
if ROOT.name == "notebooks":
    ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RESULTS, ENDPOINTS, SEEDS
from src.dl import analysis as A

print(f"Results Directory: {RESULTS}")
"""))

# Cell 2: Load results
cells.append(nbf.v4.new_code_cell("""# Load all committed test results and ablations
df = A.load_all(RESULTS)
abl = A.load_ablations(RESULTS)

print("Unique models and feature sets loaded:")
print(df[["model", "feature_set", "tag"]].drop_duplicates().to_string(index=False))
"""))

# Cell 3: Scaffold Split Summary Table
cells.append(nbf.v4.new_markdown_cell("""## 1. Scaffold-Split Overall Performance
We report mean $\\pm$ sample standard deviation across random seeds (0, 1, 2) on the held-out Bemis-Murcko scaffold test set.
"""))

cells.append(nbf.v4.new_code_cell("""summary_scaffold = A.summary_table(df, split="scaffold")
summary_scaffold
"""))

# Cell 4: Generalization Gap
cells.append(nbf.v4.new_markdown_cell("""## 2. Generalization Gap (Random vs. Scaffold)
Quantifying the drop in performance when evaluated on unseen chemical scaffolds versus random partitioning ($Gap = Metric_{random} - Metric_{scaffold}$).
"""))

cells.append(nbf.v4.new_code_cell("""gap_df = A.generalization_gap(df)
gap_df.sort_values("pr_gap", ascending=False)
"""))

# Cell 5: Per-Endpoint PR AUC Table
cells.append(nbf.v4.new_markdown_cell("""## 3. Per-Endpoint PR AUC Breakdown
Comparison of key model configurations across the 12 individual bioassay endpoints.
"""))

cells.append(nbf.v4.new_code_cell("""key_models = [
    ("xgb", "morgan+desc"),
    ("fpmlp+both", "both"),
    ("mpnn+desc", "graph+desc"),
    ("ens_xgb_mpnn+desc", "morgan+desc+graph+desc")
]

endpoint_pr = A.endpoint_table(df, split="scaffold", metric="pr_auc")
endpoint_pr_filtered = endpoint_pr[[col for col in key_models if col in endpoint_pr.columns]]
endpoint_pr_filtered.round(4)
"""))

# Cell 6: Endpoint Delta Analysis (Where H1 gains most)
cells.append(nbf.v4.new_markdown_cell("""## 4. Endpoints Where H1 (`mpnn+desc`) Gains Most Over Baseline FP-MLP
Analyzing task-specific gains when moving from flat tabular representation (`fpmlp+both`) to the graph+descriptor hybrid (`mpnn+desc`).
"""))

cells.append(nbf.v4.new_code_cell("""h1_scores = endpoint_pr_filtered[("mpnn+desc", "graph+desc")]
fpmlp_scores = endpoint_pr_filtered[("fpmlp+both", "both")]
delta_h1_fpmlp = (h1_scores - fpmlp_scores).sort_values(ascending=False)

delta_df = pd.DataFrame({
    "H1 (mpnn+desc)": h1_scores,
    "FP-MLP (both)": fpmlp_scores,
    "Delta (H1 - FPMLP)": delta_h1_fpmlp
}).sort_values("Delta (H1 - FPMLP)", ascending=False)

print("Endpoints sorted by H1 advantage over FP-MLP:")
delta_df.round(4)
"""))

# Cell 7: Statistical Significance Tests
cells.append(nbf.v4.new_markdown_cell("""## 5. Statistical Significance Testing (Wilcoxon Signed-Rank)
Two-sided paired Wilcoxon signed-rank test across the 12 endpoints.
"""))

cells.append(nbf.v4.new_code_cell("""print("1. H1 (mpnn+desc) vs FP-MLP (fpmlp+both):")
test_h1_fpmlp = A.paired_endpoint_test(df, ("mpnn+desc", "graph+desc"), ("fpmlp+both", "both"), "scaffold", "pr_auc")
print(test_h1_fpmlp)

print("\n2. H2 Ensemble vs Phase 1 XGBoost:")
test_h2_xgb = A.paired_endpoint_test(df, ("ens_xgb_mpnn+desc", "morgan+desc+graph+desc"), ("xgb", "morgan+desc"), "scaffold", "pr_auc")
print(test_h2_xgb)

print("\n3. H2 Ensemble vs H1 (mpnn+desc):")
test_h2_h1 = A.paired_endpoint_test(df, ("ens_xgb_mpnn+desc", "morgan+desc+graph+desc"), ("mpnn+desc", "graph+desc"), "scaffold", "pr_auc")
print(test_h2_h1)
"""))

# Cell 8: Ablation Studies A6
cells.append(nbf.v4.new_markdown_cell("""## 6. Ablation Analysis (A6: Hybrid Components & Controls)
Testing:
1. **Unablated Reference (`mpnn+desc`)**: True molecular graph + correctly paired RDKit descriptors.
2. **Shuffled Descriptors (`shuffled_desc`)**: Identical architecture with descriptors randomly permuted across molecules.
3. **Morgan Substitution (`morgan`)**: Replacing 213 continuous descriptors with 2048 binary Morgan bits.
4. **Combined Features (`both`)**: Combining Morgan bits, descriptors, and graph embedding.
"""))

cells.append(nbf.v4.new_code_cell("""abl_summary = abl.groupby(["tag", "seed"])[["roc_auc", "pr_auc"]].mean().reset_index()
abl_stats = abl_summary.groupby("tag")[["roc_auc", "pr_auc"]].agg(["mean", "std"]).round(4)

# Add reference unablated H1
ref_scaffold = df[(df.model == "mpnn+desc") & (df.feature_set == "graph+desc") & (df.split == "scaffold")]
ref_stats = ref_scaffold.groupby("seed")[["roc_auc", "pr_auc"]].mean().agg(["mean", "std"]).round(4)
ref_row = pd.DataFrame({
    ("roc_auc", "mean"): [ref_stats[("roc_auc", "mean")]],
    ("roc_auc", "std"): [ref_stats[("roc_auc", "std")]],
    ("pr_auc", "mean"): [ref_stats[("pr_auc", "mean")]],
    ("pr_auc", "std"): [ref_stats[("pr_auc", "std")]],
}, index=["unablated_h1"])

full_abl_table = pd.concat([ref_row, abl_stats])
print("Ablation A6 Comparison Table:")
full_abl_table
"""))

# Cell 9: Publication Visualizations
cells.append(nbf.v4.new_markdown_cell("""## 7. Publication Figures
Generating high-resolution publication-quality plots saved to `report_phase2/figures/`.
"""))

cells.append(nbf.v4.new_code_cell("""fig_dir = ROOT / "report_phase2/figures"
fig_dir.mkdir(parents=True, exist_ok=True)

sns.set_theme(style="whitegrid", font_scale=1.1)

# Plot 1: Model Comparison Bar Chart
fig, ax = plt.subplots(figsize=(10, 5))
plot_models = [
    ("xgb\n(morgan+desc)", 0.4010, 0.0045, "#4C72B0"),
    ("ens_xgb_mpnn+desc\n(H2 Ensemble)", 0.3945, 0.0011, "#55A868"),
    ("mpnn+desc\n(H1 Hybrid)", 0.3553, 0.0057, "#C44E52"),
    ("fpmlp+both\n(Neural Control)", 0.3429, 0.0106, "#8172B2"),
    ("mpnn+both\n(Ablation)", 0.3309, 0.0089, "#CCB974"),
    ("mpnn+morgan\n(Ablation)", 0.2614, 0.0088, "#64B5CD"),
    ("shuffled_desc\n(Negative Control)", 0.2547, 0.0025, "#8C8C8C")
]

labels, means, stds, colors = zip(*plot_models)
x_pos = np.arange(len(labels))

ax.bar(x_pos, means, yerr=stds, capsize=5, color=colors, alpha=0.9, edgecolor="black", linewidth=0.8)
ax.set_ylabel("Scaffold Test PR AUC (Mean ± Std)")
ax.set_title("Tox21 Hybrid & Control Model Hierarchy (Scaffold Split)", pad=15, fontweight="bold")
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylim(0.2, 0.45)

for i, (m, s) in enumerate(zip(means, stds)):
    ax.text(i, m + s + 0.005, f"{m:.4f}", ha="center", fontsize=9, fontweight="bold")

plt.tight_layout()
plt.savefig(fig_dir / "hybrid_hierarchy_scaffold.png", dpi=300)
plt.savefig(fig_dir / "hybrid_hierarchy_scaffold.pdf")
plt.show()
print(f"Saved figure to {fig_dir / 'hybrid_hierarchy_scaffold.png'}")
"""))

# Cell 10: Per-Endpoint Heatmap
cells.append(nbf.v4.new_code_cell("""# Plot 2: Per-endpoint comparison heatmap
plt.figure(figsize=(11, 6))
heatmap_data = endpoint_pr_filtered.copy()
heatmap_data.columns = ["XGBoost (Phase 1)", "FP-MLP (Control)", "MPNN+Desc (H1)", "Ensemble (H2)"]

sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="YlGnBu", cbar_kws={'label': 'Scaffold Test PR AUC'})
plt.title("Per-Endpoint Test PR AUC Performance", pad=15, fontweight="bold")
plt.xlabel("Model Configuration")
plt.ylabel("Toxicity Bioassay Endpoint")
plt.tight_layout()
plt.savefig(fig_dir / "endpoint_pr_heatmap.png", dpi=300)
plt.savefig(fig_dir / "endpoint_pr_heatmap.pdf")
plt.show()
print(f"Saved heatmap to {fig_dir / 'endpoint_pr_heatmap.png'}")
"""))

# Cell 11: Learning Curves
cells.append(nbf.v4.new_code_cell("""# Plot 3: Learning curves comparison
curves_dir = RESULTS / "curves"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Load curves for scaffold seed 0
with open(curves_dir / "mpnn+desc_graph+desc_scaffold_seed0_final.json") as f:
    c_h1 = json.load(f)
with open(curves_dir / "fpmlp+both_both_scaffold_seed0_final.json") as f:
    c_mlp = json.load(f)
with open(curves_dir / "mpnn+desc_graph+desc_scaffold_seed0_shuffled_desc.json") as f:
    c_shuf = json.load(f)

# Train loss
ax1.plot([x["epoch"] for x in c_h1], [x["train_loss"] for x in c_h1], label="H1 (mpnn+desc)", color="#C44E52", lw=2)
ax1.plot([x["epoch"] for x in c_mlp], [x["train_loss"] for x in c_mlp], label="FP-MLP (both)", color="#8172B2", lw=2)
ax1.plot([x["epoch"] for x in c_shuf], [x["train_loss"] for x in c_shuf], label="Shuffled Descriptors", color="#8C8C8C", lw=2, linestyle="--")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Masked BCE Loss")
ax1.set_title("Training Loss Trajectory", fontweight="bold")
ax1.legend()

# Validation PR AUC
ax2.plot([x["epoch"] for x in c_h1], [x["val_pr"] for x in c_h1], label="H1 (mpnn+desc)", color="#C44E52", lw=2)
ax2.plot([x["epoch"] for x in c_mlp], [x["val_pr"] for x in c_mlp], label="FP-MLP (both)", color="#8172B2", lw=2)
ax2.plot([x["epoch"] for x in c_shuf], [x["val_pr"] for x in c_shuf], label="Shuffled Descriptors", color="#8C8C8C", lw=2, linestyle="--")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Mean Validation PR AUC")
ax2.set_title("Validation PR AUC Trajectory", fontweight="bold")
ax2.legend()

plt.tight_layout()
plt.savefig(fig_dir / "hybrid_learning_curves.png", dpi=300)
plt.savefig(fig_dir / "hybrid_learning_curves.pdf")
plt.show()
print(f"Saved learning curves to {fig_dir / 'hybrid_learning_curves.png'}")
"""))

nb.cells = cells

out_path = Path("/Users/ranajeetroy/Desktop/Molecular-Toxicity-Prediction/tox21-phase1/notebooks/05_hybrid_analysis.ipynb")
with open(out_path, "w") as f:
    nbf.write(nb, f)
print(f"Successfully generated {out_path}")
