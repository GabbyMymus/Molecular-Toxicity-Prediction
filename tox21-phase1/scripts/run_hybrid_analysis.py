import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path("/Users/ranajeetroy/Desktop/Molecular-Toxicity-Prediction/tox21-phase1")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import RESULTS, ENDPOINTS, SEEDS
from src.dl import analysis as A

fig_dir = ROOT / "report_phase2/figures"
fig_dir.mkdir(parents=True, exist_ok=True)

df = A.load_all(RESULTS)
abl = A.load_ablations(RESULTS)

key_models = [
    ("xgb", "morgan+desc"),
    ("fpmlp+both", "both"),
    ("mpnn+desc", "graph+desc"),
    ("ens_xgb_mpnn+desc", "morgan+desc+graph+desc")
]

endpoint_pr = A.endpoint_table(df, split="scaffold", metric="pr_auc")
endpoint_pr_filtered = endpoint_pr[[col for col in key_models if col in endpoint_pr.columns]]

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
plt.close()
print(f"Saved {fig_dir / 'hybrid_hierarchy_scaffold.png'}")

# Plot 2: Per-endpoint heatmap
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
plt.close()
print(f"Saved {fig_dir / 'endpoint_pr_heatmap.png'}")

# Plot 3: Learning curves comparison
curves_dir = RESULTS / "curves"
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

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
plt.close()
print(f"Saved {fig_dir / 'hybrid_learning_curves.png'}")
