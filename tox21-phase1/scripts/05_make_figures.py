"""Generate report figures from final results when results are available."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import RESULTS_PATH, ROOT

results = RESULTS_PATH / "results_test.csv"
figure_dir = ROOT / "figures"
figure_dir.mkdir(exist_ok=True)
if results.exists() and results.stat().st_size > 1:
    frame = pd.read_csv(results)
    frame.groupby("model")["roc_auc"].mean().plot.bar()
    plt.ylabel("Mean ROC-AUC")
    plt.tight_layout()
    plt.savefig(figure_dir / "model_roc_auc.png", dpi=200)
    plt.close()

