# Phase 2 report handoff

`sections/member5.tex` is Malhar's report fragment. Include it from the assembled Phase 2 IEEE report with the working directory set to `report_phase2/`. The fragment uses `graphicx` and the generated tables under `tables/`.

Regenerate its data products from the repository root:

```bash
python scripts/07_tanimoto_analysis.py
python scripts/08_member5_validation.py --splits scaffold,random
```

The Tanimoto analysis uses the frozen scaffold test predictions for XGBoost, graph MPNN, and H1 (MPNN + descriptors). It writes the PDF figure, endpoint-level CSV, summary CSV, and a LaTeX table. The validation script writes prediction metrics, alignment checks, seed-variance CSV, and the corresponding LaTeX table.

The predeclared architecture ablations and shuffled-label control are generated and run with:

```bash
python scripts/09_run_malhar_ablations.py --dry-run
python scripts/09_run_malhar_ablations.py
```

The second command requires the Phase 2 dependencies (`torch` and `torch_geometric`) and writes the experiment results to `results/dl_ablation_test_malhar.csv`. The committed file contains seven predeclared conditions, three seeds per condition, and 12 endpoint rows per seed. Report values are computed by averaging endpoints within each seed and then reporting the mean and sample standard deviation across seeds. The shuffled-label control must be reported with its seed-level ROC AUCs; seed 2 exceeds the 0.55 review threshold in this run.
