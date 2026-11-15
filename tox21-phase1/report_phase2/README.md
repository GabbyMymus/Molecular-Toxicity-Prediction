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

The second command requires the Phase 2 dependencies (`torch` and `torch_geometric`) and writes the experiment results to `results/dl_ablation_test_malhar.csv`. No ablation outcome should be described until those runs finish.

H1 scaffold predictions and the corresponding result CSV came from `ranajeet/phase2` at commit `c10339827b7f509c42ed0d9f62d0f570a0b520d1`. See `results/h1_artifact_provenance.json` for source details and blob checksums.
