# Tox21 Phase 1

Reproducible multi-task toxicity prediction pipeline.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Run

Each stage has one Make target:

```bash
make clean-split
make features
make tune
make test
make figures
make test-code
```

The raw download is intentionally not committed. Place `tox21.csv.gz` in
`data/raw/` or run `python scripts/01_clean_and_split.py --download`.

## Key Pipeline Conclusions & Findings

### 1. Molecular Representation & Feature Synergy
- **Circular Morgan Fingerprints (ECFP4 Equivalent, 2048-bit, Radius 2):** Tox21 molecules exhibit an average active bit density of $29.75 \pm 16.41$ bits per molecule (median 26.0, range $[1, 160]$). All 2,048 bit positions are activated across the dataset (0 unused bits), while the top motif appears in $71.73\%$ of compounds.
- **RDKit 2D Molecular Descriptors (217 raw features):** Empirical analysis revealed 1,224 NaN cells and 6 infinite cells across 107 compounds (caused by undefined partial charges or zero-division topological indices). 3 descriptors are constant zero across all 7,709 molecules (`SMR_VSA8`, `SlogP_VSA9`, `fr_diazo`), and `fr_azide` is near-constant in scaffold training, leaving 213 retained descriptors.
- **Hybrid Modality (`morgan+desc`, 2,261 features):** Combining localized topological fragments (Morgan) with bulk physicochemical parameters (lipophilicity $\log P$, polar surface area, molecular weight) provides complementary inductive biases without expanding dimensionality beyond tabular model feasibility.

### 2. Preprocessing & Data Leakage Prevention
- **Strict Train-Only Parameter Estimation:** Imputation medians, extreme-value percentile clipping thresholds ($1.0\%$ and $99.9\%$), variance masks, means, and population standard deviations must be fitted strictly on the training partition ($n=6{,}167$). Applying these fitted parameters to validation and test partitions eliminates distributional and target leakage.
- **Outlier Normalization:** Percentile clipping effectively controls heavy-tailed graph complexity and surface-area descriptors spanning orders of magnitude without distorting core distributions.

### 3. Class Imbalance & Multi-Task Target Handling
- **Severe Class Imbalance:** Active toxic compounds comprise only $3.4\%$ to $15.5\%$ of tested samples across endpoints. Utilizing `class_weight="balanced_subsample"` within tree ensembles dynamically adjusts weights per bootstrap draw, preventing majority-class collapse without fabricating synthetic samples.
- **Sparse Multi-Endpoint Targets:** Tox21 molecules are tested against varying subsets of the 12 endpoints. Missing target labels are represented as `NaN` and must **never** be imputed as zeros or ones. Independent per-endpoint binary classifiers trained solely on observed labels achieve strong baseline discrimination (e.g. NR-AR-LBD ROC-AUC 0.830, SR-MMP ROC-AUC 0.755) while preserving metric validity via masked evaluation.

### 4. Reproducibility & Alignment Safeguards
- **Row-Order Verification:** Persistent feature matrices are cryptographically anchored to `data/processed/tox21_clean.csv` via SHA-256 row hashing (`0d406f18bc44ede1212741b1abcb1cb00982d8ea10a65a54363ad6a905a4d372`). The shared loader validates this hash prior to partition indexing, preventing silent row misalignments across team workflows.

## Results

`results/tuning_val.csv` stores validation scores for every configuration,
`results/best_configs.json` freezes the selected configurations, and
`results/results_test.csv` stores final endpoint-level metrics. Test metrics
must only be generated after tuning has frozen the configurations.