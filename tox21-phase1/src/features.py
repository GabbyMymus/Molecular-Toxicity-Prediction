"""Molecular fingerprint and descriptor generation, preprocessing, and loading."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from .config import CLEAN_PATH, FEATURES_PATH, N_BITS, ROOT, SPLITS_PATH

logger = logging.getLogger(__name__)


def compute_row_hash(
    smiles: Sequence[str],
    mol_ids: Sequence[str] | None = None,
) -> str:
    """Compute a deterministic SHA-256 hash of ordered (mol_id, smiles) pairs."""
    if mol_ids is None:
        lines = [str(s) for s in smiles]
    else:
        lines = [f"{m},{s}" for m, s in zip(mol_ids, smiles)]
    content = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    """Save a NumPy array atomically via a temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(".tmp.npy")
    np.save(temp_path, array)
    temp_path.replace(path)


def build_morgan(
    smiles: Sequence[str],
    output: Path = FEATURES_PATH / "morgan_2048.npy",
    radius: int = 2,
    n_bits: int = N_BITS,
) -> np.ndarray:
    """Generate circular Morgan fingerprints using RDKit's modern generator API.

    Parameters
    ----------
    smiles : Sequence[str]
        Ordered sequence of canonical SMILES strings.
    output : Path
        Destination path for the uint8 NumPy binary matrix.
    radius : int
        Morgan fingerprint radius (default 2).
    n_bits : int
        Number of bits in the bit vector (default 2048).

    Returns
    -------
    np.ndarray
        Array of shape (len(smiles), n_bits) with dtype uint8.
    """
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows: list[np.ndarray] = []

    for idx, smile in enumerate(smiles):
        if not isinstance(smile, str) or not smile.strip():
            raise ValueError(f"Invalid or empty SMILES at row index {idx}: {smile!r}")
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            raise ValueError(f"RDKit failed to parse SMILES at row index {idx}: {smile!r}")
        fp = gen.GetFingerprintAsNumPy(molecule)
        rows.append(fp)

    values = np.asarray(rows, dtype=np.uint8)

    # Validation
    if values.shape != (len(smiles), n_bits):
        raise ValueError(
            f"Shape mismatch in Morgan generation: expected ({len(smiles)}, {n_bits}), "
            f"got {values.shape}"
        )
    if values.dtype != np.uint8:
        raise ValueError(f"Expected uint8 dtype, got {values.dtype}")
    if not np.all((values == 0) | (values == 1)):
        raise ValueError("Morgan fingerprint array contains non-binary values")
    if not np.all(np.isfinite(values)):
        raise ValueError("Morgan fingerprint array contains non-finite values")

    _atomic_save_npy(output, values)
    return values


def build_descriptors(
    smiles: Sequence[str],
    output: Path = FEATURES_PATH / "descriptors_raw.npy",
) -> tuple[np.ndarray, list[str]]:
    """Compute all 2D molecular descriptors in RDKit in deterministic order.

    Preserves raw NaN and infinite values without early dropping or early imputation.

    Parameters
    ----------
    smiles : Sequence[str]
        Ordered sequence of canonical SMILES strings.
    output : Path
        Destination path for the float64 NumPy matrix.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        Matrix of raw descriptors (shape (len(smiles), 217), dtype float64)
        and list of descriptor names.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    descriptor_names = [name for name, _ in Descriptors.descList]
    rows: list[list[float]] = []

    for idx, smile in enumerate(smiles):
        if not isinstance(smile, str) or not smile.strip():
            raise ValueError(f"Invalid or empty SMILES at row index {idx}: {smile!r}")
        molecule = Chem.MolFromSmiles(smile)
        if molecule is None:
            raise ValueError(f"RDKit failed to parse SMILES at row index {idx}: {smile!r}")

        row: list[float] = []
        for name, fn in Descriptors.descList:
            try:
                val = float(fn(molecule))
            except Exception:
                val = float("nan")
            row.append(val)
        rows.append(row)

    values = np.asarray(rows, dtype=np.float64)

    # Validation
    if values.shape != (len(smiles), len(descriptor_names)):
        raise ValueError(
            f"Shape mismatch in descriptor generation: expected ({len(smiles)}, {len(descriptor_names)}), "
            f"got {values.shape}"
        )
    if values.dtype != np.float64:
        raise ValueError(f"Expected float64 dtype, got {values.dtype}")

    _atomic_save_npy(output, values)

    # Save descriptor names atomically
    names_path = output.parent / "descriptor_names.txt"
    temp_names = names_path.with_suffix(".tmp.txt")
    temp_names.write_text("\n".join(descriptor_names) + "\n", encoding="utf-8")
    temp_names.replace(names_path)

    return values, descriptor_names


class DescriptorPreprocessor(BaseEstimator, TransformerMixin):
    """Reusable, train-only fitted preprocessor for raw RDKit molecular descriptors.

    Transformation pipeline:
    1. Work on a copy of the input data.
    2. Normalize non-finite values (+inf, -inf -> NaN).
    3. Identify and drop columns that are entirely missing (all NaN) in training data.
    4. Impute missing values using training-set medians.
    5. Handle extreme values by clipping to the training-derived 1st and 99.9th percentiles.
    6. Identify and drop zero-variance or near-constant columns (std <= 1e-8 on training data).
    7. Standardize features using training-set mean and population standard deviation.
    8. Validate that all transformed values are strictly finite.
    """

    def __init__(
        self,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.9,
        variance_threshold: float = 1e-8,
    ) -> None:
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.variance_threshold = variance_threshold

        # Fitted attributes
        self.n_features_in_: int = 0
        self.feature_names_in_: list[str] | None = None
        self.retained_non_missing_mask_: np.ndarray | None = None
        self.medians_: np.ndarray | None = None
        self.clip_lower_: np.ndarray | None = None
        self.clip_upper_: np.ndarray | None = None
        self.variance_mask_: np.ndarray | None = None
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None
        self.retained_indices_: np.ndarray | None = None
        self.feature_names_out_: list[str] | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
        feature_names: Sequence[str] | None = None,
    ) -> DescriptorPreprocessor:
        """Fit preprocessing statistics strictly on the provided training partition.

        Parameters
        ----------
        X : np.ndarray
            Raw training descriptor matrix (2D array, shape (n_samples, n_features)).
        y : None
            Ignored; present for scikit-learn API compatibility.
        feature_names : Sequence[str] | None
            Optional descriptor column names corresponding to input features.

        Returns
        -------
        DescriptorPreprocessor
            Fitted instance.
        """
        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_arr.shape}")
        if X_arr.shape[0] == 0 or X_arr.shape[1] == 0:
            raise ValueError(f"Cannot fit on empty matrix with shape {X_arr.shape}")

        self.n_features_in_ = X_arr.shape[1]
        if feature_names is not None:
            if len(feature_names) != self.n_features_in_:
                raise ValueError(
                    f"Length of feature_names ({len(feature_names)}) does not match "
                    f"X columns ({self.n_features_in_})"
                )
            self.feature_names_in_ = list(feature_names)
        else:
            self.feature_names_in_ = [f"desc_{i}" for i in range(self.n_features_in_)]

        # Step 1: Work on a copy
        X_work = X_arr.copy()

        # Step 2: Normalize non-finite values to NaN
        X_work[~np.isfinite(X_work)] = np.nan

        # Step 3: Remove all-missing columns in training partition
        all_missing = np.all(np.isnan(X_work), axis=0)
        self.retained_non_missing_mask_ = ~all_missing

        if not np.any(self.retained_non_missing_mask_):
            raise ValueError("All descriptor columns are entirely missing in the training data.")

        X_work = X_work[:, self.retained_non_missing_mask_]
        intermediate_names = [
            name
            for name, keep in zip(self.feature_names_in_, self.retained_non_missing_mask_)
            if keep
        ]

        # Step 4: Median imputation (computed from training data only)
        self.medians_ = np.nanmedian(X_work, axis=0)
        # Fallback if any column still has nan median (e.g. edge case): replace with 0.0
        self.medians_ = np.where(np.isnan(self.medians_), 0.0, self.medians_)

        # Impute training copy
        nan_inds = np.where(np.isnan(X_work))
        X_work[nan_inds] = np.take(self.medians_, nan_inds[1])

        # Step 5: Extreme-value handling via percentile clipping
        self.clip_lower_ = np.percentile(X_work, self.lower_percentile, axis=0)
        self.clip_upper_ = np.percentile(X_work, self.upper_percentile, axis=0)

        # Ensure clip_lower <= clip_upper
        swap_mask = self.clip_lower_ > self.clip_upper_
        if np.any(swap_mask):
            self.clip_lower_[swap_mask], self.clip_upper_[swap_mask] = (
                self.clip_upper_[swap_mask],
                self.clip_lower_[swap_mask],
            )

        X_work = np.clip(X_work, self.clip_lower_, self.clip_upper_)

        # Step 6: Remove zero-variance or near-constant features (std <= threshold)
        stds = np.std(X_work, axis=0, ddof=0)
        self.variance_mask_ = stds > self.variance_threshold

        if not np.any(self.variance_mask_):
            raise ValueError("All descriptor features have zero or near-constant variance.")

        X_work = X_work[:, self.variance_mask_]

        # Final feature names out
        self.feature_names_out_ = [
            name
            for name, keep in zip(intermediate_names, self.variance_mask_)
            if keep
        ]

        # Record original column indices retained
        original_indices = np.where(self.retained_non_missing_mask_)[0]
        self.retained_indices_ = original_indices[self.variance_mask_]

        # Step 7: Standardize descriptors (mean and population std)
        self.mean_ = np.mean(X_work, axis=0)
        self.scale_ = np.std(X_work, axis=0, ddof=0)

        # Scale protection against division by zero
        zero_scale = self.scale_ <= self.variance_threshold
        if np.any(zero_scale):
            self.scale_[zero_scale] = 1.0

        # Step 8: Validate on training transformation
        z_train = (X_work - self.mean_) / self.scale_
        if not np.all(np.isfinite(z_train)):
            raise ValueError("Transformed training data contains non-finite values.")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform descriptor matrix using exclusively pre-computed training statistics.

        Parameters
        ----------
        X : np.ndarray
            Descriptor matrix (shape (n_samples, n_features_in_)).

        Returns
        -------
        np.ndarray
            Cleaned, imputed, clipped, filtered, and standardized matrix
            of shape (n_samples, n_features_out_), dtype float32.
        """
        if self.medians_ is None or self.mean_ is None or self.scale_ is None:
            raise RuntimeError("DescriptorPreprocessor must be fitted before calling transform().")

        X_arr = np.asarray(X, dtype=np.float64)
        if X_arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {X_arr.shape}")
        if X_arr.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Feature dimension mismatch: fitted on {self.n_features_in_} features, "
                f"received {X_arr.shape[1]}"
            )

        # Step 1: Work on a copy
        X_work = X_arr.copy()

        # Step 2: Normalize non-finite values to NaN
        X_work[~np.isfinite(X_work)] = np.nan

        # Step 3: Apply non-missing column filter
        X_work = X_work[:, self.retained_non_missing_mask_]

        # Step 4: Impute missing values with training medians
        nan_inds = np.where(np.isnan(X_work))
        X_work[nan_inds] = np.take(self.medians_, nan_inds[1])

        # Step 5: Clip using training percentiles
        X_work = np.clip(X_work, self.clip_lower_, self.clip_upper_)

        # Step 6: Filter constant columns using training variance mask
        X_work = X_work[:, self.variance_mask_]

        # Step 7: Standardize using training mean and scale
        z = (X_work - self.mean_) / self.scale_

        # Step 8: Validate output finiteness
        if not np.all(np.isfinite(z)):
            raise ValueError("Transformed descriptor matrix contains non-finite values.")

        return z.astype(np.float32)

    def get_feature_names_out(self, input_features: Sequence[str] | None = None) -> list[str]:
        """Return the names of retained features after preprocessing."""
        if self.feature_names_out_ is None:
            raise RuntimeError("Estimator not fitted; feature names out are unavailable.")
        return list(self.feature_names_out_)


def build_feature_manifest(
    manifest_path: Path = FEATURES_PATH / "feature_manifest.json",
    clean_csv_path: Path = CLEAN_PATH,
) -> dict:
    """Generate or update the comprehensive features manifest."""
    import importlib.metadata
    from rdkit import Chem, __version__ as rdkit_version

    df = pd.read_csv(clean_csv_path)
    smiles = df["smiles"].tolist()
    mol_ids = df["mol_id"].tolist()
    row_hash = compute_row_hash(smiles, mol_ids)

    # Load artifacts if present
    morgan_path = FEATURES_PATH / "morgan_2048.npy"
    desc_path = FEATURES_PATH / "descriptors_raw.npy"
    desc_names_path = FEATURES_PATH / "descriptor_names.txt"

    morgan_shape = list(np.load(morgan_path, mmap_mode="r").shape) if morgan_path.exists() else None
    desc_shape = list(np.load(desc_path, mmap_mode="r").shape) if desc_path.exists() else None

    descriptor_names = (
        desc_names_path.read_text(encoding="utf-8").splitlines()
        if desc_names_path.exists()
        else []
    )

    # Calculate descriptor diagnostics if raw file exists
    nan_cells = 0
    inf_cells = 0
    non_finite_rows = 0
    constant_zero_cols: list[str] = []

    if desc_path.exists() and desc_shape is not None:
        raw_desc = np.load(desc_path)
        nan_cells = int(np.isnan(raw_desc).sum())
        inf_cells = int(np.isinf(raw_desc).sum())
        non_finite_rows = int(np.any(~np.isfinite(raw_desc), axis=1).sum())
        stds = np.nanstd(raw_desc, axis=0)
        constant_zero_cols = [
            descriptor_names[i]
            for i in range(len(descriptor_names))
            if stds[i] <= 1e-8
        ]

    def _rel(p: Path) -> str:
        try:
            return str(p.relative_to(ROOT))
        except ValueError:
            return str(p)

    manifest = {
        "dataset": {
            "name": "Tox21",
            "clean_csv_path": _rel(clean_csv_path),
            "n_molecules": len(df),
            "row_hash_sha256": row_hash,
        },
        "environment": {
            "rdkit_version": rdkit_version,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": importlib.metadata.version("scikit-learn"),
        },
        "fingerprints": {
            "type": "Morgan",
            "radius": 2,
            "n_bits": N_BITS,
            "representation": "binary",
            "dtype": "uint8",
            "artifact_path": _rel(morgan_path),
            "shape": morgan_shape,
        },
        "descriptors": {
            "source": "rdkit.Chem.Descriptors.descList",
            "n_raw_descriptors": len(descriptor_names),
            "dtype": "float64",
            "artifact_path": _rel(desc_path),
            "shape": desc_shape,
            "nan_cells": nan_cells,
            "inf_cells": inf_cells,
            "non_finite_rows": non_finite_rows,
            "constant_zero_descriptors": constant_zero_cols,
        },
        "preprocessing_pipeline": {
            "steps": [
                "1. Copy input matrix",
                "2. Normalize non-finite values (+/- inf -> NaN)",
                "3. Remove all-missing columns in training partition",
                "4. Median imputation using training partition medians",
                "5. Percentile clipping (1.0% to 99.9%) fitted on training partition",
                "6. Remove zero/near-constant features (std <= 1e-8 on training partition)",
                "7. Standardize with training partition mean and population standard deviation",
                "8. Validate finiteness of float32 output",
            ]
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_manifest = manifest_path.with_suffix(".tmp.json")
    temp_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temp_manifest.replace(manifest_path)

    return manifest


def load_features(
    kind: str = "morgan",
    split: str | Path = "scaffold",
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Shared feature loader returning (X_train, X_val, X_test).

    Rows strictly follow the corresponding saved split-index order.
    Outputs are guaranteed to be finite float32 arrays.

    Parameters
    ----------
    kind : str
        Feature set to load: 'morgan', 'morgan+desc', or 'desc'.
    split : str | Path
        Split specification: 'scaffold', 'random', a split file stem, or Path to split JSON.
    seed : int
        Split seed if split is specified as a name (default 0).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (X_train, X_val, X_test) aligned in float32.
    """
    valid_kinds = ("morgan", "morgan+desc", "desc", "descriptors")
    if kind not in valid_kinds:
        raise ValueError(f"Unknown feature kind '{kind}'. Valid options: {valid_kinds}")

    # Resolve split path
    if isinstance(split, Path):
        split_path = split
    else:
        split_str = str(split)
        if split_str.endswith(".json"):
            split_path = SPLITS_PATH / split_str
        elif "_seed" in split_str:
            split_path = SPLITS_PATH / f"{split_str}.json"
        else:
            split_path = SPLITS_PATH / f"{split_str}_seed{seed}.json"

    if not split_path.exists():
        raise FileNotFoundError(f"Split file not found: {split_path}")

    with open(split_path, "r", encoding="utf-8") as f:
        split_dict = json.load(f)

    for key in ("train", "val", "test"):
        if key not in split_dict:
            raise ValueError(f"Split file {split_path} missing required partition '{key}'")

    train_idx = np.asarray(split_dict["train"], dtype=int)
    val_idx = np.asarray(split_dict["val"], dtype=int)
    test_idx = np.asarray(split_dict["test"], dtype=int)

    # Alignment and metadata check
    manifest_path = FEATURES_PATH / "feature_manifest.json"
    if manifest_path.exists() and CLEAN_PATH.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        stored_hash = manifest.get("dataset", {}).get("row_hash_sha256")
        if stored_hash:
            df = pd.read_csv(CLEAN_PATH)
            current_hash = compute_row_hash(df["smiles"], df["mol_id"])
            if current_hash != stored_hash:
                raise ValueError(
                    f"Row alignment mismatch: current dataset hash {current_hash} "
                    f"does not match feature manifest hash {stored_hash}."
                )

    total_idx_count = len(train_idx) + len(val_idx) + len(test_idx)

    # Load Morgan fingerprints
    morgan_path = FEATURES_PATH / "morgan_2048.npy"
    if not morgan_path.exists():
        raise FileNotFoundError(
            f"Morgan fingerprint artifact not found at {morgan_path}. "
            f"Run `python scripts/02_build_features.py` or build_morgan() first."
        )

    morgan_raw = np.load(morgan_path)
    n_rows = morgan_raw.shape[0]

    # Validate split index bounds
    for name, indices in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        if np.any(indices < 0) or np.any(indices >= n_rows):
            raise IndexError(f"{name} split indices are out of bounds for dataset of size {n_rows}")

    if kind == "morgan":
        X_train = morgan_raw[train_idx].astype(np.float32)
        X_val = morgan_raw[val_idx].astype(np.float32)
        X_test = morgan_raw[test_idx].astype(np.float32)

    elif kind in ("morgan+desc", "desc", "descriptors"):
        desc_path = FEATURES_PATH / "descriptors_raw.npy"
        names_path = FEATURES_PATH / "descriptor_names.txt"
        if not desc_path.exists():
            raise FileNotFoundError(
                f"Raw descriptors artifact not found at {desc_path}. "
                f"Run `python scripts/02_build_features.py` or build_descriptors() first."
            )

        desc_raw = np.load(desc_path)
        if desc_raw.shape[0] != n_rows:
            raise ValueError(
                f"Row count mismatch between Morgan ({n_rows}) and descriptors ({desc_raw.shape[0]})"
            )

        descriptor_names = (
            names_path.read_text(encoding="utf-8").splitlines()
            if names_path.exists()
            else None
        )

        # Fit preprocessor strictly on training descriptors
        preprocessor = DescriptorPreprocessor()
        preprocessor.fit(desc_raw[train_idx], feature_names=descriptor_names)

        # Transform train, val, and test partitions
        desc_train = preprocessor.transform(desc_raw[train_idx])
        desc_val = preprocessor.transform(desc_raw[val_idx])
        desc_test = preprocessor.transform(desc_raw[test_idx])

        if kind in ("desc", "descriptors"):
            X_train = desc_train
            X_val = desc_val
            X_test = desc_test
        else:
            # morgan+desc
            morgan_train = morgan_raw[train_idx].astype(np.float32)
            morgan_val = morgan_raw[val_idx].astype(np.float32)
            morgan_test = morgan_raw[test_idx].astype(np.float32)

            X_train = np.hstack([morgan_train, desc_train])
            X_val = np.hstack([morgan_val, desc_val])
            X_test = np.hstack([morgan_test, desc_test])

    # Validate output contract
    for name, arr, idx in (("X_train", X_train, train_idx), ("X_val", X_val, val_idx), ("X_test", X_test, test_idx)):
        if arr.shape[0] != len(idx):
            raise ValueError(f"{name} row count {arr.shape[0]} does not match split length {len(idx)}")
        if arr.dtype != np.float32:
            raise ValueError(f"{name} dtype must be float32, got {arr.dtype}")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")

    if X_train.shape[1] != X_val.shape[1] or X_train.shape[1] != X_test.shape[1]:
        raise ValueError(
            f"Feature dimension mismatch across partitions: "
            f"train={X_train.shape[1]}, val={X_val.shape[1]}, test={X_test.shape[1]}"
        )

    return X_train, X_val, X_test
