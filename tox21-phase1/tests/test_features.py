"""Comprehensive automated tests for feature engineering, preprocessing, and Random Forest."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.config import CLEAN_PATH, FEATURES_PATH, SPLITS_PATH
from src.evaluate import run_model
from src.features import (
    DescriptorPreprocessor,
    build_descriptors,
    build_feature_manifest,
    build_morgan,
    compute_row_hash,
    load_features,
)
from src.models.rf import GRID, MultiTaskRandomForest, make_rf


# ============================================================================
# Deliverable A: Morgan Fingerprint Tests
# ============================================================================


def test_morgan_shape_and_dtype(tmp_path):
    """Verify shape, uint8 dtype, and binary nature of Morgan fingerprints."""
    smiles = ["C", "CC", "CCO", "c1ccccc1"]
    out_file = tmp_path / "morgan_test.npy"
    fps = build_morgan(smiles, output=out_file, radius=2, n_bits=2048)

    assert fps.shape == (4, 2048)
    assert fps.dtype == np.uint8
    assert np.all((fps == 0) | (fps == 1))
    assert np.all(np.isfinite(fps))
    assert out_file.exists()


def test_morgan_deterministic(tmp_path):
    """Verify that fingerprint generation is strictly deterministic."""
    smiles = ["CCOc1ccc2nc(S(N)(=O)=O)sc2c1", "CCN1C(=O)NC(c2ccccc2)C1=O"]
    fp1 = build_morgan(smiles, output=tmp_path / "fp1.npy")
    fp2 = build_morgan(smiles, output=tmp_path / "fp2.npy")
    np.testing.assert_array_equal(fp1, fp2)


def test_morgan_invalid_smiles_raises(tmp_path):
    """Verify that invalid or unparseable SMILES raise descriptive ValueError with row index."""
    with pytest.raises(ValueError, match="RDKit failed to parse SMILES at row index 1"):
        build_morgan(["CC", "INVALID_SMILES_STRING", "CCC"], output=tmp_path / "err.npy")


def test_morgan_empty_smiles_raises(tmp_path):
    """Verify that empty or whitespace-only SMILES raise ValueError."""
    with pytest.raises(ValueError, match="Invalid or empty SMILES at row index 0"):
        build_morgan(["   ", "CC"], output=tmp_path / "err.npy")


# ============================================================================
# Deliverable B: Descriptor Generation Tests
# ============================================================================


def test_descriptors_shape_dtype_and_names(tmp_path):
    """Verify descriptor calculation dimensions, float64 dtype, and names file."""
    smiles = ["C", "CC", "CCO"]
    desc_path = tmp_path / "desc_test.npy"
    values, names = build_descriptors(smiles, output=desc_path)

    assert values.shape == (3, 217)
    assert values.dtype == np.float64
    assert len(names) == 217
    assert (tmp_path / "descriptor_names.txt").exists()

    saved = np.load(desc_path)
    np.testing.assert_array_equal(values, saved)


def test_descriptors_preserve_row_alignment(tmp_path):
    """Verify that descriptors preserve molecule row count even with unparseable checks."""
    smiles = ["C", "c1ccccc1"]
    values, _ = build_descriptors(smiles, output=tmp_path / "desc.npy")
    assert values.shape[0] == 2


# ============================================================================
# Deliverable C: Descriptor Preprocessor Pipeline Tests (Synthetic Matrices)
# ============================================================================


@pytest.fixture
def synthetic_descriptor_data():
    """Create a synthetic dataset designed to test all 8 preprocessing steps.

    Columns:
    - Col 0: Normal feature with variance (mean ~ 10, std ~ 2)
    - Col 1: Feature with NaNs and infs (+/- inf should become NaN, then imputed with median)
    - Col 2: Completely missing feature (all NaNs in train) -> MUST be dropped
    - Col 3: Constant-zero feature (std = 0) -> MUST be dropped
    - Col 4: Near-constant feature (std <= 1e-8) -> MUST be dropped
    - Col 5: Feature with extreme outlier -> MUST be clipped to [1st, 99.9th] percentile
    - Col 6: Normal binary-like feature
    """
    rng = np.random.default_rng(42)
    n_train = 200
    n_test = 50

    # Training data
    c0 = rng.normal(10.0, 2.0, size=(n_train, 1))
    c1 = rng.normal(5.0, 1.0, size=(n_train, 1))
    c1[0] = np.nan
    c1[1] = np.inf
    c1[2] = -np.inf
    c2 = np.full((n_train, 1), np.nan)  # All missing
    c3 = np.zeros((n_train, 1))  # Constant zero
    c4 = np.ones((n_train, 1)) * 3.14159  # Near-constant
    c5 = rng.uniform(0.0, 10.0, size=(n_train, 1))
    c5[10] = 1_000_000.0  # Extreme upper outlier
    c5[11] = -1_000_000.0  # Extreme lower outlier
    c6 = rng.choice([0.0, 1.0], size=(n_train, 1))

    X_train = np.hstack([c0, c1, c2, c3, c4, c5, c6])

    # Test data
    t0 = rng.normal(10.0, 2.0, size=(n_test, 1))
    t1 = rng.normal(5.0, 1.0, size=(n_test, 1))
    t1[5] = np.nan
    t1[6] = np.inf
    t2 = np.full((n_test, 1), np.nan)
    t3 = np.zeros((n_test, 1))
    t4 = np.ones((n_test, 1)) * 3.14159
    t5 = rng.uniform(0.0, 10.0, size=(n_test, 1))
    t6 = rng.choice([0.0, 1.0], size=(n_test, 1))

    X_test = np.hstack([t0, t1, t2, t3, t4, t5, t6])

    feature_names = [f"feat_{i}" for i in range(7)]
    return X_train, X_test, feature_names


def test_preprocessor_synthetic_pipeline(synthetic_descriptor_data):
    """Test that all non-finite, all-missing, constant, and outlier conditions are handled."""
    X_train, X_test, feature_names = synthetic_descriptor_data

    preprocessor = DescriptorPreprocessor()
    X_train_trans = preprocessor.fit_transform(X_train, feature_names=feature_names)
    X_test_trans = preprocessor.transform(X_test)

    # 1. Output shapes: Col 2 (all missing), Col 3 (zero), Col 4 (near-constant) must be dropped
    # Retained features: Col 0, Col 1, Col 5, Col 6 -> 4 columns
    assert X_train_trans.shape == (200, 4)
    assert X_test_trans.shape == (50, 4)
    assert X_train_trans.dtype == np.float32
    assert X_test_trans.dtype == np.float32

    # 2. All outputs must be finite
    assert np.all(np.isfinite(X_train_trans))
    assert np.all(np.isfinite(X_test_trans))

    # 3. Transformed training data must have approx zero mean and unit variance
    np.testing.assert_allclose(np.mean(X_train_trans, axis=0), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.std(X_train_trans, axis=0), 1.0, atol=1e-4)

    # 4. Feature names out
    names_out = preprocessor.get_feature_names_out()
    assert names_out == ["feat_0", "feat_1", "feat_5", "feat_6"]


def test_preprocessor_no_data_leakage(synthetic_descriptor_data):
    """Verify that transforming test data does NOT alter fitted training statistics."""
    X_train, X_test, _ = synthetic_descriptor_data

    prep = DescriptorPreprocessor()
    prep.fit(X_train)

    train_medians = prep.medians_.copy()
    train_clip_lower = prep.clip_lower_.copy()
    train_clip_upper = prep.clip_upper_.copy()
    train_mean = prep.mean_.copy()
    train_scale = prep.scale_.copy()

    # Transform test set multiple times
    _ = prep.transform(X_test)
    _ = prep.transform(X_test)

    # Verify fitted parameters are completely unchanged
    np.testing.assert_array_equal(prep.medians_, train_medians)
    np.testing.assert_array_equal(prep.clip_lower_, train_clip_lower)
    np.testing.assert_array_equal(prep.clip_upper_, train_clip_upper)
    np.testing.assert_array_equal(prep.mean_, train_mean)
    np.testing.assert_array_equal(prep.scale_, train_scale)


def test_preprocessor_scikit_learn_pipeline_compatibility(synthetic_descriptor_data):
    """Verify that DescriptorPreprocessor works inside a scikit-learn Pipeline."""
    X_train, _, _ = synthetic_descriptor_data
    pipe = Pipeline([("prep", DescriptorPreprocessor())])
    out = pipe.fit_transform(X_train)
    assert out.shape == (200, 4)
    assert out.dtype == np.float32


def test_preprocessor_dimension_mismatch_raises():
    """Verify that transforming a matrix with wrong number of columns raises ValueError."""
    prep = DescriptorPreprocessor()
    rng = np.random.default_rng(0)
    prep.fit(rng.normal(size=(10, 5)))
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        prep.transform(rng.normal(size=(5, 3)))


# ============================================================================
# Deliverable D: Feature Loader Tests
# ============================================================================


def test_load_features_morgan():
    """Test loading morgan fingerprints for scaffold split."""
    X_train, X_val, X_test = load_features(kind="morgan", split="scaffold", seed=0)

    assert X_train.shape == (6167, 2048)
    assert X_val.shape == (771, 2048)
    assert X_test.shape == (771, 2048)
    assert X_train.dtype == np.float32
    assert X_val.dtype == np.float32
    assert X_test.dtype == np.float32
    assert np.all(np.isfinite(X_train))
    assert np.all(np.isfinite(X_val))
    assert np.all(np.isfinite(X_test))


def test_load_features_morgan_plus_desc():
    """Test loading morgan+desc with train-only preprocessing."""
    X_train, X_val, X_test = load_features(kind="morgan+desc", split="scaffold", seed=0)

    # 2048 Morgan bits + 213 retained descriptors = 2261
    assert X_train.shape == (6167, 2261)
    assert X_val.shape == (771, 2261)
    assert X_test.shape == (771, 2261)
    assert X_train.dtype == np.float32
    assert np.all(np.isfinite(X_train))
    assert np.all(np.isfinite(X_val))
    assert np.all(np.isfinite(X_test))


def test_load_features_invalid_kind():
    """Test that invalid feature kind raises ValueError."""
    with pytest.raises(ValueError, match="Unknown feature kind"):
        load_features(kind="invalid_kind", split="scaffold")


def test_load_features_missing_split():
    """Test that nonexistent split file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError, match="Split file not found"):
        load_features(kind="morgan", split="nonexistent_split_xyz")


# ============================================================================
# Deliverable E: Random Forest MultiTask Tests
# ============================================================================


def test_rf_multitask_fit_and_nan_handling():
    """Verify that MultiTaskRandomForest handles missing target labels correctly."""
    X = np.random.randn(40, 10)
    # 2 tasks with NaNs
    y = np.array(
        [[1.0, np.nan], [0.0, 1.0], [np.nan, 0.0], [1.0, 1.0]] * 10
    )

    clf = make_rf({"n_estimators": 5}, seed=42)
    clf.fit(X, y)

    assert len(clf.estimators_) == 2
    assert clf.estimators_[0] is not None
    assert clf.estimators_[1] is not None

    preds = clf.predict_proba(X)
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert preds[0].shape == (40, 2)
    assert preds[1].shape == (40, 2)


def test_rf_integration_with_evaluate():
    """Verify that MultiTaskRandomForest integrates with evaluate.run_model."""
    X_train = np.random.randn(60, 15)
    y_train = np.random.choice([0.0, 1.0, np.nan], size=(60, 3), p=[0.45, 0.45, 0.1])
    X_eval = np.random.randn(20, 15)
    y_eval = np.random.choice([0.0, 1.0, np.nan], size=(20, 3), p=[0.45, 0.45, 0.1])

    model = make_rf({"n_estimators": 5}, seed=1)
    metrics = run_model(model, X_train, y_train, X_eval, y_eval)

    assert len(metrics) == 3
    for m in metrics:
        assert "roc_auc" in m
        assert "pr_auc" in m
        assert "n" in m


def test_rf_grid_budget():
    """Verify that GRID contains exactly 8 parameter configurations."""
    assert len(GRID) == 8
    for cfg in GRID:
        assert "n_estimators" in cfg
        assert "max_depth" in cfg
        assert "min_samples_leaf" in cfg
