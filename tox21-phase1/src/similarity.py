"""Tanimoto similarity calculation for molecular fingerprints.

Implements the maximum Tanimoto similarity study comparing evaluation molecules
to the training molecule set, as required by the Phase 1 specification.
"""

from typing import Union
import numpy as np


def max_tanimoto(X_ev: np.ndarray, X_tr: np.ndarray) -> np.ndarray:
    """For each eval molecule, computes the highest Tanimoto similarity to any training molecule.

    Parameters
    ----------
    X_ev : np.ndarray
        Query/evaluation binary Morgan fingerprint array of shape (N_eval, 2048)
        or containing 2048 fingerprint bits in the first 2048 columns.
    X_tr : np.ndarray
        Training binary Morgan fingerprint array of shape (N_train, 2048)
        or containing 2048 fingerprint bits in the first 2048 columns.

    Returns
    -------
    np.ndarray
        1D float32 array of shape (N_eval,) where each element is the maximum
        Tanimoto similarity of that evaluation molecule to any training molecule.
    """
    # Use first 2048 columns if descriptors are concatenated
    if X_ev.shape[1] > 2048:
        X_ev = X_ev[:, :2048]
    if X_tr.shape[1] > 2048:
        X_tr = X_tr[:, :2048]

    A = X_ev.astype(np.float32)
    B = X_tr.astype(np.float32)

    # Vectorized computation of bit intersection and union
    # inter[i, j] = A[i] . B[j]
    inter = A @ B.T
    # union[i, j] = sum(A[i]) + sum(B[j]) - inter[i, j]
    union = A.sum(axis=1, keepdims=True) + B.sum(axis=1, keepdims=True).T - inter

    # Avoid division by zero for any edge cases (e.g., all-zero fingerprints)
    similarity_matrix = inter / np.maximum(union, 1.0)
    return similarity_matrix.max(axis=1).astype(np.float32)


def max_tanimoto_to_train(train_fingerprints: np.ndarray, query_fingerprints: np.ndarray) -> np.ndarray:
    """Compatibility wrapper matching alternative argument order (train, query).

    Parameters
    ----------
    train_fingerprints : np.ndarray
        Training fingerprint matrix (N_train, 2048).
    query_fingerprints : np.ndarray
        Query fingerprint matrix (N_query, 2048).

    Returns
    -------
    np.ndarray
        Array of shape (N_query,) with max Tanimoto similarities.
    """
    return max_tanimoto(query_fingerprints, train_fingerprints)
