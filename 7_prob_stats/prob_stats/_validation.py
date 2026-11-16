"""
shared input checks, so every estimator states its shape contract once
and fails the same way when it's broken
"""

from __future__ import annotations
import numpy as np

_SUM_TO_ONE_ATOL = 1e-8

def as_1d_float(data: np.ndarray, name: str = "data") -> np.ndarray:
    """float64, 1-d, non-empty and finite, or I raise; shape (n,)"""
    arr = np.asarray(data, dtype=np.float64)
    
    if arr.ndim != 1:
        raise ValueError(
            f"{name} must be 1-D with shape (n,), got shape {arr.shape}"
            f"({arr.ndim} dimensions)"
        )
    
    if arr.size == 0:
        raise ValueError(
            f"{name} must contain at least one observation but it got (0,)"
        )
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValueError(
            f"{name} contains {n_bad} non-finite values(s) (nan or inf);"
            f"estimators require finite observations"
        )
    
    return np.ascontiguousarray(arr)


def as_2d_float(X: np.ndarray, name: str = "X") -> np.ndarray:
    """float64, 2-d, at least one row and column, finite; shape (n, d)"""
    arr = np.asarray(X, dtype=np.float64)
    
    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D with shape (n, d), got shape {arr.shape}. "
            f"For a single feature use X.reshape(-1, 1); for a single "
            f"observation use X.reshape(1, -1)"
        )
    n,d = arr.shape
    if n == 0 or d == 0:
        raise ValueError(
            f"{name} must have at least one row and one column, got "
            f"shape {arr.shape}"
        )
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.count_nonzero(~np.isfinite(arr)))
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s) (nan or inf)"
        )
    return np.ascontiguousarray(arr)

def as_distribution(p: np.ndarray, name: str = "p") -> np.ndarray:
    """a 1-d probability vector: non-negative and summing to 1; shape (k,)"""
    arr = as_1d_float(p, name=name)
    
    if np.any(arr<0.0):
        n_bad = int(np.count_nonzero(arr<0.0))
        raise ValueError(
            f"{name} contains {n_bad} negative value(s); probabilities must be >= 0"
        )
        
    total = float(arr.sum()) # scalar
    if not np.isclose(total, 1.0, rtol=0.0, atol=_SUM_TO_ONE_ATOL):
        raise ValueError(
            f"{name} must sum to 1.0 (within {_SUM_TO_ONE_ATOL:g}),"
            f"got {total!r}. Normalize before calling, or use bayes_update() if you intended an unnormalized score vector"
        )
    return arr

def as_distribution_rows(P: np.ndarray, name: str = "P") -> np.ndarray:
    """a batch of distributions, every row sums to 1; shape (n, k)"""
    arr = as_2d_float(P, name=name)

    if np.any(arr < 0.0):
        raise ValueError(f"{name} must be non-negative")

    row_sums = arr.sum(axis=1)  # shape: (n,)
    bad = ~np.isclose(row_sums, 1.0, rtol=0.0, atol=_SUM_TO_ONE_ATOL)
    if np.any(bad):
        idx = int(np.flatnonzero(bad)[0])
        raise ValueError(
            f"every row of {name} must sum to 1.0; row {idx} sums to "
            f"{row_sums[idx]!r} ({int(bad.sum())} bad row(s) total)"
        )
    return arr

def as_binary(data: np.ndarray, name: str = "data") -> np.ndarray:
    """1-d labels that are exactly 0 or 1; shape (n,)"""
    arr = as_1d_float(data, name=name)
    
    is_binary = (arr == 0.0) | (arr == 1.0) # shape: (n,), dtype bool
    if not np.all(is_binary):
        offenders = np.unique(arr[~is_binary])[:5] # shape: (<=5,) limits display to first 5 values for easy debugging
        raise ValueError(
            f"{name} must contain only 0 and 1, found other value(s): "
            f"{offenders.tolist()}"
        )
        
    return arr

# alias with a name that reads correctly at classification call points
as_labels = as_binary

def check_xy(
    X: np.ndarray, y: np.ndarray, name_x: str = "X", name_y: str = "y"
) -> tuple[np.ndarray, np.ndarray]:
    """X (n, d) and y (n,) that agree on n"""
    X_arr = as_2d_float(X, name=name_x)  # shape: (n, d)
    y_arr = as_1d_float(y, name=name_y)  # shape: (n,)

    if X_arr.shape[0] != y_arr.shape[0]:
        raise ValueError(
            f"{name_x} and {name_y} must agree on the number of "
            f"observations, got {X_arr.shape[0]} rows in {name_x} and "
            f"{y_arr.shape[0]} in {name_y}"
        )
    return X_arr, y_arr

