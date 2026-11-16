"""
objectives for the two models, each returning (loss, gradient)
both are means over the batch plus l2 * ||theta||^2, with the intercept left unpenalized
"""

from __future__ import annotations

import numpy as np

from prob_stats._validation import as_1d_float, as_2d_float

__all__ = [
    "sigmoid",
    "augment",
    "penalty_mask_for",
    "mse_objective",
    "logistic_objective",
    "mean_squared_error",
    "binary_cross_entropy_from_logits",
]


def sigmoid(z: np.ndarray) -> np.ndarray:
    """
    1 / (1 + e^-z), branched so exp never sees a large positive number
    the naive form overflows near z = -710
    """
    z_arr = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z_arr)  # shape: same as z

    pos = z_arr >= 0.0  # shape: same as z, bool
    out[pos] = 1.0 / (1.0 + np.exp(-z_arr[pos]))  # z >= 0, so e^-z <= 1

    exp_z = np.exp(z_arr[~pos])  # all arguments <= 0, cannot overflow
    out[~pos] = exp_z / (1.0 + exp_z)

    return out


def augment(X: np.ndarray) -> np.ndarray:
    """prepend a column of ones for the intercept: (n, d) -> (n, d + 1)"""
    X_arr = as_2d_float(X, name="X")  # shape: (n, d)
    n = X_arr.shape[0]
    ones = np.ones((n, 1), dtype=np.float64)  # shape: (n, 1)
    return np.hstack([ones, X_arr])  # shape: (n, d + 1)


def penalty_mask_for(n_params: int, fit_intercept: bool) -> np.ndarray:
    """1 for every parameter the penalty touches, 0 for the intercept; shape (p,)"""
    if n_params < 1:
        raise ValueError(f"n_params must be >= 1, got {n_params}")

    mask = np.ones(n_params, dtype=np.float64)  # shape: (p,)
    if fit_intercept:
        mask[0] = 0.0
    return mask


def mse_objective(
    theta: np.ndarray,
    X_aug: np.ndarray,
    y: np.ndarray,
    l2: float,
    penalty_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    mean squared error + l2 penalty, and its gradient
    loss = mean((X theta - y)^2) + l2 * sum(mask * theta^2)
    grad = (2 / n) X^T (X theta - y) + 2 l2 (mask * theta)
    """
    theta_arr = as_1d_float(theta, name="theta")  # shape: (p,)
    X = as_2d_float(X_aug, name="X_aug")  # shape: (n, p)
    y_arr = as_1d_float(y, name="y")  # shape: (n,)
    mask = as_1d_float(penalty_mask, name="penalty_mask")  # shape: (p,)

    n, p = X.shape
    if theta_arr.shape[0] != p:
        raise ValueError(
            f"theta has shape {theta_arr.shape} but X_aug has {p} columns"
        )
    if y_arr.shape[0] != n:
        raise ValueError(
            f"y has shape {y_arr.shape} but X_aug has {n} rows"
        )
    if mask.shape[0] != p:
        raise ValueError(
            f"penalty_mask has shape {mask.shape}, expected ({p},)"
        )
    if l2 < 0.0:
        raise ValueError(f"l2 must be >= 0, got {l2}")

    y_pred = X @ theta_arr  # shape: (n,)
    residual = y_pred - y_arr  # shape: (n,)

    data_loss = float((residual @ residual) / n)  # scalar
    penalty = float(l2 * np.sum(mask * theta_arr**2))  # scalar

    grad_data = (2.0 / n) * (X.T @ residual)  # shape: (p,), d/dtheta of the mean squared residual
    grad_pen = 2.0 * l2 * mask * theta_arr  # shape: (p,)

    return data_loss + penalty, grad_data + grad_pen


def logistic_objective(
    theta: np.ndarray,
    X_aug: np.ndarray,
    y: np.ndarray,
    l2: float,
    penalty_mask: np.ndarray,
) -> tuple[float, np.ndarray]:
    """
    mean binary cross-entropy from logits + l2 penalty, and its gradient
    loss = mean(softplus(z) - y z) + l2 * sum(mask * theta^2), with z = X theta
    grad = X^T (sigmoid(z) - y) / n + 2 l2 (mask * theta)
    """
    theta_arr = as_1d_float(theta, name="theta")  # shape: (p,)
    X = as_2d_float(X_aug, name="X_aug")  # shape: (n, p)
    y_arr = as_1d_float(y, name="y")  # shape: (n,)
    mask = as_1d_float(penalty_mask, name="penalty_mask")  # shape: (p,)

    n, p = X.shape
    if theta_arr.shape[0] != p:
        raise ValueError(
            f"theta has shape {theta_arr.shape} but X_aug has {p} columns"
        )
    if y_arr.shape[0] != n:
        raise ValueError(
            f"y has shape {y_arr.shape} but X_aug has {n} rows"
        )
    if mask.shape[0] != p:
        raise ValueError(
            f"penalty_mask has shape {mask.shape}, expected ({p},)"
        )
    if l2 < 0.0:
        raise ValueError(f"l2 must be >= 0, got {l2}")
    if not np.all((y_arr == 0.0) | (y_arr == 1.0)):
        raise ValueError(
            "y must contain only 0 and 1 for the logistic objective"
        )

    z = X @ theta_arr  # shape: (n,)

    # softplus(z) - y*z, stable for any finite z
    per_example = np.logaddexp(0.0, z) - y_arr * z  # shape: (n,)
    data_loss = float(per_example.mean())  # scalar
    penalty = float(l2 * np.sum(mask * theta_arr**2))  # scalar

    probs = sigmoid(z)  # shape: (n,)
    resid = probs - y_arr  # shape: (n,), prediction minus target, same shape as day 6

    grad_data = (X.T @ resid) / n  # shape: (p,)
    grad_pen = 2.0 * l2 * mask * theta_arr  # shape: (p,)

    return data_loss + penalty, grad_data + grad_pen


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """mean of (y_pred - y_true)^2, no penalty"""
    yt = as_1d_float(y_true, name="y_true")  # shape: (n,)
    yp = as_1d_float(y_pred, name="y_pred")  # shape: (n,)

    if yt.shape != yp.shape:
        raise ValueError(
            f"y_true and y_pred must have the same shape, got {yt.shape} "
            f"and {yp.shape}"
        )

    diff = yp - yt  # shape: (n,)
    return float((diff @ diff) / yt.shape[0])


def binary_cross_entropy_from_logits(
    y_true: np.ndarray, logits: np.ndarray, reduction: str = "mean"
) -> float | np.ndarray:
    """
    cross-entropy straight from logits, using softplus(z) - y z
    going through probabilities first would give log(0) = -inf on saturated predictions
    """
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(
            f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}"
        )

    yt = as_1d_float(y_true, name="y_true")  # shape: (n,)
    z = as_1d_float(logits, name="logits")  # shape: (n,)

    if yt.shape != z.shape:
        raise ValueError(
            f"y_true and logits must have the same shape, got {yt.shape} "
            f"and {z.shape}"
        )
    if not np.all((yt == 0.0) | (yt == 1.0)):
        raise ValueError("y_true must contain only 0 and 1")

    per_example = np.logaddexp(0.0, z) - yt * z  # shape: (n,)

    if reduction == "none":
        return per_example
    if reduction == "sum":
        return float(per_example.sum())
    return float(per_example.mean())
