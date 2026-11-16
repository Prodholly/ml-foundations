"""regression and classification metrics, written out so I know exactly what each one measures"""

from __future__ import annotations

import numpy as np

from prob_stats._validation import as_1d_float, as_labels
from prob_stats.losses import binary_cross_entropy_from_logits

__all__ = [
    "rmse",
    "r2_score",
    "accuracy",
    "log_loss",
    "brier_score",
    "roc_auc",
    "expected_calibration_error",
    "reliability_curve",
    "_average_ranks",
]


def _check_pair(
    a: np.ndarray, b: np.ndarray, name_a: str, name_b: str
) -> tuple[np.ndarray, np.ndarray]:
    """two 1-d arrays of the same length; shapes (n,), (n,)"""
    arr_a = as_1d_float(a, name=name_a)
    arr_b = as_1d_float(b, name=name_b)
    if arr_a.shape != arr_b.shape:
        raise ValueError(
            f"{name_a} and {name_b} must have the same shape, got "
            f"{arr_a.shape} and {arr_b.shape}"
        )
    return arr_a, arr_b


# ------
# regression
# ---------
def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """root mean squared error, in the units of y, which is why I report it over MSE"""
    yt, yp = _check_pair(y_true, y_pred, "y_true", "y_pred")
    diff = yp - yt  # shape: (n,)
    return float(np.sqrt((diff @ diff) / yt.shape[0]))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    1 - SS_res / SS_tot
    can go negative on held-out data, which means worse than predicting the mean, not a bug
    I raise when y_true is constant instead of making up a 0 or 1
    """
    yt, yp = _check_pair(y_true, y_pred, "y_true", "y_pred")

    resid = yt - yp  # shape: (n,)
    ss_res = float(resid @ resid)  # scalar

    centered = yt - yt.mean()  # shape: (n,)
    ss_tot = float(centered @ centered)  # scalar

    if ss_tot == 0.0:
        raise ValueError(
            "R^2 is undefined when y_true has zero variance (SS_tot = 0); "
            "use rmse() instead"
        )
    return float(1.0 - ss_res / ss_tot)


# -------------
# classification
# -----
def accuracy(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> float:
    """fraction of correct hard predictions after thresholding P(y = 1)"""
    yt, yp = _check_pair(y_true, y_prob, "y_true", "y_prob")
    yt = as_labels(yt, name="y_true")
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"threshold must be in (0, 1), got {threshold}")

    pred = (yp >= threshold).astype(np.float64)  # shape: (n,)
    return float(np.mean(pred == yt))


def log_loss(y_true: np.ndarray, logits: np.ndarray) -> float:
    """mean binary cross-entropy, computed from logits so it stays finite"""
    return float(
        binary_cross_entropy_from_logits(y_true, logits, reduction="mean")
    )


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    mean squared error of the predicted probabilities, in [0, 1]
    punishes confident mistakes less than log loss, so a few saturated predictions can't dominate it
    """
    yt, yp = _check_pair(y_true, y_prob, "y_true", "y_prob")
    yt = as_labels(yt, name="y_true")
    diff = yp - yt  # shape: (n,)
    return float((diff @ diff) / yt.shape[0])


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """
    1-based ranks, ties get their group's average rank
    written out to avoid a scipy dependency, and because ties are where AUC goes wrong
    """
    vals = as_1d_float(values, name="values")  # shape: (n,)
    n = vals.shape[0]

    order = np.argsort(vals, kind="stable")  # shape: (n,)
    sorted_vals = vals[order]  # shape: (n,)
    ranks = np.empty(n, dtype=np.float64)  # shape: (n,)

    i = 0
    while i < n:
        j = i
        while j + 1 < n and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        # positions i..j are tied; give each the average 1-based rank
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1

    return ranks


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    P(random positive scores above random negative), ties counted as half
    via Mann-Whitney U: (R_pos - n_pos (n_pos + 1) / 2) / (n_pos n_neg), exact and O(n log n)
    only the ordering matters, so AUC can't see miscalibration; I never report it alone
    """
    yt, ys = _check_pair(y_true, y_score, "y_true", "y_score")
    yt = as_labels(yt, name="y_true")

    n_pos = float(np.count_nonzero(yt == 1.0))  # scalar
    n_neg = float(np.count_nonzero(yt == 0.0))  # scalar

    if n_pos == 0.0 or n_neg == 0.0:
        raise ValueError(
            f"ROC AUC is undefined with only one class present "
            f"(n_pos={int(n_pos)}, n_neg={int(n_neg)})"
        )

    ranks = _average_ranks(ys)  # shape: (n,)
    rank_sum_pos = float(ranks[yt == 1.0].sum())  # scalar

    u_statistic = rank_sum_pos - n_pos * (n_pos + 1.0) / 2.0  # scalar
    return float(u_statistic / (n_pos * n_neg))


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> float:
    """
    ECE = sum_b (|B_b| / n) * |mean(y in B_b) - mean(p in B_b)|, equal-width bins
    the number depends on the binning, so I only compare ECEs computed with the same n_bins
    """
    yt, yp = _check_pair(y_true, y_prob, "y_true", "y_prob")
    yt = as_labels(yt, name="y_true")

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if np.any(yp < 0.0) or np.any(yp > 1.0):
        raise ValueError("y_prob must lie in [0, 1]")

    n = yt.shape[0]
    edges = np.linspace(0.0, 1.0, n_bins + 1)  # shape: (n_bins + 1,)

    # bin index per observation; clip so p == 1.0 lands in the last bin
    # rather than one past the end
    idx = np.clip(np.digitize(yp, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        in_bin = idx == b  # shape: (n,), bool
        count = int(np.count_nonzero(in_bin))
        if count == 0:
            continue
        observed = float(yt[in_bin].mean())  # scalar
        predicted = float(yp[in_bin].mean())  # scalar
        ece += (count / n) * abs(observed - predicted)

    return float(ece)


def reliability_curve(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict[str, np.ndarray]:
    """
    per-bin mean prediction, observed frequency and count, for a reliability diagram
    empty bins come back as nan; below the diagonal means overconfident
    """
    yt, yp = _check_pair(y_true, y_prob, "y_true", "y_prob")
    yt = as_labels(yt, name="y_true")

    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)  # shape: (n_bins + 1,)
    centers = (edges[:-1] + edges[1:]) / 2.0  # shape: (n_bins,)
    idx = np.clip(np.digitize(yp, edges[1:-1], right=False), 0, n_bins - 1)

    mean_pred = np.full(n_bins, np.nan)  # shape: (n_bins,)
    mean_obs = np.full(n_bins, np.nan)  # shape: (n_bins,)
    counts = np.zeros(n_bins, dtype=np.int64)  # shape: (n_bins,)

    for b in range(n_bins):
        in_bin = idx == b  # shape: (n,), bool
        c = int(np.count_nonzero(in_bin))
        counts[b] = c
        if c > 0:
            mean_pred[b] = float(yp[in_bin].mean())
            mean_obs[b] = float(yt[in_bin].mean())

    return {
        "bin_center": centers,
        "mean_predicted": mean_pred,
        "mean_observed": mean_obs,
        "count": counts,
    }
