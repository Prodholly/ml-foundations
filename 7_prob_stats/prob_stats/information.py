"""
entropy, cross-entropy and KL divergence, single and batched, in nats
the identity tying them together: H(P, Q) = H(P) + KL(P || Q)
"""

from __future__ import annotations

import numpy as np

from prob_stats._validation import as_distribution, as_distribution_rows

__all__ = ["entropy", "cross_entropy", "kl_divergence", "verify_relationship", "cross_entropy_batch", "kl_divergence_batch"]

def entropy(p: np.ndarray) -> float:
    """H(P), the expected surprise; zeros contribute exactly 0 (0 log 0 = 0)"""
    p_arr = as_distribution(p, name="p")  # shape: (k,)

    # support mask: the outcomes that actually carry probability
    support = p_arr > 0.0  # shape: (k,), dtype bool

    p_support = p_arr[support]  # shape: (m,) where m = support.sum() <= k
    surprisal = -np.log(p_support)  # shape: (m,), all finite since p > 0

    # H(P) = E_P[-log P(X)], the probability-weighted mean surprisal
    return float(np.sum(p_support * surprisal))  # scalar


def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """H(P, Q); infinite when q is 0 somewhere p is not"""
    p_arr = as_distribution(p, name="p")  # shape: (k,)
    q_arr = as_distribution(q, name="q")  # shape: (k,)

    if p_arr.shape != q_arr.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p_arr.shape} and "
            f"{q_arr.shape}; cross-entropy requires distributions over "
            f"the same outcome space"
        )

    support = p_arr > 0.0  # shape: (k,), dtype bool

    p_support = p_arr[support]  # shape: (m,)
    q_support = q_arr[support]  # shape: (m,)

    # support mismatch: p puts mass somewhere q rules out entirely
    if np.any(q_support == 0.0):
        return float(np.inf)

    return float(-np.sum(p_support * np.log(q_support)))  # scalar

def cross_entropy_batch(
    P: np.ndarray, Q: np.ndarray, reduction: str = "mean"
) -> float | np.ndarray:
    """row-wise H(P, Q) over (n, k) arrays, reduced by mean, sum or none"""
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(
            f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}"
        )

    P_arr = as_distribution_rows(P, name="P")  # shape: (n, k)
    Q_arr = as_distribution_rows(Q, name="Q")  # shape: (n, k)

    if P_arr.shape != Q_arr.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P_arr.shape} and "
            f"{Q_arr.shape}"
        )

    support = P_arr > 0.0  # shape: (n, k), bool

    # infinite where P has mass and Q does not
    mismatch = support & (Q_arr == 0.0)  # shape: (n, k), bool

    terms = np.zeros_like(P_arr)  # shape: (n, k)
    safe = support & ~mismatch  # shape: (n, k), bool
    terms[safe] = -P_arr[safe] * np.log(Q_arr[safe])

    per_example = terms.sum(axis=1)  # shape: (n,)
    per_example[mismatch.any(axis=1)] = np.inf

    if reduction == "none":
        return per_example
    if reduction == "sum":
        return float(per_example.sum())
    return float(per_example.mean())
    

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(P || Q), the extra nats paid for coding P with Q; never negative"""
    p_arr = as_distribution(p, name="p")  # shape: (k,)
    q_arr = as_distribution(q, name="q")  # shape: (k,)

    if p_arr.shape != q_arr.shape:
        raise ValueError(
            f"p and q must have the same shape, got {p_arr.shape} and "
            f"{q_arr.shape}; KL divergence requires distributions over "
            f"the same outcome space"
        )

    support = p_arr > 0.0  # shape: (k,), dtype bool

    p_support = p_arr[support]  # shape: (m,)
    q_support = q_arr[support]  # shape: (m,)

    if np.any(q_support == 0.0):
        # P considers possible what Q rules out: unbounded divergence
        return float(np.inf)

    # elementwise log-ratio, then probability-weighted sum
    log_ratio = np.log(p_support / q_support)  # shape: (m,)
    return float(np.sum(p_support * log_ratio))  # scalar


def kl_divergence_batch(
    P: np.ndarray, Q: np.ndarray, reduction: str = "mean"
) -> float | np.ndarray:
    """row-wise KL over (n, k) arrays, reduced by mean, sum or none"""
    if reduction not in {"mean", "sum", "none"}:
        raise ValueError(
            f"reduction must be 'mean', 'sum', or 'none', got {reduction!r}"
        )

    P_arr = as_distribution_rows(P, name="P")  # shape: (n, k)
    Q_arr = as_distribution_rows(Q, name="Q")  # shape: (n, k)

    if P_arr.shape != Q_arr.shape:
        raise ValueError(
            f"P and Q must have the same shape, got {P_arr.shape} and "
            f"{Q_arr.shape}"
        )

    support = P_arr > 0.0  # shape: (n, k), bool
    mismatch = support & (Q_arr == 0.0)  # shape: (n, k), bool

    terms = np.zeros_like(P_arr)  # shape: (n, k)
    safe = support & ~mismatch
    terms[safe] = P_arr[safe] * np.log(P_arr[safe] / Q_arr[safe])

    per_example = terms.sum(axis=1)  # shape: (n,)
    per_example[mismatch.any(axis=1)] = np.inf

    if reduction == "none":
        return per_example
    if reduction == "sum":
        return float(per_example.sum())
    return float(per_example.mean())


def verify_relationship(
    p: np.ndarray, q: np.ndarray, atol: float = 1e-6, verbose: bool = True
) -> bool:
    """check H(P, Q) == H(P) + KL(P || Q) numerically, and print the pieces if verbose"""
    h_p = entropy(p)  # scalar
    h_pq = cross_entropy(p, q)  # scalar
    kl_pq = kl_divergence(p, q)  # scalar

    rhs = h_p + kl_pq  # scalar Numerically verify that: H(P, Q) == H(P) + KL(P || Q)
    if np.isinf(h_pq) and np.isinf(rhs):
        holds = True
        residual = 0.0
    else:
        residual = abs(h_pq - rhs)  # scalar
        holds = bool(residual <= atol)

    if verbose:
        print("Verifying  H(P, Q) == H(P) + KL(P || Q)")
        print(f"H(P) = {h_p:.12f} nats")
        print(f" KL(P || Q) = {kl_pq:.12f} nats")
        print(f" H(P) + KL(P||Q)  = {rhs:.12f} nats   <- right-hand side")
        print(f" H(P, Q)  = {h_pq:.12f} nats   <- left-hand side")
        print(f" |difference| = {residual:.3e}  (tolerance {atol:.1e})")
        print(f"identity holds = {holds}")

    return holds
