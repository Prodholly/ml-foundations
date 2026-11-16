"""plain fixed-step gradient descent, with diagnostics I can log for every run"""

from __future__ import annotations

from typing import Callable, Protocol

import numpy as np

from prob_stats._validation import as_1d_float

__all__ = ["gradient_descent", "OptimizeResult"]


class _Objective(Protocol):
    """what the optimizer expects: theta (p,) -> (scalar loss, gradient (p,))"""

    def __call__(self, theta: np.ndarray) -> tuple[float, np.ndarray]: ...


class OptimizeResult:
    """final theta, loss, gradient norm, step count, and whether (and why) it converged"""
    __slots__ = (
        "theta",
        "loss",
        "grad_norm",
        "n_iter",
        "converged",
        "reason",
        "history",
    )

    def __init__(
        self,
        theta: np.ndarray,
        loss: float,
        grad_norm: float,
        n_iter: int,
        converged: bool,
        reason: str,
        history: dict[str, list[float]],
    ) -> None:
        self.theta = theta
        self.loss = loss
        self.grad_norm = grad_norm
        self.n_iter = n_iter
        self.converged = converged
        self.reason = reason
        self.history = history

    def __repr__(self) -> str:
        return (
            f"OptimizeResult(loss={self.loss:.6g}, "
            f"grad_norm={self.grad_norm:.3g}, n_iter={self.n_iter}, "
            f"converged={self.converged}, reason={self.reason!r})"
        )


def gradient_descent(
    objective: _Objective,
    theta0: np.ndarray,
    lr: float = 0.1,
    max_iter: int = 2_000,
    grad_tol: float = 1e-8,
    loss_tol: float = 0.0,
    record_every: int = 1,
) -> OptimizeResult:
    """
    theta <- theta - lr * grad until the largest |grad| entry drops below grad_tol
    convergence is judged on the gradient, not the loss: a tiny loss change can just mean a tiny step
    I raise instead of returning nan when lr is past the stability bound
    """
    theta = as_1d_float(theta0, name="theta0").copy()  # shape: (p,)
    p = theta.shape[0]

    if lr <= 0.0 or not np.isfinite(lr):
        raise ValueError(f"lr must be finite and > 0, got {lr}")
    if max_iter < 1:
        raise ValueError(f"max_iter must be >= 1, got {max_iter}")
    if grad_tol < 0.0 or loss_tol < 0.0:
        raise ValueError("grad_tol and loss_tol must be >= 0")
    if record_every < 1:
        raise ValueError(f"record_every must be >= 1, got {record_every}")

    history: dict[str, list[float]] = {
        "iter": [],
        "loss": [],
        "grad_norm": [],
    }

    loss, grad = objective(theta)
    grad = np.asarray(grad, dtype=np.float64)  # shape: (p,)
    if grad.shape != (p,):
        raise ValueError(
            f"objective returned a gradient of shape {grad.shape}, "
            f"expected ({p},)"
        )

    converged = False
    reason = "max_iter"
    n_iter = 0
    prev_loss = loss

    for it in range(1, max_iter + 1):
        n_iter = it

        if not np.isfinite(loss) or not np.all(np.isfinite(grad)):
            raise RuntimeError(
                f"Objective became non-finite at iteration {it} "
                f"(loss={loss}). The learning rate {lr} is too large for "
                f"this problem's curvature. Reduce lr, or standardize "
                f"the features -- the stability bound scales with the "
                f"largest eigenvalue of X.T @ X."
            )

        grad_norm = float(np.max(np.abs(grad)))  # scalar, the largest single coordinate

        if it == 1 or it % record_every == 0:
            history["iter"].append(float(it))
            history["loss"].append(float(loss))
            history["grad_norm"].append(grad_norm)

        if grad_norm <= grad_tol:
            converged = True
            reason = "grad_tol"
            break

        
        theta = theta - lr * grad  # shape: (p,), one step downhill

        loss, grad = objective(theta)
        grad = np.asarray(grad, dtype=np.float64)  # shape: (p,)

        if loss_tol > 0.0 and abs(prev_loss - loss) <= loss_tol:
            converged = True
            reason = "loss_tol"
            break
        prev_loss = loss

    final_grad_norm = float(np.max(np.abs(grad)))

    # record the terminal state if the loop exited without recording it
    if not history["iter"] or history["iter"][-1] != float(n_iter):
        history["iter"].append(float(n_iter))
        history["loss"].append(float(loss))
        history["grad_norm"].append(final_grad_norm)

    return OptimizeResult(
        theta=theta,
        loss=float(loss),
        grad_norm=final_grad_norm,
        n_iter=n_iter,
        converged=converged,
        reason=reason,
        history=history,
    )
