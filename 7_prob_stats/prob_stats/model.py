"""linear and logistic regression on top of the losses and the optimizer"""

from __future__ import annotations

import numpy as np

from prob_stats._validation import as_2d_float, as_labels, check_xy
from prob_stats.distributions import log_likelihood_gaussian, mle_gaussian
from prob_stats.information import cross_entropy_batch
from prob_stats.losses import (
    augment,
    logistic_objective,
    mse_objective,
    penalty_mask_for,
    sigmoid,
)
from prob_stats.optim import gradient_descent

__all__ = ["LinearRegression", "LogisticRegression"]


class _LinearModelBase:
    """shared plumbing: penalty, intercept handling, fitted-state checks, decision_function"""
    def __init__(self, l2: float = 0.0, fit_intercept: bool = True) -> None:
        if l2 < 0.0 or not np.isfinite(l2):
            raise ValueError(f"l2 must be finite and >= 0, got {l2}")
        self.l2 = float(l2)
        self.fit_intercept = bool(fit_intercept)

        # fitted state, all None until fit() succeeds
        self.theta_: np.ndarray | None = None  # shape: (p,)
        self.n_features_in_: int | None = None
        self.optimize_result_ = None

    # -- fitted-parameter views ---
    @property
    def intercept_(self) -> float:
        """fitted intercept, or 0.0 when fit_intercept=False"""
        self._check_fitted()
        return float(self.theta_[0]) if self.fit_intercept else 0.0

    @property
    def coef_(self) -> np.ndarray:
        """fitted feature weights; shape (d,)"""
        self._check_fitted()
        return self.theta_[1:].copy() if self.fit_intercept else self.theta_.copy()

    # -- helpers --------
    def _check_fitted(self) -> None:
        if self.theta_ is None:
            raise RuntimeError(
                f"{type(self).__name__} is not fitted; call fit() first"
            )

    def _design(self, X: np.ndarray) -> np.ndarray:
        """validate X against the fitted feature count and add the ones column"""
        X_arr = as_2d_float(X, name="X")  # shape: (n, d)
        if (
            self.n_features_in_ is not None
            and X_arr.shape[1] != self.n_features_in_
        ):
            raise ValueError(
                f"X has {X_arr.shape[1]} features, but this model was "
                f"fitted on {self.n_features_in_}"
            )
        return augment(X_arr) if self.fit_intercept else X_arr

    def _mask(self, p: int) -> np.ndarray:
        """which parameters the penalty touches (everything but the intercept)"""
        return penalty_mask_for(p, self.fit_intercept)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """raw linear scores X theta; shape (n,)"""
        self._check_fitted()
        X_aug = self._design(X)  # shape: (n, p)
        return X_aug @ self.theta_  # shape: (n,)


class LinearRegression(_LinearModelBase):
    """least squares with an optional l2 penalty (ridge), solved exactly or by gradient descent"""
    def __init__(self, l2: float = 0.0, fit_intercept: bool = True) -> None:
        super().__init__(l2=l2, fit_intercept=fit_intercept)
        self.sigma_sq_: float | None = None
        self.log_likelihood_: float | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = "closed_form",
        lr: float = 0.1,
        max_iter: int = 2_000,
        grad_tol: float = 1e-10,
    ) -> "LinearRegression":
        """
        closed form solves (X^T X + n l2 P) theta = X^T y; "gd" runs gradient descent on the same objective
        afterwards I store the residual variance (MLE) and the training log-likelihood
        """
        X_arr, y_arr = check_xy(X, y)  # shapes: (n, d), (n,)
        n, d = X_arr.shape
        self.n_features_in_ = d

        X_aug = augment(X_arr) if self.fit_intercept else X_arr  # (n, p)
        p = X_aug.shape[1]
        mask = self._mask(p)  # shape: (p,)

        if method == "closed_form":
            # (X.T X + n*l2*P) theta = X.T y
            gram = X_aug.T @ X_aug  # shape: (p, p)
            reg = n * self.l2 * np.diag(mask)  # shape: (p, p)
            rhs = X_aug.T @ y_arr  # shape: (p,)
            # in floats a singular gram is only *nearly* singular, so solve() rarely raises;
            # check the rank directly when there's no penalty to rescue it
            rank_deficient = self.l2 == 0.0 and np.linalg.matrix_rank(X_aug) < p
            try:
                if rank_deficient:
                    raise np.linalg.LinAlgError("rank-deficient X")
                self.theta_ = np.linalg.solve(gram + reg, rhs)  # (p,)
            except np.linalg.LinAlgError as exc:
                raise ValueError(
                    f"Normal equations are singular with n={n}, d={d}, "
                    f"l2={self.l2}. This happens when n < d or when "
                    f"features are exactly collinear. Any l2 > 0 makes "
                    f"the system positive definite and solvable."
                ) from exc
            self.optimize_result_ = None

        elif method == "gd":
            def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
                return mse_objective(theta, X_aug, y_arr, self.l2, mask)

            result = gradient_descent(
                objective,
                theta0=np.zeros(p),  # shape: (p,)
                lr=lr,
                max_iter=max_iter,
                grad_tol=grad_tol,
            )
            self.theta_ = result.theta  # shape: (p,)
            self.optimize_result_ = result

        else:
            raise ValueError(
                f"method must be 'closed_form' or 'gd', got {method!r}"
            )

        # residual-variance MLE and training log-likelihood, computed by
        # the pass 1 estimators on this model's own residuals
        residuals = y_arr - (X_aug @ self.theta_)  # shape: (n,)
        _, self.sigma_sq_ = mle_gaussian(residuals)

        if self.sigma_sq_ > 0.0:
            self.log_likelihood_ = log_likelihood_gaussian(
                residuals, mu=0.0, sigma_sq=self.sigma_sq_
            )
        else:
            # perfect fit: a zero-variance Gaussian has no density
            # reported as None rather than inf
            self.log_likelihood_ = None

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X theta; shape (n,)"""
        return self.decision_function(X)


class LogisticRegression(_LinearModelBase):
    """binary logistic regression with an optional l2 penalty, fitted by gradient descent"""
    def __init__(self, l2: float = 0.0, fit_intercept: bool = True) -> None:
        super().__init__(l2=l2, fit_intercept=fit_intercept)
        self.log_loss_: float | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lr: float = 0.5,
        max_iter: int = 5_000,
        grad_tol: float = 1e-8,
    ) -> "LogisticRegression":
        """
        minimize mean cross-entropy + l2 penalty; there's no closed form, so it's gradient descent only
        """
        X_arr, y_arr = check_xy(X, y)  # shapes: (n, d), (n,)
        y_arr = as_labels(y_arr, name="y")  # shape: (n,), in {0, 1}
        n, d = X_arr.shape
        self.n_features_in_ = d

        X_aug = augment(X_arr) if self.fit_intercept else X_arr  # (n, p)
        p = X_aug.shape[1]
        mask = self._mask(p)  # shape: (p,)

        def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
            return logistic_objective(theta, X_aug, y_arr, self.l2, mask)

        result = gradient_descent(
            objective,
            theta0=np.zeros(p),  # shape: (p,)
            lr=lr,
            max_iter=max_iter,
            grad_tol=grad_tol,
        )
        self.theta_ = result.theta  # shape: (p,)
        self.optimize_result_ = result

        # training cross-entropy, routed through the pass 1 primitive to
        # keep one implementation of the quantity rather than two
        probs = self.predict_proba(X_arr)  # shape: (n,)
        P = np.column_stack([1.0 - y_arr, y_arr])  # shape: (n, 2), one-hot
        Q = np.column_stack([1.0 - probs, probs])  # shape: (n, 2)
        # guard the saturated case so the reported number stays finite;
        # the *objective* needs no such guard because it works in logit
        # space (see losses.logistic_objective)
        Q = np.clip(Q, 1e-300, 1.0)
        Q = Q / Q.sum(axis=1, keepdims=True)
        self.log_loss_ = float(cross_entropy_batch(P, Q, reduction="mean"))

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """P(y = 1 | x) = sigmoid(X theta); shape (n,)"""
        return sigmoid(self.decision_function(X))  # shape: (n,)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """hard 0/1 labels at the given threshold; shape (n,)"""
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                f"threshold must be in (0, 1), got {threshold}"
            )
        return (self.predict_proba(X) >= threshold).astype(np.float64)
