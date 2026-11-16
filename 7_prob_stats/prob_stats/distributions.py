"""maximum likelihood for the gaussian and the bernoulli, plus the gaussian log-likelihood"""

from __future__ import annotations
import numpy as np
from prob_stats._validation import as_1d_float, as_binary

__all__ = ["mle_gaussian", "log_likelihood_gaussian", "mle_bernoulli"]

def mle_gaussian(data: np.ndarray) -> tuple[float, float]: # returns estimated mean and estimated variance
    """
    (mu_hat, sigma_sq_hat) that maximize the gaussian likelihood
    the variance divides by n, not n - 1: that's the MLE, and it's biased on purpose
    """
    x = as_1d_float(data, name="data")
    n = x.shape[0] # number of observations / scalar int
    
    mu_hat = float(x.sum() / n)
    deviations = x - mu_hat
    sigma_sq_hat = float((deviations**2).sum() / n)  # / n, not / (n - 1)
    
    return mu_hat, sigma_sq_hat

def log_likelihood_gaussian(data: np.ndarray, mu: float, sigma_sq: float) -> float:
    """
    log p(data | mu, sigma_sq), summed over all n points, in nats
    computed straight from the log form so a large n never underflows
    """
    x = as_1d_float(data, name="data") # shape: (n,)
    n = x.shape[0] # scalar int
    
    mu = float(mu)
    sigma_sq = float(sigma_sq)
    
    if not np.isfinite(mu):
        raise ValueError(f"mu must be finite but got {mu!r}")

    if not np.isfinite(sigma_sq) or sigma_sq <= 0.0:
        raise ValueError(
            f"sigma_sq must be finite and strictly positive but got "
            f"{sigma_sq!r}. A zero or negative variance cannot define a Gaussian density"
        )
        
    # term 1: the normalizing constant once per observation
    # -(n/2) * log(2* pi*sigma^2)
    log_norm = -(0.5 * n) * np.log(2 * np.pi * sigma_sq) # scalar
    
    # term2: -(1 / (2 sigma^2)) * sum_i (x_i - mu)^2 
    data_fit = -(1 / (2 * sigma_sq)) * np.sum((x - mu)**2)
    
    return float(log_norm + data_fit)


def mle_bernoulli(data: np.ndarray) -> float:
    """fraction of ones: the p that maximizes the bernoulli likelihood"""
    x = as_binary(data, name = "data") # shape: (n,), values in {0.0, 1.0}
    n = x.shape[0] # some scalar int
    
    n_successes = float(x.sum()) #scalar
    return float(n_successes / n)
