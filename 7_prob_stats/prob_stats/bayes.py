"""bayes' rule over a discrete hypothesis space, and MAP for a coin with a beta prior"""

from __future__ import annotations
import numpy as np
from prob_stats._validation import as_1d_float, as_binary

__all__ = ["bayes_update", "map_bernoulli"]


def bayes_update(prior: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    """
    posterior over k hypotheses from a prior (k,) and a likelihood (k,)
    I raise if the evidence is zero, since then no hypothesis can explain the data
    """
    prior_arr  = as_1d_float(prior, name="prior") # shape: (k,)
    lik_arr = as_1d_float(likelihood, name="likelihood") # shape: (k,)
    
    if prior_arr.shape != lik_arr.shape:
        raise ValueError(
            f"prior and likelihood must have the same shape but you gave "
            f"{prior_arr.shape} and {lik_arr.shape}"
        )
        
    if np.any(prior_arr < 0.0):
        raise ValueError("prior must be non-negative")
    if np.any(lik_arr <0.0):
        raise ValueError("likelihood must be non-negative")
    
    prior_sum = float(prior_arr.sum()) # scalar
    if not np.isclose(prior_sum, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(
            f"prior must sum to 1.0 however you gave {prior_sum!r}."
            f"a prior is a distribution over hypothesis; normalize it first"
        )
        
    # bayes' rule (unnormalized)
    unnormalized = lik_arr * prior_arr # shape: (k,)
    
    # the evidence P(X): a scalar obtained by summing out the hypothesis
    evidence = float(unnormalized.sum())
    if evidence <= 0.0:
        raise ValueError(
            "Unnormalized posterior sums to zero, as such, every hypothesis with "
            "non-zero prior assigns zero likelihood to the observed "
            "data. The hypothesis space cannot explain the data and no "
            "posterior exists."
        )
    posterior = unnormalized / evidence # shape: (k, ), sums to 1
    return posterior


def map_bernoulli(data: np.ndarray, alpha: float, beta: float) -> float:
    """
    mode of the Beta(k + alpha, n - k + beta) posterior
    alpha = beta = 1 is a flat prior, and then MAP equals the MLE
    """
    x = as_binary(data, name="data") #shape: (n,), values in {0.0, 1.0}
    n = x.shape[0] # scalar int
    
    alpha =float(alpha)
    beta = float(beta)
    
    for name, value in (("alpha", alpha), ("beta", beta)):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(
                f"{name} must be finite and strictly positive but you provided "
                f"{value!r}; Beta shape parameters are > 0 by definition"
            )
            
    n_successes = float(x.sum()) # scalar, =k
    
    a_post = n_successes + alpha # scalar
    b_post = (n-n_successes) + beta # scalar
    
    # the posterior is Beta(a_post, b_post); where its peak sits depends on which side of 1 they fall
    if a_post > 1.0 and b_post > 1.0:  # hump-shaped, the mode is inside (0, 1)
        return float((a_post - 1.0) / (a_post + b_post - 2.0))
    
    if a_post == 1.0 and b_post == 1.0:  # flat, every p equally likely, so I pick the middle
        return 0.5
    
    if a_post < 1.0 and b_post < 1.0:  # U-shaped, no single peak
        raise ValueError(
            f"Posterior is Beta({a_post}, {b_post}), which is U-shaped "
            f"and has no unique mode (the density diverges at both 0 "
            f"and 1). MAP is undefined. Use alpha, beta >= 1, or "
            f"collect at least one observation, or switch to the "
            f"posterior mean (a / (a + b)), which is always defined."
        )

    # J-shaped, the peak sits on a boundary
    return 0.0 if a_post <= 1.0 and b_post >= 1.0 else 1.0
