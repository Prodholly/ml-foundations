"""
day 7 package: probability, information theory, and the two linear models built on top of them
pass 1 is distributions, bayes, information; pass 2 is losses, optim, model, metrics, splits
"""

from prob_stats.bayes import bayes_update, map_bernoulli
from prob_stats.distributions import (
    log_likelihood_gaussian,
    mle_bernoulli,
    mle_gaussian,
)
from prob_stats.information import (
    cross_entropy,
    cross_entropy_batch,
    entropy,
    kl_divergence,
    kl_divergence_batch,
    verify_relationship,
)
from prob_stats.losses import (
    augment,
    binary_cross_entropy_from_logits,
    logistic_objective,
    mean_squared_error,
    mse_objective,
    penalty_mask_for,
    sigmoid,
)
from prob_stats.metrics import (
    accuracy,
    brier_score,
    expected_calibration_error,
    log_loss,
    r2_score,
    reliability_curve,
    rmse,
    roc_auc,
)
from prob_stats.model import LinearRegression, LogisticRegression
from prob_stats.optim import OptimizeResult, gradient_descent
from prob_stats.splits import Standardizer, train_val_test_split

__all__ = [
    # distributions
    "mle_gaussian",
    "log_likelihood_gaussian",
    "mle_bernoulli",
    # bayes
    "bayes_update",
    "map_bernoulli",
    # information
    "entropy",
    "cross_entropy",
    "kl_divergence",
    "verify_relationship",
    "cross_entropy_batch",
    "kl_divergence_batch",
    # losses
    "sigmoid",
    "augment",
    "penalty_mask_for",
    "mse_objective",
    "logistic_objective",
    "mean_squared_error",
    "binary_cross_entropy_from_logits",
    # optim
    "gradient_descent",
    "OptimizeResult",
    # models
    "LinearRegression",
    "LogisticRegression",
    # metrics
    "rmse",
    "r2_score",
    "accuracy",
    "log_loss",
    "brier_score",
    "roc_auc",
    "expected_calibration_error",
    "reliability_curve",
    # splits
    "train_val_test_split",
    "Standardizer",
]
