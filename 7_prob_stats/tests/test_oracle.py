"""
correctness oracles: pytorch and scikit-learn, imported only in this file

torch.autograd checks my hand-derived gradients, to 1e-12
sklearn checks the fitted solutions, once the penalty conventions are converted:
    Ridge:               alpha = n * l2
    LogisticRegression:  C     = 1 / (2 * n * l2)

every tensor is float64; torch's float32 default caps agreement near 1e-7, loose enough to hide a sign error
for logistic regression I compare objective values tightly and coefficients loosely,
since my fixed-step GD and sklearn's lbfgs approach the same optimum at different speeds

run:  pytest 7_prob_stats/tests/test_oracle.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from prob_stats import (
    LinearRegression,
    LogisticRegression,
    augment,
    binary_cross_entropy_from_logits,
    cross_entropy_batch,
    entropy,
    kl_divergence,
    kl_divergence_batch,
    log_likelihood_gaussian,
    logistic_objective,
    mle_gaussian,
    mse_objective,
    penalty_mask_for,
    sigmoid,
)

SEED = 42


def _regression_data(n=300, d=5, seed=SEED):
    """X (n,d), y (n,)"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = X @ rng.normal(size=d) + 1.5 + rng.normal(scale=0.5, size=n)
    return X, y


def _classification_data(n=300, d=5, seed=SEED):
    """X (n,d), y (n,) in {0,1}"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    logits = X @ rng.normal(size=d) - 0.2
    y = (rng.random(n) < sigmoid(logits)).astype(np.float64)
    return X, y


# =====================================================================
# PyTorch: gradients and information quantities
# =====================================================================
class TestTorchGradients:
    """
    every hand-derived gradient, checked against autograd
    if one of these is wrong, every model here quietly converges to the wrong point
    """

    def test_mse_gradient_matches_autograd(self):
        torch = pytest.importorskip("torch")

        rng = np.random.default_rng(SEED)
        X_aug = augment(rng.normal(size=(60, 4)))  # shape: (60, 5)
        y = rng.normal(size=60)  # shape: (60,)
        mask = penalty_mask_for(5, True)  # shape: (5,)
        l2 = 0.37

        for _ in range(10):
            theta = rng.normal(size=5)  # shape: (5,)
            _, ours = mse_objective(theta, X_aug, y, l2, mask)  # (5,)

            t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
            Xt = torch.tensor(X_aug, dtype=torch.float64)
            yt = torch.tensor(y, dtype=torch.float64)
            mt = torch.tensor(mask, dtype=torch.float64)

            loss = ((Xt @ t - yt) ** 2).mean() + l2 * (mt * t**2).sum()
            loss.backward()

            np.testing.assert_allclose(
                ours, t.grad.numpy(), rtol=1e-12, atol=1e-12
            )

    def test_logistic_gradient_matches_autograd(self):
        torch = pytest.importorskip("torch")

        rng = np.random.default_rng(SEED)
        X_aug = augment(rng.normal(size=(60, 4)))  # shape: (60, 5)
        y = (rng.random(60) < 0.5).astype(np.float64)  # shape: (60,)
        mask = penalty_mask_for(5, True)
        l2 = 0.21

        for _ in range(10):
            theta = rng.normal(size=5)  # shape: (5,)
            ours_loss, ours_grad = logistic_objective(
                theta, X_aug, y, l2, mask
            )

            t = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
            Xt = torch.tensor(X_aug, dtype=torch.float64)
            yt = torch.tensor(y, dtype=torch.float64)
            mt = torch.tensor(mask, dtype=torch.float64)

            z = Xt @ t
            # softplus(z) - y*z is exactly our formulation
            loss = (torch.nn.functional.softplus(z) - yt * z).mean() + l2 * (
                mt * t**2
            ).sum()
            loss.backward()

            assert ours_loss == pytest.approx(loss.item(), rel=1e-12)
            np.testing.assert_allclose(
                ours_grad, t.grad.numpy(), rtol=1e-12, atol=1e-12
            )

    def test_bce_matches_torch_with_logits(self):
        torch = pytest.importorskip("torch")

        rng = np.random.default_rng(SEED)
        y = (rng.random(200) < 0.5).astype(np.float64)  # shape: (200,)
        # include saturated logits, where the probability-space version
        # of this loss would return inf
        logits = np.concatenate(
            [rng.normal(scale=3.0, size=198), [-60.0, 60.0]]
        )  # shape: (200,)

        ours = binary_cross_entropy_from_logits(y, logits)
        theirs = torch.nn.functional.binary_cross_entropy_with_logits(
            torch.tensor(logits, dtype=torch.float64),
            torch.tensor(y, dtype=torch.float64),
        ).item()

        assert ours == pytest.approx(theirs, rel=1e-12)

    def test_sigmoid_matches_torch(self):
        torch = pytest.importorskip("torch")

        z = np.array([-1e5, -700.0, -1.0, 0.0, 1.0, 700.0, 1e5])  # (7,)
        ours = sigmoid(z)
        theirs = torch.sigmoid(torch.tensor(z, dtype=torch.float64)).numpy()
        np.testing.assert_allclose(ours, theirs, rtol=1e-14, atol=0.0)


class TestTorchInformation:
    """Entropy and KL against ``torch.distributions``, batched"""

    def test_batched_entropy_matches_categorical(self):
        torch = pytest.importorskip("torch")

        rng = np.random.default_rng(SEED)
        P = rng.dirichlet(np.ones(6), size=25)  # shape: (25, 6)

        ours = np.array([entropy(row) for row in P])  # shape: (25,)
        theirs = (
            torch.distributions.Categorical(
                probs=torch.tensor(P, dtype=torch.float64)
            )
            .entropy()
            .numpy()
        )  # shape: (25,)

        np.testing.assert_allclose(ours, theirs, rtol=1e-12)

    def test_batched_kl_matches_torch(self):
        torch = pytest.importorskip("torch")

        rng = np.random.default_rng(SEED)
        P = rng.dirichlet(np.ones(5), size=30)  # shape: (30, 5)
        Q = rng.dirichlet(np.ones(5), size=30)  # shape: (30, 5)

        ours = kl_divergence_batch(P, Q, reduction="none")  # shape: (30,)
        theirs = torch.distributions.kl_divergence(
            torch.distributions.Categorical(
                probs=torch.tensor(P, dtype=torch.float64)
            ),
            torch.distributions.Categorical(
                probs=torch.tensor(Q, dtype=torch.float64)
            ),
        ).numpy()  # shape: (30,)

        np.testing.assert_allclose(ours, theirs, rtol=1e-11, atol=1e-14)

        # scalar and batched paths must agree with each other too
        for i in range(5):
            assert ours[i] == pytest.approx(
                kl_divergence(P[i], Q[i]), rel=1e-12
            )

    def test_one_hot_cross_entropy_equals_torch_classification_loss(self):
        """
        my batched cross-entropy on one-hot targets must equal F.cross_entropy on the logits
        torch takes logits and applies log-softmax itself; mine takes probabilities, so I softmax first
        """
        torch = pytest.importorskip("torch")

        rng = np.random.default_rng(SEED)
        logits = torch.tensor(
            rng.normal(size=(20, 4)), dtype=torch.float64
        )  # shape: (20, 4)
        targets = torch.tensor(rng.integers(0, 4, size=20))  # shape: (20,)

        probs = torch.softmax(logits, dim=1).numpy()  # shape: (20, 4)
        one_hot = np.zeros((20, 4))  # shape: (20, 4)
        one_hot[np.arange(20), targets.numpy()] = 1.0

        ours = cross_entropy_batch(one_hot, probs, reduction="mean")
        theirs = torch.nn.functional.cross_entropy(logits, targets).item()

        assert ours == pytest.approx(theirs, rel=1e-12)

        # one-hot targets have zero entropy, so cross-entropy IS KL
        assert ours == pytest.approx(
            kl_divergence_batch(one_hot, probs, reduction="mean"), rel=1e-12
        )


# =====================================================================
# scikit-learn: fitted solutions
# =====================================================================
class TestSklearnRegression:
    def test_unpenalized_matches_linear_regression(self):
        sklm = pytest.importorskip("sklearn.linear_model")

        X, y = _regression_data(n=400, d=6)

        ours = LinearRegression(l2=0.0).fit(X, y)
        theirs = sklm.LinearRegression().fit(X, y)

        np.testing.assert_allclose(ours.coef_, theirs.coef_, rtol=1e-9)
        assert ours.intercept_ == pytest.approx(
            float(theirs.intercept_), rel=1e-9
        )

    def test_penalized_matches_ridge_with_alpha_conversion(self):
        """
        mine is a mean plus l2 * ||w||^2, Ridge's is a sum plus alpha * ||w||^2, so alpha = n * l2
        neither penalizes the intercept
        """
        sklm = pytest.importorskip("sklearn.linear_model")

        X, y = _regression_data(n=500, d=8)
        n = X.shape[0]

        for l2 in (1e-4, 1e-2, 0.1, 1.0):
            ours = LinearRegression(l2=l2).fit(X, y)
            theirs = sklm.Ridge(alpha=n * l2, fit_intercept=True).fit(X, y)

            np.testing.assert_allclose(
                ours.coef_, theirs.coef_, rtol=1e-8, atol=1e-10
            )
            assert ours.intercept_ == pytest.approx(
                float(theirs.intercept_), rel=1e-8
            )

    def test_wrong_alpha_convention_visibly_disagrees(self):
        """
        negative control: alpha = l2 instead of n * l2 must give a clearly different fit,
        otherwise the test above could pass for the wrong reason
        """
        sklm = pytest.importorskip("sklearn.linear_model")

        X, y = _regression_data(n=500, d=8)
        ours = LinearRegression(l2=1.0).fit(X, y)
        naive = sklm.Ridge(alpha=1.0).fit(X, y)

        assert not np.allclose(ours.coef_, naive.coef_, rtol=1e-3)

    def test_residual_variance_matches_numpy_ddof_zero(self):
        """the residual variance is the MLE, so it matches np.var with ddof=0"""
        X, y = _regression_data(n=300, d=4)
        model = LinearRegression().fit(X, y)
        residuals = y - model.predict(X)  # shape: (300,)

        _, mle_var = mle_gaussian(residuals)
        assert model.sigma_sq_ == pytest.approx(mle_var, rel=1e-12)
        assert model.sigma_sq_ == pytest.approx(
            float(np.var(residuals, ddof=0)), rel=1e-12
        )
        assert model.sigma_sq_ != pytest.approx(
            float(np.var(residuals, ddof=1)), rel=1e-12
        )


class TestSklearnLogistic:
    def test_objective_matches_sklearn_with_c_conversion(self):
        """
        sklearn minimizes 0.5 ||w||^2 + C sum_i loss_i; I minimize mean_i loss_i + l2 ||w||^2
        multiply mine by n and match:  n l2 = 1 / (2C), so C = 1 / (2 n l2)
        compared on the objective, since a coefficient gap could just mean one optimizer is less converged
        """
        sklm = pytest.importorskip("sklearn.linear_model")

        X, y = _classification_data(n=600, d=6)
        n, d = X.shape
        X_aug = augment(X)  # shape: (n, d+1)
        mask = penalty_mask_for(d + 1, True)  # shape: (d+1,)

        for l2 in (1e-3, 1e-2, 0.1):
            ours = LogisticRegression(l2=l2).fit(
                X, y, lr=0.5, max_iter=50_000, grad_tol=1e-11
            )
            theirs = sklm.LogisticRegression(
                C=1.0 / (2.0 * n * l2),
                penalty="l2",
                solver="lbfgs",
                tol=1e-10,
                max_iter=5_000,
                fit_intercept=True,
            ).fit(X, y)

            their_theta = np.concatenate(
                [np.atleast_1d(theirs.intercept_), theirs.coef_.ravel()]
            )  # shape: (d+1,)

            obj_ours, _ = logistic_objective(
                ours.theta_, X_aug, y, l2, mask
            )
            obj_theirs, _ = logistic_objective(
                their_theta, X_aug, y, l2, mask
            )

            # our solution must be essentially as good at our own
            # objective as sklearn's is
            assert obj_ours <= obj_theirs + 1e-7
            assert obj_ours == pytest.approx(obj_theirs, rel=1e-5)

            # and it must actually be a stationary point
            assert ours.optimize_result_.grad_norm < 1e-8

            # coefficients then agree, loosely -- documented tolerance
            np.testing.assert_allclose(
                ours.coef_, theirs.coef_.ravel(), rtol=5e-3, atol=5e-3
            )

    def test_predicted_probabilities_match_sklearn(self):
        sklm = pytest.importorskip("sklearn.linear_model")

        X, y = _classification_data(n=500, d=5)
        n = X.shape[0]
        l2 = 0.01

        ours = LogisticRegression(l2=l2).fit(
            X, y, lr=0.5, max_iter=50_000, grad_tol=1e-11
        )
        theirs = sklm.LogisticRegression(
            C=1.0 / (2.0 * n * l2), solver="lbfgs", tol=1e-10, max_iter=5_000
        ).fit(X, y)

        np.testing.assert_allclose(
            ours.predict_proba(X),
            theirs.predict_proba(X)[:, 1],
            rtol=1e-3,
            atol=1e-3,
        )


class TestSklearnMetrics:
    def test_metrics_match_sklearn(self):
        skm = pytest.importorskip("sklearn.metrics")
        from prob_stats import brier_score, log_loss, r2_score, roc_auc

        rng = np.random.default_rng(SEED)

        # regression metrics
        y_true = rng.normal(size=400)  # shape: (400,)
        y_pred = y_true + rng.normal(scale=0.5, size=400)  # shape: (400,)
        assert r2_score(y_true, y_pred) == pytest.approx(
            skm.r2_score(y_true, y_pred), rel=1e-12
        )

        # classification metrics
        labels = (rng.random(400) < 0.4).astype(np.float64)  # shape: (400,)
        logits = rng.normal(scale=2.0, size=400)  # shape: (400,)
        probs = sigmoid(logits)  # shape: (400,)

        assert roc_auc(labels, probs) == pytest.approx(
            skm.roc_auc_score(labels, probs), rel=1e-12
        )
        assert log_loss(labels, logits) == pytest.approx(
            skm.log_loss(labels, probs), rel=1e-8
        )
        assert brier_score(labels, probs) == pytest.approx(
            skm.brier_score_loss(labels, probs), rel=1e-12
        )

    def test_auc_matches_sklearn_under_heavy_ties(self):
        """tie handling is the entire correctness question for AUC"""
        skm = pytest.importorskip("sklearn.metrics")
        from prob_stats import roc_auc

        rng = np.random.default_rng(SEED)
        labels = (rng.random(300) < 0.5).astype(np.float64)  # shape: (300,)
        # only 4 distinct scores -> massive ties
        scores = rng.choice([0.1, 0.3, 0.6, 0.9], size=300)  # shape: (300,)

        assert roc_auc(labels, scores) == pytest.approx(
            skm.roc_auc_score(labels, scores), rel=1e-12
        )


# =====================================================================
# the MAP / L2 identity
# =====================================================================
class TestMAPEquivalence:
    """
    L2 regularization is MAP under a gaussian prior theta_j ~ N(0, tau^2), an identity, so any gap is a bug
    sum-scale MAP:  -sum_i log p(y_i | x_i, theta) + ||theta||^2 / (2 tau^2)
    mine, on the mean scale:  mean NLL + l2 * ||theta||^2
    multiply mine by n and match:  n * l2 = 1 / (2 tau^2)  <=>  tau^2 = 1 / (2 n l2)
    """

    @staticmethod
    def _tau_sq(n: int, l2: float) -> float:
        return 1.0 / (2.0 * n * l2)

    def test_logistic_l2_equals_gaussian_prior_map(self):
        rng = np.random.default_rng(SEED)
        X, y = _classification_data(n=150, d=5)
        n, d = X.shape
        X_aug = augment(X)  # shape: (n, d+1)
        mask = penalty_mask_for(d + 1, True)  # shape: (d+1,)

        for l2 in (1e-3, 0.05, 0.5):
            tau_sq = self._tau_sq(n, l2)
            for _ in range(5):
                theta = rng.normal(size=d + 1)  # shape: (d+1,)

                our_obj, _ = logistic_objective(theta, X_aug, y, l2, mask)

                nll_sum = float(
                    binary_cross_entropy_from_logits(
                        y, X_aug @ theta, reduction="sum"
                    )
                )
                log_prior = -float(np.sum(mask * theta**2)) / (2.0 * tau_sq)
                map_obj = nll_sum - log_prior

                assert n * our_obj == pytest.approx(map_obj, rel=1e-12)

    def test_linear_l2_equals_gaussian_prior_map(self):
        rng = np.random.default_rng(SEED)
        X, y = _regression_data(n=150, d=5)
        n, d = X.shape
        X_aug = augment(X)  # shape: (n, d+1)
        mask = penalty_mask_for(d + 1, True)

        for l2 in (1e-3, 0.05, 0.5):
            tau_sq = self._tau_sq(n, l2)
            for _ in range(5):
                theta = rng.normal(size=d + 1)  # shape: (d+1,)

                our_obj, _ = mse_objective(theta, X_aug, y, l2, mask)

                # Gaussian NLL on the sum-of-squares scale
                resid = X_aug @ theta - y  # shape: (n,)
                sse = float(resid @ resid)
                log_prior = -float(np.sum(mask * theta**2)) / (2.0 * tau_sq)

                assert n * our_obj == pytest.approx(
                    sse - log_prior, rel=1e-12
                )

    def test_flat_prior_recovers_mle(self):
        """tau^2 -> infinity means l2 -> 0 means plain MLE; checked with a negligible penalty"""
        X, y = _regression_data(n=300, d=4)
        mle = LinearRegression(l2=0.0).fit(X, y)
        nearly_flat = LinearRegression(l2=1e-14).fit(X, y)
        np.testing.assert_allclose(
            mle.coef_, nearly_flat.coef_, rtol=1e-7, atol=1e-9
        )

    def test_gaussian_nll_is_the_regression_objective(self):
        """
        minimizing MSE maximizes the gaussian log-likelihood, so the fitted parameters
        must be the argmax of log_likelihood_gaussian on the residuals
        """
        X, y = _regression_data(n=400, d=3)
        model = LinearRegression(l2=0.0).fit(X, y)

        resid = y - model.predict(X)  # shape: (400,)
        best = log_likelihood_gaussian(resid, 0.0, model.sigma_sq_)

        # perturbing the fitted weights must lower the log-likelihood
        for j in range(3):
            for delta in (-0.1, 0.1):
                coef = model.coef_.copy()
                coef[j] += delta
                worse_resid = y - (X @ coef + model.intercept_)
                _, var = mle_gaussian(worse_resid)
                assert (
                    log_likelihood_gaussian(worse_resid, 0.0, var) < best
                )
