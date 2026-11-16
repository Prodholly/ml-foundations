"""pass 2 tests: losses, optimizer, models, metrics, splits, plus adversarial inputs"""



from __future__ import annotations

import numpy as np
import pytest

from prob_stats import (
    LinearRegression,
    LogisticRegression,
    Standardizer,
    accuracy,
    augment,
    binary_cross_entropy_from_logits,
    brier_score,
    expected_calibration_error,
    gradient_descent,
    log_loss,
    logistic_objective,
    mse_objective,
    penalty_mask_for,
    r2_score,
    reliability_curve,
    rmse,
    roc_auc,
    sigmoid,
    train_val_test_split,
)
from prob_stats.metrics import _average_ranks

SEED = 42


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def make_regression(n=200, d=5, noise=1.0, seed=SEED):
    """synthetic linear data; returns X (n,d), y (n,), true_w (d,)"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))  # shape: (n, d)
    w = rng.normal(size=d)  # shape: (d,)
    y = X @ w + 2.0 + rng.normal(scale=noise, size=n)  # shape: (n,)
    return X, y, w


def make_classification(n=300, d=5, scale=1.5, seed=SEED):
    """synthetic logistic data; returns X (n,d), y (n,), true_w (d,)"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))  # shape: (n, d)
    w = rng.normal(scale=scale, size=d)  # shape: (d,)
    logits = X @ w - 0.3  # shape: (n,)
    y = (rng.random(n) < sigmoid(logits)).astype(np.float64)  # shape: (n,)
    return X, y, w


# ---------------------------------------------------------------------
# sigmoid
# ---------------------------------------------------------------------
class TestSigmoid:
    def test_trivial_midpoint_and_symmetry(self):
        assert sigmoid(np.array([0.0]))[0] == pytest.approx(0.5)
        z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])  # shape: (5,)
        np.testing.assert_allclose(
            sigmoid(z) + sigmoid(-z), np.ones(5), atol=1e-15
        )

    def test_realistic_monotone_and_bounded(self):
        z = np.linspace(-10.0, 10.0, 201)  # shape: (201,)
        p = sigmoid(z)  # shape: (201,)
        assert np.all(np.diff(p) > 0.0)
        assert np.all((p >= 0.0) & (p <= 1.0))

    def test_edge_extreme_logits_do_not_overflow(self):
        """
        the naive 1/(1+exp(-z)) warns and overflows near z = -710
        the branched form must stay silent and finite
        """
        z = np.array([-1e6, -800.0, 0.0, 800.0, 1e6])  # shape: (5,)
        with np.errstate(over="raise", invalid="raise"):
            p = sigmoid(z)
        assert np.all(np.isfinite(p))
        assert p[0] == 0.0 and p[-1] == 1.0


# ---------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------
class TestLosses:
    def test_trivial_mse_zero_at_perfect_fit(self):
        X_aug = augment(np.array([[1.0], [2.0], [3.0]]))  # shape: (3, 2)
        theta = np.array([0.0, 2.0])  # shape: (2,)
        y = X_aug @ theta  # shape: (3,)
        mask = penalty_mask_for(2, True)  # shape: (2,)

        loss, grad = mse_objective(theta, X_aug, y, 0.0, mask)

        assert loss == pytest.approx(0.0)
        np.testing.assert_allclose(grad, np.zeros(2), atol=1e-14)

    def test_trivial_bce_at_zero_logits(self):
        """all logits 0 -> p = 0.5 -> loss = log 2 for any labels"""
        X_aug = np.zeros((4, 2))  # shape: (4, 2)
        theta = np.zeros(2)  # shape: (2,)
        y = np.array([0.0, 1.0, 1.0, 0.0])  # shape: (4,)
        mask = penalty_mask_for(2, True)

        loss, _ = logistic_objective(theta, X_aug, y, 0.0, mask)
        assert loss == pytest.approx(np.log(2.0))

    def test_realistic_bce_from_logits_is_stable(self):
        """
        probability-space log loss would be +inf here; the logit-space
        formulation stays finite; pins the whole reason for logaddexp
        """
        y = np.array([1.0, 0.0])  # shape: (2,)
        logits = np.array([-900.0, 900.0])  # maximally wrong, saturated

        loss = binary_cross_entropy_from_logits(y, logits)

        assert np.isfinite(loss)
        assert loss == pytest.approx(900.0, rel=1e-12)

    def test_realistic_l2_penalty_excludes_intercept(self):
        """changing only the intercept must not change the penalty"""
        X_aug = augment(np.ones((5, 2)))  # shape: (5, 3)
        y = np.zeros(5)  # shape: (5,)
        mask = penalty_mask_for(3, True)  # shape: (3,)

        theta_a = np.array([0.0, 1.0, 1.0])  # shape: (3,)
        theta_b = np.array([7.0, 1.0, 1.0])  # shape: (3,)

        pen_a = mse_objective(theta_a, X_aug, y, 1.0, mask)[0] - mse_objective(
            theta_a, X_aug, y, 0.0, mask
        )[0]
        pen_b = mse_objective(theta_b, X_aug, y, 1.0, mask)[0] - mse_objective(
            theta_b, X_aug, y, 0.0, mask
        )[0]

        assert pen_a == pytest.approx(pen_b)

    def test_realistic_gradients_match_finite_differences(self):
        """
        central differences as a cheap independent check on both hand-derived gradients
        test_oracle.py repeats it against torch.autograd at a much tighter tolerance
        """
        rng = np.random.default_rng(SEED)
        X_aug = augment(rng.normal(size=(40, 3)))  # shape: (40, 4)
        mask = penalty_mask_for(4, True)  # shape: (4,)
        theta = rng.normal(size=4)  # shape: (4,)
        eps = 1e-6

        y_reg = rng.normal(size=40)  # shape: (40,)
        y_cls = (rng.random(40) < 0.5).astype(np.float64)  # shape: (40,)

        for obj, y in ((mse_objective, y_reg), (logistic_objective, y_cls)):
            _, grad = obj(theta, X_aug, y, 0.3, mask)  # shape: (4,)
            numeric = np.empty_like(grad)  # shape: (4,)
            for j in range(theta.shape[0]):
                step = np.zeros_like(theta)
                step[j] = eps
                hi = obj(theta + step, X_aug, y, 0.3, mask)[0]
                lo = obj(theta - step, X_aug, y, 0.3, mask)[0]
                numeric[j] = (hi - lo) / (2.0 * eps)
            np.testing.assert_allclose(grad, numeric, rtol=1e-6, atol=1e-8)

    def test_edge_rejects_shape_and_domain_errors(self):
        X_aug = np.ones((5, 2))
        mask = penalty_mask_for(2, True)
        with pytest.raises(ValueError, match="columns"):
            mse_objective(np.zeros(3), X_aug, np.zeros(5), 0.0, mask)
        with pytest.raises(ValueError, match="rows"):
            mse_objective(np.zeros(2), X_aug, np.zeros(4), 0.0, mask)
        with pytest.raises(ValueError, match="l2 must be"):
            mse_objective(np.zeros(2), X_aug, np.zeros(5), -1.0, mask)
        with pytest.raises(ValueError, match="only 0 and 1"):
            logistic_objective(
                np.zeros(2), X_aug, np.full(5, 0.5), 0.0, mask
            )


# ---------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------
class TestGradientDescent:
    def test_trivial_quadratic_reaches_known_minimum(self):
        """f(t) = (t - 3)^2, minimum at 3"""

        def objective(theta):
            return float((theta[0] - 3.0) ** 2), np.array(
                [2.0 * (theta[0] - 3.0)]
            )

        res = gradient_descent(objective, np.zeros(1), lr=0.1, max_iter=500)

        assert res.theta[0] == pytest.approx(3.0, abs=1e-6)
        assert res.converged

    def test_realistic_loss_decreases_monotonically(self):
        X, y, _ = make_regression()
        X_aug = augment(X)  # shape: (200, 6)
        mask = penalty_mask_for(6, True)

        res = gradient_descent(
            lambda t: mse_objective(t, X_aug, y, 0.0, mask),
            np.zeros(6),
            lr=0.1,
            max_iter=500,
            record_every=1,
        )
        losses = np.array(res.history["loss"])  # shape: (m,)
        assert np.all(np.diff(losses) <= 1e-12)

    def test_realistic_history_and_reason_recorded(self):
        def objective(theta):
            return float(theta @ theta), 2.0 * theta

        res = gradient_descent(objective, np.ones(3), lr=0.1, max_iter=300)

        assert res.reason in {"grad_tol", "loss_tol", "max_iter"}
        assert len(res.history["loss"]) == len(res.history["iter"])
        assert res.grad_norm < 1e-7

    def test_edge_raises_on_divergence(self):
        """lr above the stability bound must raise, not return nan"""

        def objective(theta):
            return float(theta @ theta), 2.0 * theta

        with pytest.raises(RuntimeError, match="non-finite"):
            gradient_descent(objective, np.ones(2), lr=50.0, max_iter=500)

    def test_edge_rejects_bad_arguments(self):
        def objective(theta):
            return 0.0, np.zeros_like(theta)

        with pytest.raises(ValueError, match="lr must be"):
            gradient_descent(objective, np.zeros(2), lr=0.0)
        with pytest.raises(ValueError, match="max_iter"):
            gradient_descent(objective, np.zeros(2), max_iter=0)

    def test_edge_detects_wrong_gradient_shape(self):
        def bad_objective(theta):
            return 0.0, np.zeros(theta.shape[0] + 1)

        with pytest.raises(ValueError, match="gradient of shape"):
            gradient_descent(bad_objective, np.zeros(3))


# ---------------------------------------------------------------------
# LinearRegression
# ---------------------------------------------------------------------
class TestLinearRegression:
    def test_trivial_recovers_exact_line(self):
        """noiseless y = 2x + 1 must be fit exactly"""
        X = np.array([[0.0], [1.0], [2.0], [3.0]])  # shape: (4, 1)
        y = 2.0 * X.ravel() + 1.0  # shape: (4,)

        model = LinearRegression().fit(X, y)

        assert model.coef_[0] == pytest.approx(2.0)
        assert model.intercept_ == pytest.approx(1.0)
        assert model.sigma_sq_ == pytest.approx(0.0, abs=1e-20)
        # zero residual variance has no density; reported as None
        assert model.log_likelihood_ is None

    def test_realistic_recovers_known_weights(self):
        X, y, w_true = make_regression(n=5_000, d=4, noise=0.5)
        model = LinearRegression().fit(X, y)
        np.testing.assert_allclose(model.coef_, w_true, atol=0.05)
        assert model.intercept_ == pytest.approx(2.0, abs=0.05)

    def test_realistic_gd_matches_closed_form(self):
        """
        gradient descent must land on the exact least-squares answer
        this proves the optimizer before I trust it on logistic regression, where there's no closed form to check
        """
        X, y, _ = make_regression(n=400, d=4)

        exact = LinearRegression(l2=0.1).fit(X, y, method="closed_form")
        iterative = LinearRegression(l2=0.1).fit(
            X, y, method="gd", lr=0.1, max_iter=20_000, grad_tol=1e-12
        )

        np.testing.assert_allclose(
            iterative.coef_, exact.coef_, rtol=1e-5, atol=1e-7
        )
        assert iterative.intercept_ == pytest.approx(
            exact.intercept_, abs=1e-7
        )

    def test_realistic_sigma_sq_is_residual_mle(self):
        """sigma_sq_ must equal mle_gaussian on the residuals"""
        from prob_stats import mle_gaussian

        X, y, _ = make_regression(n=300, d=3)
        model = LinearRegression().fit(X, y)

        residuals = y - model.predict(X)  # shape: (300,)
        _, expected = mle_gaussian(residuals)

        assert model.sigma_sq_ == pytest.approx(expected, rel=1e-12)
        # 1/n, not 1/(n-1) -- the pass 1 convention, carried forward
        assert model.sigma_sq_ == pytest.approx(
            np.var(residuals, ddof=0), rel=1e-12
        )

    def test_realistic_l2_shrinks_coefficients(self):
        X, y, _ = make_regression(n=100, d=10)
        norms = [
            np.linalg.norm(LinearRegression(l2=l).fit(X, y).coef_)
            for l in (0.0, 0.01, 0.1, 1.0, 10.0)
        ]
        assert norms == sorted(norms, reverse=True)

    def test_edge_raises_when_underdetermined(self):
        """n < d with no penalty: singular normal equations"""
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(5, 20))  # shape: (5, 20)
        y = rng.normal(size=5)  # shape: (5,)

        with pytest.raises(ValueError, match="singular"):
            LinearRegression(l2=0.0).fit(X, y)

        # any penalty makes it solvable -- regularization as a numerical
        # fix, not only a generalization device
        model = LinearRegression(l2=0.1).fit(X, y)
        assert model.coef_.shape == (20,)
        assert np.all(np.isfinite(model.coef_))

    def test_edge_predict_before_fit_and_wrong_width(self):
        model = LinearRegression()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict(np.ones((3, 2)))

        X, y, _ = make_regression(n=50, d=3)
        model.fit(X, y)
        with pytest.raises(ValueError, match="fitted on 3"):
            model.predict(np.ones((5, 7)))


# ---------------------------------------------------------------------
# LogisticRegression
# ---------------------------------------------------------------------
class TestLogisticRegression:
    def test_trivial_separates_two_clusters(self):
        X = np.array([[-3.0], [-2.5], [2.5], [3.0]])  # shape: (4, 1)
        y = np.array([0.0, 0.0, 1.0, 1.0])  # shape: (4,)

        model = LogisticRegression(l2=0.01).fit(X, y)

        assert accuracy(y, model.predict_proba(X)) == 1.0
        assert model.coef_[0] > 0.0

    def test_realistic_recovers_weight_direction(self):
        """
        the size of the weights depends on the penalty, but their direction
        must line up with the weights that generated the data
        """
        X, y, w_true = make_classification(n=4_000, d=4, scale=1.0)
        model = LogisticRegression(l2=1e-4).fit(
            X, y, lr=0.5, max_iter=20_000
        )

        cos = float(
            model.coef_ @ w_true
            / (np.linalg.norm(model.coef_) * np.linalg.norm(w_true))
        )
        assert cos > 0.95

    def test_realistic_converges_to_stationary_point(self):
        """
        the honest convergence check: a small gradient norm; A small
        loss change can just mean a small step
        """
        X, y, _ = make_classification(n=500, d=5)
        model = LogisticRegression(l2=0.01).fit(
            X, y, lr=0.5, max_iter=20_000, grad_tol=1e-10
        )
        assert model.optimize_result_.converged
        assert model.optimize_result_.grad_norm < 1e-9

    def test_realistic_log_loss_attribute_matches_metric(self):
        X, y, _ = make_classification(n=400, d=4)
        model = LogisticRegression(l2=0.01).fit(X, y)

        from_metric = log_loss(y, model.decision_function(X))
        assert model.log_loss_ == pytest.approx(from_metric, rel=1e-8)

    def test_edge_rejects_non_binary_labels(self):
        X, _, _ = make_classification(n=50, d=3)
        with pytest.raises(ValueError, match="only 0 and 1"):
            LogisticRegression().fit(X, np.full(50, 0.5))

    def test_edge_threshold_shifts_predictions(self):
        X, y, _ = make_classification(n=300, d=3)
        model = LogisticRegression(l2=0.01).fit(X, y)

        n_pos_low = model.predict(X, threshold=0.2).sum()
        n_pos_high = model.predict(X, threshold=0.8).sum()
        assert n_pos_low >= n_pos_high

        with pytest.raises(ValueError, match="threshold"):
            model.predict(X, threshold=1.0)


# ---------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------
class TestMetrics:
    def test_trivial_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])  # shape: (3,)
        assert rmse(y, y) == pytest.approx(0.0)
        assert r2_score(y, y) == pytest.approx(1.0)

        labels = np.array([0.0, 1.0, 1.0])  # shape: (3,)
        probs = np.array([0.01, 0.99, 0.99])  # shape: (3,)
        assert accuracy(labels, probs) == 1.0
        assert roc_auc(labels, probs) == pytest.approx(1.0)

    def test_trivial_auc_of_random_scores_is_half(self):
        """all-tied scores give AUC exactly 0.5 under average ranks"""
        y = np.array([0.0, 1.0, 0.0, 1.0])  # shape: (4,)
        tied = np.full(4, 0.5)  # shape: (4,)
        assert roc_auc(y, tied) == pytest.approx(0.5)

    def test_realistic_auc_invariant_to_monotone_transform(self):
        """
        the defining property, and the reason AUC cannot detect
        miscalibration
        """
        rng = np.random.default_rng(SEED)
        y = (rng.random(400) < 0.4).astype(np.float64)  # shape: (400,)
        p = rng.random(400)  # shape: (400,)

        a = roc_auc(y, p)
        b = roc_auc(y, np.log(p / (1.0 - p)))  # logit transform
        c = roc_auc(y, p**3)  # another increasing map

        assert a == pytest.approx(b, rel=1e-12)
        assert a == pytest.approx(c, rel=1e-12)

    def test_realistic_average_ranks_handles_ties(self):
        values = np.array([10.0, 20.0, 20.0, 30.0])  # shape: (4,)
        np.testing.assert_allclose(
            _average_ranks(values), [1.0, 2.5, 2.5, 4.0]
        )

    def test_realistic_log_loss_is_proper(self):
        """
        A proper scoring rule is minimized by reporting the truth
        with a true rate of 0.7, the logit of 0.7 must beat any other
        """
        rng = np.random.default_rng(SEED)
        y = (rng.random(20_000) < 0.7).astype(np.float64)  # shape: (20000,)

        def loss_at(p):
            logit = np.log(p / (1.0 - p))
            return log_loss(y, np.full(y.shape[0], logit))

        best = loss_at(0.7)
        for p in (0.5, 0.6, 0.65, 0.75, 0.8, 0.9):
            assert loss_at(p) > best

    def test_realistic_calibration_detects_overconfidence(self):
        """
        inflating confidence leaves AUC untouched and raises ECE
        this is the pair of facts that justifies reporting both
        """
        rng = np.random.default_rng(SEED)
        p_true = rng.uniform(0.1, 0.9, size=5_000)  # shape: (5000,)
        y = (rng.random(5_000) < p_true).astype(np.float64)  # shape: (5000,)

        # push every probability toward its nearest extreme
        p_over = np.clip(p_true + 0.35 * np.sign(p_true - 0.5), 0.0, 1.0)

        assert expected_calibration_error(y, p_over) > (
            expected_calibration_error(y, p_true) + 0.1
        )
        assert roc_auc(y, p_over) == pytest.approx(
            roc_auc(y, p_true), abs=0.02
        )

    def test_realistic_brier_bounded(self):
        rng = np.random.default_rng(SEED)
        y = (rng.random(500) < 0.5).astype(np.float64)
        p = rng.random(500)
        assert 0.0 <= brier_score(y, p) <= 1.0

    def test_edge_r2_negative_on_bad_predictions(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])  # shape: (4,)
        bad = np.array([10.0, 10.0, 10.0, 10.0])  # shape: (4,)
        assert r2_score(y, bad) < 0.0

    def test_edge_r2_undefined_for_constant_target(self):
        with pytest.raises(ValueError, match="zero variance"):
            r2_score(np.full(5, 3.0), np.arange(5.0))

    def test_edge_auc_undefined_for_single_class(self):
        with pytest.raises(ValueError, match="one class"):
            roc_auc(np.zeros(10), np.random.default_rng(0).random(10))

    def test_edge_reliability_curve_marks_empty_bins(self):
        y = np.array([0.0, 1.0])  # shape: (2,)
        p = np.array([0.05, 0.95])  # shape: (2,)

        curve = reliability_curve(y, p, n_bins=10)

        assert curve["count"].sum() == 2
        assert np.count_nonzero(curve["count"] == 0) == 8
        assert np.isnan(curve["mean_observed"][5])


# ---------------------------------------------------------------------
# splits
# ---------------------------------------------------------------------
class TestSplits:
    def test_trivial_partition_is_exact(self):
        tr, va, te = train_val_test_split(100, 0.2, 0.2, seed=SEED)

        all_idx = np.concatenate([tr, va, te])
        assert all_idx.shape[0] == 100
        np.testing.assert_array_equal(np.sort(all_idx), np.arange(100))
        assert len(set(tr) & set(va)) == 0
        assert len(set(tr) & set(te)) == 0
        assert len(set(va) & set(te)) == 0

    def test_trivial_seed_reproduces_split(self):
        a = train_val_test_split(200, seed=7)
        b = train_val_test_split(200, seed=7)
        c = train_val_test_split(200, seed=8)

        for x, y in zip(a, b):
            np.testing.assert_array_equal(x, y)
        assert not np.array_equal(a[0], c[0])

    def test_realistic_stratification_preserves_rate(self):
        rng = np.random.default_rng(SEED)
        y = (rng.random(1_000) < 0.2).astype(np.float64)  # shape: (1000,)

        tr, va, te = train_val_test_split(1_000, stratify=y, seed=SEED)

        for idx in (tr, va, te):
            assert y[idx].mean() == pytest.approx(0.2, abs=0.03)

    def test_realistic_standardizer_fits_on_train_only(self):
        rng = np.random.default_rng(SEED)
        X = rng.normal(loc=50.0, scale=10.0, size=(300, 4))  # shape: (300,4)
        tr, _, te = train_val_test_split(300, seed=SEED)

        scaler = Standardizer().fit(X[tr])
        X_tr = scaler.transform(X[tr])  # shape: (m, 4)
        X_te = scaler.transform(X[te])  # shape: (k, 4)

        # train is exactly standardized; test is not, and must not be
        np.testing.assert_allclose(X_tr.mean(axis=0), 0.0, atol=1e-12)
        np.testing.assert_allclose(X_tr.std(axis=0, ddof=0), 1.0, atol=1e-12)
        assert not np.allclose(X_te.mean(axis=0), 0.0, atol=1e-6)

    def test_edge_standardizer_requires_fit(self):
        scaler = Standardizer()
        with pytest.raises(RuntimeError, match="not fitted"):
            scaler.transform(np.ones((3, 2)))
        assert not hasattr(scaler, "fit_transform")

    def test_edge_constant_feature_gets_unit_scale(self):
        X = np.column_stack([np.ones(20), np.arange(20.0)])  # shape: (20, 2)
        scaler = Standardizer().fit(X)

        assert scaler.n_constant_ == 1
        assert scaler.scale_[0] == 1.0
        out = scaler.transform(X)  # shape: (20, 2)
        assert np.all(np.isfinite(out))
        np.testing.assert_allclose(out[:, 0], 0.0)

    def test_edge_rejects_impossible_fractions(self):
        with pytest.raises(ValueError, match="must be < 1"):
            train_val_test_split(100, val_frac=0.6, test_frac=0.5)
        with pytest.raises(ValueError, match="n_samples"):
            train_val_test_split(2)


# ---------------------------------------------------------------------
# adversarial / stress
# ---------------------------------------------------------------------
class TestAdversarial:
    def test_collinear_features_need_a_penalty(self):
        """exactly duplicated columns make the Gram matrix singular"""
        rng = np.random.default_rng(SEED)
        x = rng.normal(size=(100, 1))  # shape: (100, 1)
        X = np.hstack([x, x, rng.normal(size=(100, 1))])  # shape: (100, 3)
        y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=100)  # shape: (100,)

        with pytest.raises(ValueError, match="singular"):
            LinearRegression(l2=0.0).fit(X, y)

        model = LinearRegression(l2=0.01).fit(X, y)
        assert np.all(np.isfinite(model.coef_))
        # the penalty splits the shared signal across the twins
        assert model.coef_[0] == pytest.approx(model.coef_[1], rel=1e-6)

    def test_extreme_feature_scales_break_gd_but_not_closed_form(self):
        """
        one feature at 1e6 inflates the largest eigenvalue of X^T X by 1e12, so any usable lr diverges
        standardizing fixes it; the closed form never cared
        """
        rng = np.random.default_rng(SEED)
        X = np.column_stack(
            [rng.normal(size=200), rng.normal(scale=1e6, size=200)]
        )  # shape: (200, 2)
        y = X[:, 0] + 1e-6 * X[:, 1] + rng.normal(scale=0.1, size=200)

        with pytest.raises(RuntimeError, match="non-finite"):
            LinearRegression().fit(X, y, method="gd", lr=0.1, max_iter=200)

        # closed form is scale-invariant in the sense that matters
        exact = LinearRegression().fit(X, y, method="closed_form")
        assert np.all(np.isfinite(exact.coef_))

        # standardizing restores gradient descent
        scaler = Standardizer().fit(X)
        gd = LinearRegression().fit(
            scaler.transform(X), y, method="gd", lr=0.1, max_iter=5_000
        )
        assert np.all(np.isfinite(gd.coef_))
        assert gd.optimize_result_.converged or (
            gd.optimize_result_.grad_norm < 1e-4
        )

    def test_separable_data_has_no_finite_mle(self):
        """
        perfectly separable classes: without a penalty the weight keeps growing until max_iter stops it
        any penalty gives the optimum a finite home
        """
        X = np.array([[-5.0], [-4.0], [4.0], [5.0]])  # shape: (4, 1)
        y = np.array([0.0, 0.0, 1.0, 1.0])  # shape: (4,)

        unpenalized = LogisticRegression(l2=0.0).fit(
            X, y, lr=0.5, max_iter=5_000
        )
        penalized = LogisticRegression(l2=0.1).fit(
            X, y, lr=0.5, max_iter=5_000
        )

        assert not unpenalized.optimize_result_.converged
        assert abs(unpenalized.coef_[0]) > 3.0 * abs(penalized.coef_[0])
        assert penalized.optimize_result_.converged

        # the divergence is only logarithmic (2.5 at 5k steps, 3.1 at 50k), so the
        # real signature is "more steps, bigger weight" vs "penalized stays put"
        longer = LogisticRegression(l2=0.0).fit(X, y, lr=0.5, max_iter=50_000)
        assert abs(longer.coef_[0]) > abs(unpenalized.coef_[0]) + 0.3
        penalized_longer = LogisticRegression(l2=0.1).fit(X, y, lr=0.5, max_iter=50_000)
        assert penalized_longer.coef_[0] == pytest.approx(penalized.coef_[0], rel=1e-6)

        # both classify perfectly; only one has usable probabilities
        assert accuracy(y, unpenalized.predict_proba(X)) == 1.0
        assert np.max(unpenalized.predict_proba(X)) > 0.999

    def test_single_class_labels(self):
        """all-one-class training: the intercept absorbs everything"""
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(50, 3))  # shape: (50, 3)
        y = np.ones(50)  # shape: (50,)

        model = LogisticRegression(l2=0.1).fit(X, y, lr=0.5, max_iter=2_000)

        assert np.all(model.predict_proba(X) > 0.5)
        # AUC is undefined with one class, and must say so
        with pytest.raises(ValueError, match="one class"):
            roc_auc(y, model.predict_proba(X))

    def test_n_equals_one(self):
        """minimal viable fit; degenerate but must not crash"""
        X = np.array([[1.0, 2.0]])  # shape: (1, 2)
        y = np.array([3.0])  # shape: (1,)

        model = LinearRegression(l2=1.0).fit(X, y)
        assert np.all(np.isfinite(model.coef_))
        assert model.sigma_sq_ == pytest.approx(0.0, abs=1e-12)

    def test_huge_l2_drives_weights_to_zero(self):
        """the lambda -> infinity limit: an intercept-only model"""
        X, y, _ = make_regression(n=200, d=5)

        model = LinearRegression(l2=1e9).fit(X, y)

        np.testing.assert_allclose(model.coef_, 0.0, atol=1e-6)
        assert model.intercept_ == pytest.approx(y.mean(), abs=1e-3)

    def test_nan_and_inf_rejected_everywhere(self):
        """no non-finite value may enter any public entry point"""
        X = np.array([[1.0, 2.0], [np.nan, 4.0]])  # shape: (2, 2)
        y = np.array([1.0, 2.0])  # shape: (2,)

        with pytest.raises(ValueError, match="non-finite"):
            LinearRegression().fit(X, y)
        with pytest.raises(ValueError, match="non-finite"):
            LogisticRegression().fit(X, np.array([0.0, 1.0]))
        with pytest.raises(ValueError, match="non-finite"):
            rmse(np.array([1.0, np.inf]), np.array([1.0, 2.0]))

    def test_duplicate_rows_do_not_break_anything(self):
        """A pathological but legal dataset: every row identical"""
        X = np.tile(np.array([[1.0, 2.0]]), (50, 1))  # shape: (50, 2)
        y = np.full(50, 5.0)  # shape: (50,)

        model = LinearRegression(l2=0.1).fit(X, y)
        assert np.all(np.isfinite(model.predict(X)))
