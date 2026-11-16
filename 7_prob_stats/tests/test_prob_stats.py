"""pass 1 tests: distributions, bayes, information"""

from __future__ import annotations

import math

import numpy as np
import pytest

from prob_stats import (
    bayes_update,
    cross_entropy,
    entropy,
    kl_divergence,
    log_likelihood_gaussian,
    map_bernoulli,
    mle_bernoulli,
    mle_gaussian,
    verify_relationship,
)

SEED = 42


# ------------------
# mle_gaussian
# -----
class TestMLEGaussian:
    def test_trivial_single_observation(self):
        """n=1: mean is the point itself, variance is exactly zero"""
        mu, var = mle_gaussian(np.array([7.0]))
        assert mu == pytest.approx(7.0)
        assert var == pytest.approx(0.0)

    def test_trivial_hand_checkable(self):
        """[1,2,3]: mean 2, biased variance (1+0+1)/3 = 2/3"""
        mu, var = mle_gaussian(np.array([1.0, 2.0, 3.0]))
        assert mu == pytest.approx(2.0)
        assert var == pytest.approx(2.0 / 3.0)

    def test_realistic_recovers_known_params(self):
        """generate from N(5, 4); MLE should recover ~5 and ~4"""
        rng = np.random.default_rng(SEED)
        data = rng.normal(loc=5.0, scale=2.0, size=10_000)  # shape: (10000,)

        mu, var = mle_gaussian(data)

        assert mu == pytest.approx(5.0, abs=0.1)
        assert var == pytest.approx(4.0, abs=0.2)

    def test_realistic_matches_numpy_ddof_zero(self):
        """the MLE variance is np.var(ddof=0), not ddof=1"""
        rng = np.random.default_rng(SEED)
        data = rng.normal(0.0, 3.0, size=500)  # shape: (500,)

        _, var = mle_gaussian(data)

        assert var == pytest.approx(np.var(data, ddof=0), rel=1e-12)
        assert var != pytest.approx(np.var(data, ddof=1), rel=1e-12)

    def test_edge_mle_is_actually_the_maximum(self):
        rng = np.random.default_rng(SEED)
        data = rng.normal(2.0, 1.5, size=200)  # shape: (200,)

        mu, var = mle_gaussian(data)
        ll_best = log_likelihood_gaussian(data, mu, var)

        for d_mu in (-0.25, 0.25):
            assert log_likelihood_gaussian(data, mu + d_mu, var) < ll_best
        for factor in (0.6, 1.6):
            assert log_likelihood_gaussian(data, mu, var * factor) < ll_best

    def test_edge_rejects_bad_input(self):
        with pytest.raises(ValueError, match="at least one observation"):
            mle_gaussian(np.array([]))
        with pytest.raises(ValueError, match="non-finite"):
            mle_gaussian(np.array([1.0, np.nan, 3.0]))
        with pytest.raises(ValueError, match="1-D"):
            mle_gaussian(np.ones((3, 2)))


# ----
# log_likelihood_gaussian
# -------------
class TestLogLikelihoodGaussian:
    def test_trivial_standard_normal_at_zero(self):
        """one observation at the mean of N(0,1): ll = -0.5*log(2*pi)"""
        ll = log_likelihood_gaussian(np.array([0.0]), mu=0.0, sigma_sq=1.0)
        assert ll == pytest.approx(-0.5 * math.log(2.0 * math.pi))

    def test_realistic_sums_over_observations(self):
    
        data = np.array([0.0, 0.0, 0.0])  # shape: (3,)
        single = log_likelihood_gaussian(data[:1], mu=0.0, sigma_sq=1.0)
        triple = log_likelihood_gaussian(data, mu=0.0, sigma_sq=1.0)
        assert triple == pytest.approx(3.0 * single)

    def test_realistic_can_be_positive(self):
       
        ll = log_likelihood_gaussian(np.array([0.0]), mu=0.0, sigma_sq=0.001)
        assert ll > 0.0

    def test_edge_rejects_nonpositive_variance(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="strictly positive"):
                log_likelihood_gaussian(np.array([1.0]), mu=0.0, sigma_sq=bad)

    def test_edge_large_n_does_not_underflow(self):
       
        rng = np.random.default_rng(SEED)
        data = rng.normal(0.0, 1.0, size=100_000)  # shape: (100000,)

        ll = log_likelihood_gaussian(data, mu=0.0, sigma_sq=1.0)

        assert np.isfinite(ll)
        assert ll < 0.0  # ~ -141_900 nats


# ------
# mle_bernoulli
# -
class TestMLEBernoulli:
    def test_trivial_all_heads(self):
        assert mle_bernoulli(np.ones(10)) == 1.0

    def test_trivial_half(self):
        assert mle_bernoulli(np.array([1.0, 0.0, 1.0, 0.0])) == 0.5

    def test_realistic_recovers_known_p(self):
        rng = np.random.default_rng(SEED)
        data = (rng.random(20_000) < 0.3).astype(np.float64)  # shape: (20000,)
        assert mle_bernoulli(data) == pytest.approx(0.3, abs=0.02)

    def test_edge_all_tails_is_degenerate_zero(self):
       
        assert mle_bernoulli(np.zeros(5)) == 0.0

    def test_edge_rejects_non_binary(self):
        with pytest.raises(ValueError, match="only 0 and 1"):
            mle_bernoulli(np.array([0.0, 0.7, 1.0]))
        with pytest.raises(ValueError, match="only 0 and 1"):
            mle_bernoulli(np.array([0.0, 2.0]))


# -------
# bayes_update
# -------------------
class TestBayesUpdate:
    def test_trivial_uniform_prior_passes_likelihood_through(self):
        """with a flat prior the posterior is the normalized likelihood"""
        prior = np.array([0.5, 0.5])  # shape: (2,)
        likelihood = np.array([0.8, 0.4])  # shape: (2,)

        posterior = bayes_update(prior, likelihood)

        assert posterior.shape == (2,)
        np.testing.assert_allclose(posterior, [2.0 / 3.0, 1.0 / 3.0])

    def test_trivial_always_normalized(self):
        rng = np.random.default_rng(SEED)
        prior = np.array([0.2, 0.3, 0.5])  # shape: (3,)
        likelihood = rng.random(3)  # shape: (3,), NOT normalized

        posterior = bayes_update(prior, likelihood)

        assert posterior.sum() == pytest.approx(1.0)
        assert np.all(posterior >= 0.0)

    def test_realistic_sequential_equals_batch(self):
        
        prior = np.array([0.5, 0.5])  # shape: (2,)
        lik_a = np.array([0.7, 0.2])  # shape: (2,)
        lik_b = np.array([0.6, 0.9])  # shape: (2,)

        sequential = bayes_update(bayes_update(prior, lik_a), lik_b)
        batch = bayes_update(prior, lik_a * lik_b)

        np.testing.assert_allclose(sequential, batch)

    def test_edge_zero_prior_hypothesis_stays_dead(self):
        
        prior = np.array([0.0, 1.0])  # shape: (2,)
        likelihood = np.array([0.99, 0.01])  # shape: (2,)

        posterior = bayes_update(prior, likelihood)

        assert posterior[0] == 0.0
        assert posterior[1] == 1.0

    def test_edge_raises_on_zero_evidence(self):
        """every hypothesis gives the data zero likelihood: no posterior exists, so it must raise"""
        with pytest.raises(ValueError, match="sums to zero"):
            bayes_update(np.array([0.5, 0.5]), np.array([0.0, 0.0]))

    def test_edge_rejects_shape_mismatch_and_bad_prior(self):
        with pytest.raises(ValueError, match="same shape"):
            bayes_update(np.array([0.5, 0.5]), np.array([1.0, 1.0, 1.0]))
        with pytest.raises(ValueError, match="must sum to 1"):
            bayes_update(np.array([0.5, 0.9]), np.array([1.0, 1.0]))


# ------
# map_bernoulli
# --
class TestMAPBernoulli:
    def test_trivial_reduces_to_mle_with_uniform_prior(self):
        """alpha=beta=1 contributes zero pseudocounts: MAP == MLE"""
        data = np.array([1.0, 1.0, 0.0])  # 2 heads, 1 tail; shape: (3,)
        assert abs(
            map_bernoulli(data, alpha=1.0, beta=1.0) - mle_bernoulli(data)
        ) < 1e-9

    def test_trivial_hand_derived_value(self):
        
        result = map_bernoulli(np.zeros(5), alpha=2.0, beta=2.0)
        assert result == pytest.approx(1.0 / 7.0)

    def test_realistic_prior_shrinks_toward_its_mean(self):
        
        data = np.ones(8)  # 8 heads, 0 tails; MLE = 1.0; shape: (8,)

        mle = mle_bernoulli(data)
        weak = map_bernoulli(data, alpha=2.0, beta=2.0)
        strong = map_bernoulli(data, alpha=20.0, beta=20.0)

        assert mle == 1.0
        assert 0.5 < strong < weak < mle

    def test_realistic_prior_washes_out_with_data(self):
        
        rng = np.random.default_rng(SEED)
        small = np.array([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=np.float64)  # shape: (10,), 7 of 10; a 5/5 draw would make MAP = MLE = 0.5 and gap 0
        large = (rng.random(100_000) < 0.7).astype(np.float64)  # shape: (100000,)

        gap_small = abs(
            map_bernoulli(small, 5.0, 5.0) - mle_bernoulli(small)
        )
        gap_large = abs(
            map_bernoulli(large, 5.0, 5.0) - mle_bernoulli(large)
        )

        assert gap_large < gap_small
        assert gap_large < 1e-4

    def test_edge_boundary_modes(self):
       
        assert map_bernoulli(np.ones(1), alpha=0.5, beta=0.5) == 1.0
        assert map_bernoulli(np.zeros(1), alpha=0.5, beta=0.5) == 0.0

    def test_edge_u_shaped_posterior_unreachable_with_data(self):
        # a U-shaped posterior needs a_post < 1 and b_post < 1, but every observation adds 1
        # to one of them, and as_binary rejects n = 0. so with tiny priors the mode lands on a boundary instead
        assert map_bernoulli(np.array([0.0]), alpha=0.3, beta=0.2) == 0.0  # Beta(0.3, 1.2), peak at 0
        assert map_bernoulli(np.array([1.0]), alpha=0.3, beta=0.2) == 1.0  # Beta(1.3, 0.2), peak at 1

    def test_edge_rejects_nonpositive_hyperparameters(self):
        for bad in (0.0, -1.0):
            with pytest.raises(ValueError, match="strictly positive"):
                map_bernoulli(np.ones(3), alpha=bad, beta=1.0)
            with pytest.raises(ValueError, match="strictly positive"):
                map_bernoulli(np.ones(3), alpha=1.0, beta=bad)


# -----
# entropy
# ---
class TestEntropy:
    def test_trivial_point_mass_is_zero(self):
        """no uncertainty, no entropy -- and exactly 0.0, not -1e-10"""
        h = entropy(np.array([1.0, 0.0, 0.0]))
        assert h == 0.0
        assert h >= 0.0  # fails under the p*log(p+eps) shortcut

    def test_trivial_uniform_is_log_k(self):
        for k in (2, 4, 10):
            p = np.full(k, 1.0 / k)  # shape: (k,)
            assert entropy(p) == pytest.approx(math.log(k))

    def test_realistic_uniform_is_maximum(self):
        uniform = np.array([0.25, 0.25, 0.25, 0.25])  # shape: (4,)
        peaked = np.array([0.97, 0.01, 0.01, 0.01])  # shape: (4,)
        assert entropy(uniform) > entropy(peaked)

    def test_realistic_monotone_in_concentration(self):
        """progressively concentrating mass monotonically lowers entropy"""
        dists = [
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([0.40, 0.30, 0.20, 0.10]),
            np.array([0.70, 0.20, 0.05, 0.05]),
            np.array([0.97, 0.01, 0.01, 0.01]),
        ]
        values = [entropy(d) for d in dists]
        assert values == sorted(values, reverse=True)

    def test_edge_zeros_contribute_exactly_nothing(self):
        """0 * log(0) = 0: padding with zeros must not change entropy"""
        base = np.array([0.5, 0.5])  # shape: (2,)
        padded = np.array([0.5, 0.5, 0.0, 0.0, 0.0])  # shape: (5,)
        assert entropy(padded) == pytest.approx(entropy(base), rel=1e-15)

    def test_edge_rejects_unnormalized_and_negative(self):
        with pytest.raises(ValueError, match="must sum to 1"):
            entropy(np.array([0.5, 0.2]))
        with pytest.raises(ValueError, match="negative"):
            entropy(np.array([1.5, -0.5]))


# -----
# cross_entropy
# -----------------
class TestCrossEntropy:
    def test_trivial_equals_entropy_when_identical(self):
        p = np.array([0.3, 0.3, 0.4])  # shape: (3,)
        assert cross_entropy(p, p) == pytest.approx(entropy(p))

    def test_trivial_one_hot_target(self):
        p = np.array([0.0, 1.0, 0.0])  # shape: (3,)
        q = np.array([0.1, 0.7, 0.2])  # shape: (3,)
        assert cross_entropy(p, q) == pytest.approx(-math.log(0.7))

    def test_realistic_bounded_below_by_entropy(self):
        """H(P,Q) >= H(P) for every Q -- the consequence of KL >= 0"""
        rng = np.random.default_rng(SEED)
        for _ in range(50):
            p = rng.dirichlet(np.ones(5))  # shape: (5,)
            q = rng.dirichlet(np.ones(5))  # shape: (5,)
            assert cross_entropy(p, q) >= entropy(p) - 1e-12

    def test_realistic_asymmetric(self):
        p = np.array([0.5, 0.5])  # shape: (2,)
        q = np.array([0.9, 0.1])  # shape: (2,)
        assert cross_entropy(p, q) != pytest.approx(cross_entropy(q, p))

    def test_edge_infinite_on_support_mismatch(self):
        """q rules out something p considers possible: infinite surprise"""
        p = np.array([0.5, 0.5])  # shape: (2,)
        q = np.array([1.0, 0.0])  # shape: (2,)
        assert cross_entropy(p, q) == np.inf

    def test_edge_finite_when_only_p_has_a_zero(self):
        """the reverse mismatch is harmless: p's zero masks q's value"""
        p = np.array([1.0, 0.0])  # shape: (2,)
        q = np.array([0.5, 0.5])  # shape: (2,)
        assert cross_entropy(p, q) == pytest.approx(-math.log(0.5))


# ----
# kl_divergence
# -----
class TestKLDivergence:
    def test_trivial_zero_when_identical(self):
        p = np.array([0.3, 0.3, 0.4])  # shape: (3,)
        assert abs(kl_divergence(p, p)) < 1e-9

    def test_trivial_nonnegative(self):
        rng = np.random.default_rng(SEED)
        for _ in range(100):
            p = rng.dirichlet(np.ones(4))  # shape: (4,)
            q = rng.dirichlet(np.ones(4))  # shape: (4,)
            assert kl_divergence(p, q) >= -1e-12  # gibbs' inequality

    def test_realistic_asymmetry_is_substantial(self):
        
        p = np.array([0.5, 0.5])  # shape: (2,)
        q = np.array([0.99, 0.01])  # shape: (2,)

        forward = kl_divergence(p, q)
        reverse = kl_divergence(q, p)

        assert forward > reverse
        assert forward / reverse > 2.0

    def test_realistic_increases_with_divergence(self):
        """moving q away from p monotonically increases KL"""
        p = np.array([0.5, 0.5])  # shape: (2,)
        qs = [
            np.array([0.5, 0.5]),
            np.array([0.6, 0.4]),
            np.array([0.8, 0.2]),
            np.array([0.95, 0.05]),
        ]
        values = [kl_divergence(p, q) for q in qs]
        assert values == sorted(values)

    def test_edge_infinite_on_support_mismatch(self):
        p = np.array([0.5, 0.5])  # shape: (2,)
        q = np.array([1.0, 0.0])  # shape: (2,)
        assert kl_divergence(p, q) == np.inf

    def test_edge_one_hot_p_makes_kl_equal_cross_entropy(self):

        p = np.array([0.0, 1.0, 0.0])  # shape: (3,)
        q = np.array([0.2, 0.5, 0.3])  # shape: (3,)

        assert entropy(p) == 0.0
        assert kl_divergence(p, q) == pytest.approx(cross_entropy(p, q))


# ----------
# verify_relationship
# -------
class TestVerifyRelationship:
    def test_trivial_identity_holds(self):
        p = np.array([0.4, 0.4, 0.2])  # shape: (3,)
        q = np.array([0.3, 0.5, 0.2])  # shape: (3,)
        assert verify_relationship(p, q, verbose=False) is True

    def test_realistic_identity_holds_at_machine_precision(self):
        
        p = np.array([0.4, 0.4, 0.2])  # shape: (3,)
        q = np.array([0.3, 0.5, 0.2])  # shape: (3,)
        assert abs(
            cross_entropy(p, q) - (entropy(p) + kl_divergence(p, q))
        ) < 1e-9

    def test_realistic_holds_across_random_pairs(self):
        rng = np.random.default_rng(SEED)
        for _ in range(200):
            k = int(rng.integers(2, 8))
            p = rng.dirichlet(np.ones(k))  # shape: (k,)
            q = rng.dirichlet(np.ones(k))  # shape: (k,)
            assert verify_relationship(p, q, atol=1e-10, verbose=False)

    def test_edge_holds_when_both_sides_infinite(self):
        p = np.array([0.5, 0.5])  # shape: (2,)
        q = np.array([1.0, 0.0])  # shape: (2,)
        assert verify_relationship(p, q, verbose=False) is True


# ---------
# PyTorch oracle
# ------
class TestTorchOracle:

    def test_log_likelihood_matches_torch_normal(self):
        torch = pytest.importorskip("torch")

        rng = np.random.default_rng(SEED)
        data = rng.normal(1.5, 2.0, size=1_000)  # shape: (1000,)
        mu, sigma_sq = 1.2, 3.5

        ours = log_likelihood_gaussian(data, mu, sigma_sq)

        dist = torch.distributions.Normal(
            loc=torch.tensor(mu, dtype=torch.float64),
            scale=torch.tensor(math.sqrt(sigma_sq), dtype=torch.float64),
        )
        theirs = (
            dist.log_prob(torch.tensor(data, dtype=torch.float64))
            .sum()
            .item()
        )

        assert ours == pytest.approx(theirs, rel=1e-12)

    def test_mle_gaussian_matches_torch_biased_moments(self):
        torch = pytest.importorskip("torch")

        rng = np.random.default_rng(SEED)
        data = rng.normal(-3.0, 1.2, size=2_000)  # shape: (2000,)
        t = torch.tensor(data, dtype=torch.float64)

        mu, var = mle_gaussian(data)

        assert mu == pytest.approx(t.mean().item(), rel=1e-12)
        # unbiased=False is the MLE; torch's default (True) is Bessel-
        # corrected and will NOT match -- the 1/n vs 1/(n-1) gap
        assert var == pytest.approx(
            t.var(unbiased=False).item(), rel=1e-12
        )
        assert var != pytest.approx(t.var().item(), rel=1e-12)

    def test_entropy_matches_torch_categorical(self):
        torch = pytest.importorskip("torch")

        rng = np.random.default_rng(SEED)
        for _ in range(20):
            p = rng.dirichlet(np.ones(6))  # shape: (6,)

            ours = entropy(p)
            theirs = (
                torch.distributions.Categorical(
                    probs=torch.tensor(p, dtype=torch.float64)
                )
                .entropy()
                .item()
            )

            assert ours == pytest.approx(theirs, rel=1e-12)

    def test_kl_matches_torch_kl_divergence(self):
        torch = pytest.importorskip("torch")

        rng = np.random.default_rng(SEED)
        for _ in range(20):
            p = rng.dirichlet(np.ones(5))  # shape: (5,)
            q = rng.dirichlet(np.ones(5))  # shape: (5,)

            ours = kl_divergence(p, q)
            theirs = torch.distributions.kl_divergence(
                torch.distributions.Categorical(
                    probs=torch.tensor(p, dtype=torch.float64)
                ),
                torch.distributions.Categorical(
                    probs=torch.tensor(q, dtype=torch.float64)
                ),
            ).item()

            assert ours == pytest.approx(theirs, rel=1e-12)

    def test_cross_entropy_matches_torch_classification_loss(self):
        
        torch = pytest.importorskip("torch")

        logits = torch.tensor([[1.2, -0.5, 2.1]], dtype=torch.float64)
        target_index = 2

        probs = torch.softmax(logits, dim=1).numpy().ravel()  # shape: (3,)
        one_hot = np.zeros(3)  # shape: (3,)
        one_hot[target_index] = 1.0

        ours = cross_entropy(one_hot, probs)
        theirs = torch.nn.functional.cross_entropy(
            logits, torch.tensor([target_index])
        ).item()

        assert ours == pytest.approx(theirs, rel=1e-12)
