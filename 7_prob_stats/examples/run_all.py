"""
a printed walk through pass 1: MLE, MAP, bayes updates and the information identity
usage: make -f 7_prob_stats/Makefile demo
"""

from __future__ import annotations

import numpy as np

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


def rule(title: str) -> None:
    """print a section header"""
    print(f"\n{'=' * 24}\n{title}\n{'=' * 68}")


def demo_gaussian_mle() -> None:
    rule("1. Gaussian MLE -- recovering parameters from data")

    rng = np.random.default_rng(SEED)
    true_mu, true_sigma = 5.0, 2.0
    data = rng.normal(true_mu, true_sigma, size=10_000)  # shape: (10000,)

    print(f"data shape : {data.shape}")
    print(f"true (mu, sigma^2) : ({true_mu}, {true_sigma**2})")

    mu_hat, sigma_sq_hat = mle_gaussian(data)
    print(f"  MLE  (mu_hat, sigma_sq) : ({mu_hat:.4f}, {sigma_sq_hat:.4f})")

    print(f"\n  numpy var ddof=0 (= MLE)  : {np.var(data, ddof=0):.6f}")
    print(f"  numpy var ddof=1 (Bessel) : {np.var(data, ddof=1):.6f}")
    print("  ^ the 1/n vs 1/(n-1) gap: MLE is the biased one, by design.")

    ll = log_likelihood_gaussian(data, mu_hat, sigma_sq_hat)
    print(f"\n  log-likelihood at MLE  : {ll:.4f} nats (n = {data.shape[0]})")
    print("  Perturbing either parameter must lower it:")
    for label, m, v in [
        ("mu + 0.5", mu_hat + 0.5, sigma_sq_hat),
        ("mu - 0.5", mu_hat - 0.5, sigma_sq_hat),
        ("var x 1.5", mu_hat, sigma_sq_hat * 1.5),
        ("var x 0.5", mu_hat, sigma_sq_hat * 0.5),
    ]:
        print(
            f"{label:<10} -> {log_likelihood_gaussian(data, m, v):.4f}"
        )


def demo_bernoulli_mle_vs_map() -> None:
    rule("2. Bernoulli MLE vs MAP -- what a prior buys you")

    data = np.zeros(5)  # 5 tails, 0 heads; shape: (5,)
    print(f"  data shape : {data.shape}")
    print(f" data : {data.astype(int).tolist()}")

    print(f"\n  MLE p_hat: {mle_bernoulli(data):.6f}")
    print("  ^ 'heads is impossible', on the strength of 5 observations.")
    print("Any downstream log(p_hat) is -inf. This is overfitting.\n")

    for a, b in [(1.0, 1.0), (2.0, 2.0), (5.0, 5.0), (20.0, 20.0)]:
        p_map = map_bernoulli(data, alpha=a, beta=b)
        print(f"  MAP Beta({a:>4}, {b:>4})      : {p_map:.6f}")
    print("  ^ alpha=beta=1 is the uniform prior and reproduces the MLE.")
    print("    Stronger priors pull harder toward 0.5.")

    print("\n  Same priors, but now with 500 observations at p = 0.2:")
    rng = np.random.default_rng(SEED)
    big = (rng.random(500) < 0.2).astype(np.float64)  # shape: (500,)
    print(f" MLE : {mle_bernoulli(big):.6f}")
    print(f"  MAP Beta(20, 20)  : {map_bernoulli(big, 20.0, 20.0):.6f}")
    print("  ^ the likelihood now dominates; the prior has washed out.")


def demo_bayes_update() -> None:
    rule("3. Bayes' rule -- sequential belief update over 3 hypotheses")

    # three coins: biased tails, fair, biased heads
    hypotheses = ["p=0.25", "p=0.50", "p=0.75"]
    p_heads = np.array([0.25, 0.50, 0.75])  # shape: (3,)
    belief = np.array([1 / 3, 1 / 3, 1 / 3])  # shape: (3,)

    print(f"  hypotheses      : {hypotheses}")
    print(f"  prior shape      : {belief.shape}")
    print(f"  prior : {np.round(belief, 4).tolist()}")

    observations = [1, 1, 1, 0, 1]  # H H H T H
    for step, outcome in enumerate(observations, start=1):
        # P(outcome | hypothesis_k) for each k
        likelihood = p_heads if outcome == 1 else (1.0 - p_heads)  # shape: (3,)
        belief = bayes_update(belief, likelihood)  # shape: (3,)
        label = "H" if outcome == 1 else "T"
        print(
            f"  after obs {step} ({label})          : "
            f"{np.round(belief, 4).tolist()}  (sums to {belief.sum():.6f})"
        )

    print(f"\n  most probable hypothesis  : {hypotheses[int(belief.argmax())]}")


def demo_information() -> None:
    rule("4. Entropy, cross-entropy, KL divergence")

    uniform = np.array([0.25, 0.25, 0.25, 0.25])  # shape: (4,)
    peaked = np.array([0.97, 0.01, 0.01, 0.01])  # shape: (4,)
    point = np.array([1.00, 0.00, 0.00, 0.00])  # shape: (4,)

    print(f"  all distributions shape   : {uniform.shape}")
    print(f"\n  H(uniform)                : {entropy(uniform):.6f} nats"
          f"  (max = log 4 = {np.log(4):.6f})")
    print(f"  H(peaked)  : {entropy(peaked):.6f} nats")
    print(f"  H(point mass) : {entropy(point):.6f} nats  (min = 0)")
    print("  ^ exactly 0.0, not -1e-10: zeros are masked, not epsilon-ed.")

    p = np.array([0.5, 0.5])  # shape: (2,)
    q = np.array([0.99, 0.01])  # shape: (2,)
    fwd, rev = kl_divergence(p, q), kl_divergence(q, p)
    print(f"\n  KL(P || Q)  : {fwd:.6f} nats")
    print(f"  KL(Q || P)  : {rev:.6f} nats")
    print(f"  ratio : {fwd / rev:.4f}x -- not symmetric")

    print("\n  Support mismatch:")
    print(f" KL([.5,.5] || [1,0]) : {kl_divergence(p, point[:2])}")
    print("  ^ +inf is correct, not an error: Q ruled out what P allows.")

    print("\n  One-hot target (classification):")
    one_hot = np.array([0.0, 1.0, 0.0])  # shape: (3,)
    pred = np.array([0.2, 0.5, 0.3])  # shape: (3,)
    print(f" H(P)  : {entropy(one_hot):.6f}")
    print(f" H(P, Q) : {cross_entropy(one_hot, pred):.6f}")
    print(f"KL(P || Q)   : {kl_divergence(one_hot, pred):.6f}")
    print("  ^ H(P) = 0, so cross-entropy loss IS KL divergence here.")

    rule("5. Identity check: H(P, Q) = H(P) + KL(P || Q)")
    verify_relationship(
        np.array([0.4, 0.4, 0.2]), np.array([0.3, 0.5, 0.2])
    )


def main() -> None:
    print("prob_stats -- demonstration of all public functions")
    print(f"numpy {np.__version__}, seed {SEED}")

    demo_gaussian_mle()
    demo_bernoulli_mle_vs_map()
    demo_bayes_update()
    demo_information()

    print(f"\n{'=' * 68}\nDone. Run the test suite with:")
    print("  pytest 7_prob_stats/tests -v")
    print("=" * 14)


if __name__ == "__main__":
    main()
