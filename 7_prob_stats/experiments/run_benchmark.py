"""
how fit time scales with n and with d, for the closed form and both gradient-descent fits
usage: make -f 7_prob_stats/Makefile bench
"""



from __future__ import annotations

import time

import numpy as np

from experiments.common import (
    FIXED_LR,
    RESULTS_DIR,
    RunLogger,
    derive_seed,
    environment_record,
    get_pyplot,
    save_figure,
)
from prob_stats import LinearRegression, LogisticRegression, Standardizer, sigmoid

# grids; one variable moves per sweep; the other is pinned
N_GRID = [100, 300, 1_000, 3_000, 10_000, 30_000]
D_FIXED_FOR_N_SWEEP = 20

D_GRID = [2, 5, 10, 20, 50, 100, 200, 400]
N_FIXED_FOR_D_SWEEP = 5_000

# iteration budget for the benchmark only; much lower than the
# experiment's, because here we are timing cost per unit of work, not
# measuring a converged solution; held identical across all sizes so
# the comparison is fair
BENCH_MAX_ITER = 200
N_REPEATS = 3


def timed(fn, n_repeats: int = N_REPEATS) -> float:
    """
    fastest of n_repeats wall-clock runs; noise only adds time, so the min is the cleanest estimate
    """
    best = float("inf")
    for _ in range(n_repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def make_data(n: int, d: int, seed: int):
    """standardized X (n, d), regression y (n,) and labels (n,)"""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))  # shape: (n, d)
    X = Standardizer().fit(X).transform(X)  # shape: (n, d)

    w = rng.normal(size=d)  # shape: (d,)
    y_reg = X @ w + rng.normal(scale=0.5, size=n)  # shape: (n,)
    y_cls = (rng.random(n) < sigmoid(X @ w)).astype(np.float64)  # (n,)
    return X, y_reg, y_cls


def bench_one(log: RunLogger, sweep: str, n: int, d: int) -> None:
    """time all three fitters at one (n, d) and log three rows"""
    seed = derive_seed(hash(sweep) % 10_000, n, d)
    X, y_reg, y_cls = make_data(n, d, seed)

    cases = {
        "linear_closed_form": lambda: LinearRegression(l2=1e-3).fit(
            X, y_reg, method="closed_form"
        ),
        "linear_gd": lambda: LinearRegression(l2=1e-3).fit(
            X,
            y_reg,
            method="gd",
            lr=FIXED_LR,
            max_iter=BENCH_MAX_ITER,
            grad_tol=0.0,
        ),
        "logistic_gd": lambda: LogisticRegression(l2=1e-3).fit(
            X, y_cls, lr=FIXED_LR, max_iter=BENCH_MAX_ITER, grad_tol=0.0
        ),
    }

    for name, fn in cases.items():
        seconds = timed(fn)
        log.log(
            sweep=sweep,
            method=name,
            n=n,
            d=d,
            seed=seed,
            max_iter=BENCH_MAX_ITER,
            n_repeats=N_REPEATS,
            seconds=seconds,
        )
        print(f"    {name:<20} n={n:>6} d={d:>4}  {seconds * 1e3:>9.2f} ms")


def fit_exponent(sizes: np.ndarray, seconds: np.ndarray) -> float:
    """slope of log(time) vs log(size); nan with fewer than 3 usable points"""
    ok = (sizes > 0) & (seconds > 0) & np.isfinite(seconds)
    if int(ok.sum()) < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(sizes[ok]), np.log(seconds[ok]), deg=1)
    return float(slope)


def plot_sweep(log: RunLogger, sweep: str, axis: str, filename: str, title: str):
    """log-log timing plot with fitted exponents in the legend"""
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(7, 5))

    exponents: dict[str, float] = {}
    for method in ("linear_closed_form", "linear_gd", "logistic_gd"):
        rows = log.where(sweep=sweep, method=method)
        if not rows:
            continue
        sizes = np.array([r[axis] for r in rows], dtype=np.float64)
        secs = np.array([r["seconds"] for r in rows], dtype=np.float64)
        order = np.argsort(sizes)
        sizes, secs = sizes[order], secs[order]

        slope = fit_exponent(sizes, secs)
        exponents[method] = slope
        ax.plot(
            sizes,
            secs * 1e3,
            marker="o",
            label=f"{method}  (slope {slope:.2f})",
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"{axis}  (log scale)")
    ax.set_ylabel("fit time, ms  (log scale)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    save_figure(fig, filename)
    plt.close(fig)
    return exponents


def main() -> None:
    log = RunLogger(RESULTS_DIR / "benchmark.jsonl", header=environment_record())

    print("B1: time vs n  (d fixed at %d)" % D_FIXED_FOR_N_SWEEP)
    for n in N_GRID:
        bench_one(log, "vs_n", n, D_FIXED_FOR_N_SWEEP)

    print("\nB2: time vs d  (n fixed at %d)" % N_FIXED_FOR_D_SWEEP)
    for d in D_GRID:
        bench_one(log, "vs_d", N_FIXED_FOR_D_SWEEP, d)

    print("\nPlots:")
    exp_n = plot_sweep(
        log,
        "vs_n",
        "n",
        "benchmark_vs_n.png",
        f"Fit time vs n  (d = {D_FIXED_FOR_N_SWEEP}, "
        f"{BENCH_MAX_ITER} GD iterations)",
    )
    exp_d = plot_sweep(
        log,
        "vs_d",
        "d",
        "benchmark_vs_d.png",
        f"Fit time vs d  (n = {N_FIXED_FOR_N_SWEEP if False else N_FIXED_FOR_D_SWEEP}, "
        f"{BENCH_MAX_ITER} GD iterations)",
    )

    # --- analysis, computed from the measurements ---------------------
    print("\n" + "=" * 68)
    print("Empirical complexity exponents (slope of log time vs log size)")
    print("=" * 68)
    for label, table in (("vs n", exp_n), ("vs d", exp_d)):
        print(f"\n  {label}:")
        for method, slope in table.items():
            print(f"    {method:<20} {slope:>6.2f}")

    cf = exp_d.get("linear_closed_form", float("nan"))
    gd = exp_d.get("linear_gd", float("nan"))
    print("\n  Prediction B2: the closed form grows superlinearly in d")
    print("  (d^3 factorization) while GD stays ~linear in d.")
    if np.isfinite(cf) and np.isfinite(gd):
        verdict = "SUPPORTED" if cf > gd + 0.3 else "NOT SUPPORTED"
        print(
            f"  Measured: closed_form {cf:.2f} vs gd {gd:.2f}  ->  {verdict}"
        )

    # crossover: smallest d at which the closed form is slower than GD
    rows = log.where(sweep="vs_d")
    by_d: dict[float, dict[str, float]] = {}
    for r in rows:
        by_d.setdefault(r["d"], {})[r["method"]] = r["seconds"]
    crossover = next(
        (
            d
            for d in sorted(by_d)
            if by_d[d].get("linear_closed_form", 0.0)
            > by_d[d].get("linear_gd", float("inf"))
        ),
        None,
    )
    print(
        f"\n  Crossover (closed form slower than {BENCH_MAX_ITER}-step GD): "
        f"{'d = ' + str(int(crossover)) if crossover else 'not reached on this grid'}"
    )
    print(f"\nLog: {log.path}")


if __name__ == "__main__":
    main()
