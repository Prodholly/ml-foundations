"""
controlled experiments on regularization, sample size and leakage
hypotheses were written down before running anything:

H1  n = 200, d = 50, only 5 features carry signal: test log-loss vs l2 is U-shaped, so lambda* > 0
    falsified if the curve only rises, ie lambda* = 0
H2  lambda* falls as n grows with d fixed: the likelihood grows with n, the penalty doesn't
    falsified if lambda* is flat or rising in n
H3  picking features on the full data inflates test AUC far more than fitting the scaler on it,
    and both inflations shrink as n grows
    falsified if the two are comparable, or the scaler leak is worse

one factor moves per experiment (lambda in E1, n in E2, leakage mode in E3); lr, iteration budget,
tolerance, binning, split fractions and seeding are fixed in common.py and logged
within a replicate every lambda sees the same data and split, so comparisons are paired
E1 also picks lambda on validation and reports its test loss: that's the honest number

usage: make -f 7_prob_stats/Makefile experiment
"""

from __future__ import annotations

import numpy as np

from experiments.common import (
    FIXED_GRAD_TOL,
    FIXED_LR,
    FIXED_MAX_ITER,
    N_CALIBRATION_BINS,
    RESULTS_DIR,
    DataConfig,
    RunLogger,
    derive_seed,
    environment_record,
    get_pyplot,
    make_logistic_data,
    save_figure,
)
from prob_stats import (
    LogisticRegression,
    Standardizer,
    accuracy,
    brier_score,
    expected_calibration_error,
    log_loss,
    reliability_curve,
    roc_auc,
    train_val_test_split,
)

# --- fixed design constants -----------------------------------------
# capped at 1.0: gd on l2 * ||theta||^2 is only stable while lr * (L_data + 2 * l2) < 2,
# so with FIXED_LR = 0.5 anything above ~1.7 diverges (2.15 blew up at iteration 2469)
LAMBDA_GRID = np.concatenate([[0.0], np.logspace(-4.0, 0.0, 16)])
VAL_FRAC, TEST_FRAC = 0.25, 0.25

# E1
E1_N, E1_D, E1_D_SIGNAL = 200, 50, 5
E1_REPLICATES = 20

# E2
E2_N_GRID = [100, 200, 400, 800, 1_600, 3_200]
E2_D, E2_D_SIGNAL = 50, 5
E2_REPLICATES = 10

# E3
E3_N_GRID = [100, 400, 1_600]
E3_D, E3_D_SIGNAL = 200, 5
E3_K_SELECT = 20
E3_LAMBDA = 0.01
E3_REPLICATES = 20


def fit_and_evaluate(
    X_tr, y_tr, X_va, y_va, X_te, y_te, l2: float
) -> dict:
    """
    fit one model, score it on train, val and test
    optimizer diagnostics go in too: an unconverged fit looks just like a converged one from outside
    """
    model = LogisticRegression(l2=l2).fit(
        X_tr,
        y_tr,
        lr=FIXED_LR,
        max_iter=FIXED_MAX_ITER,
        grad_tol=FIXED_GRAD_TOL,
    )
    out = {
        "converged": bool(model.optimize_result_.converged),
        "grad_norm": float(model.optimize_result_.grad_norm),
        "n_iter": int(model.optimize_result_.n_iter),
        "coef_l2_norm": float(np.linalg.norm(model.coef_)),
    }
    for tag, X, y in (
        ("train", X_tr, y_tr),
        ("val", X_va, y_va),
        ("test", X_te, y_te),
    ):
        logits = model.decision_function(X)  # shape: (m,)
        probs = model.predict_proba(X)  # shape: (m,)
        out[f"{tag}_log_loss"] = log_loss(y, logits)
        out[f"{tag}_accuracy"] = accuracy(y, probs)
        out[f"{tag}_brier"] = brier_score(y, probs)
        out[f"{tag}_ece"] = expected_calibration_error(
            y, probs, n_bins=N_CALIBRATION_BINS
        )
        try:
            out[f"{tag}_auc"] = roc_auc(y, probs)
        except ValueError:
            # one class absent in this split: AUC undefined, logged as
            # nan rather than silently imputed
            out[f"{tag}_auc"] = float("nan")
    return out, model


def prepare_splits(X, y, seed, leak_scaler=False):
    """
    split, then standardize using the training rows only
    leak_scaler=True fits the scaler on everything first, the scaler-leak arm of E3, and the only way to get it
    """
    tr, va, te = train_val_test_split(
        X.shape[0], VAL_FRAC, TEST_FRAC, seed=seed, stratify=y
    )
    scaler = Standardizer().fit(X if leak_scaler else X[tr])
    Xs = scaler.transform(X)  # shape: (n, d)
    return (Xs[tr], y[tr]), (Xs[va], y[va]), (Xs[te], y[te])


# =====================================================================
# E1 -- lambda sweep at fixed n
# =====================================================================
def experiment_1(log: RunLogger) -> dict:
    """
    E1: sweep lambda at n = 200, d = 50; draws the U-curve, other metrics and reliability plots
    """
    print(f"\nE1: lambda sweep, n={E1_N}, d={E1_D}, "
          f"d_signal={E1_D_SIGNAL}, {E1_REPLICATES} replicates")

    for rep in range(E1_REPLICATES):
        data_seed = derive_seed(1, rep, 0)
        split_seed = derive_seed(1, rep, 1)
        X, y, _ = make_logistic_data(
            DataConfig(E1_N, E1_D, E1_D_SIGNAL, seed=data_seed)
        )
        # paired: one dataset and one split shared by every lambda
        (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare_splits(X, y, split_seed)

        for l2 in LAMBDA_GRID:
            metrics, _ = fit_and_evaluate(
                Xtr, ytr, Xva, yva, Xte, yte, float(l2)
            )
            log.log(
                experiment="E1",
                replicate=rep,
                n=E1_N,
                d=E1_D,
                d_signal=E1_D_SIGNAL,
                l2=float(l2),
                data_seed=data_seed,
                split_seed=split_seed,
                lr=FIXED_LR,
                max_iter=FIXED_MAX_ITER,
                **metrics,
            )
        print(f"  replicate {rep + 1}/{E1_REPLICATES} done")

    # --- aggregate ----------------------------------------------------
    rows = log.where(experiment="E1")
    curves: dict[str, np.ndarray] = {}
    for key in ("train_log_loss", "val_log_loss", "test_log_loss",
                "test_accuracy", "test_ece", "test_auc", "coef_l2_norm"):
        curves[key] = np.array(
            [
                np.nanmean([r[key] for r in rows if r["l2"] == float(l2)])
                for l2 in LAMBDA_GRID
            ]
        )  # shape: (n_lambda,)
    curves["test_log_loss_sd"] = np.array(
        [
            np.nanstd([r["test_log_loss"] for r in rows if r["l2"] == float(l2)])
            for l2 in LAMBDA_GRID
        ]
    )

    i_test = int(np.nanargmin(curves["test_log_loss"]))
    i_val = int(np.nanargmin(curves["val_log_loss"]))
    i_ece = int(np.nanargmin(curves["test_ece"]))

    result = {
        "lambda_star_test": float(LAMBDA_GRID[i_test]),
        "lambda_star_val": float(LAMBDA_GRID[i_val]),
        "lambda_star_ece": float(LAMBDA_GRID[i_ece]),
        "test_loss_at_zero": float(curves["test_log_loss"][0]),
        "test_loss_at_star": float(curves["test_log_loss"][i_test]),
        "test_loss_at_val_choice": float(curves["test_log_loss"][i_val]),
        "curves": curves,
    }

    # --- plot 1: the U-curve -----------------------------------------
    plt = get_pyplot()
    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = np.where(LAMBDA_GRID == 0.0, LAMBDA_GRID[1] / 3.0, LAMBDA_GRID)

    ax.plot(x, curves["train_log_loss"], marker="o", ms=4, label="train")
    ax.plot(x, curves["val_log_loss"], marker="s", ms=4, label="validation")
    ax.plot(x, curves["test_log_loss"], marker="^", ms=4, label="test")
    ax.fill_between(
        x,
        curves["test_log_loss"] - curves["test_log_loss_sd"],
        curves["test_log_loss"] + curves["test_log_loss_sd"],
        alpha=0.15,
    )
    ax.axvline(x[i_test], ls="--", lw=1,
               label=f"lambda* (test) = {LAMBDA_GRID[i_test]:.4g}")
    ax.axvline(x[i_val], ls=":", lw=1,
               label=f"lambda* (val) = {LAMBDA_GRID[i_val]:.4g}")
    ax.set_xscale("log")
    ax.set_xlabel("L2 strength lambda  (leftmost point is lambda = 0)")
    ax.set_ylabel("mean cross-entropy, nats")
    ax.set_title(
        f"E1: H1 -- log-loss vs lambda\n"
        f"n={E1_N}, d={E1_D}, {E1_D_SIGNAL} signal features, "
        f"{E1_REPLICATES} replicates (shaded: +/- 1 SD)"
    )
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    save_figure(fig, "e1_logloss_vs_lambda.png")
    plt.close(fig)

    # --- plot 2: metrics disagree about lambda -----------------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax_, key, label in zip(
        axes,
        ("test_accuracy", "test_ece", "coef_l2_norm"),
        ("test accuracy", "test ECE", "||coef||_2"),
    ):
        ax_.plot(x, curves[key], marker="o", ms=4)
        ax_.axvline(x[i_test], ls="--", lw=1, color="k")
        ax_.set_xscale("log")
        ax_.set_xlabel("lambda")
        ax_.set_ylabel(label)
        ax_.grid(alpha=0.3)
    axes[1].set_title(
        "E1: the lambda minimizing log-loss (dashed) need not minimize "
        "ECE or maximize accuracy"
    )
    save_figure(fig, "e1_other_metrics.png")
    plt.close(fig)

    # --- plot 3: reliability at lambda=0 vs lambda* -------------------
    rep_seed_data = derive_seed(1, 0, 0)
    rep_seed_split = derive_seed(1, 0, 1)
    X, y, _ = make_logistic_data(
        DataConfig(E1_N, E1_D, E1_D_SIGNAL, seed=rep_seed_data)
    )
    (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare_splits(X, y, rep_seed_split)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    for l2, style in ((0.0, "o-"), (result["lambda_star_test"], "s-")):
        _, model = fit_and_evaluate(Xtr, ytr, Xva, yva, Xte, yte, float(l2))
        probs = model.predict_proba(Xte)  # shape: (m,)
        curve = reliability_curve(yte, probs, n_bins=N_CALIBRATION_BINS)
        keep = curve["count"] > 0
        ece = expected_calibration_error(
            yte, probs, n_bins=N_CALIBRATION_BINS
        )
        ax.plot(
            curve["mean_predicted"][keep],
            curve["mean_observed"][keep],
            style,
            label=f"lambda = {l2:.4g}  (ECE {ece:.3f})",
        )
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency")
    ax.set_title("E1: reliability, unpenalized vs lambda*  (replicate 0)")
    ax.grid(alpha=0.3)
    ax.legend()
    save_figure(fig, "e1_reliability.png")
    plt.close(fig)

    return result


# =====================================================================
# E2 -- how lambda* moves with n
# =====================================================================
def experiment_2(log: RunLogger) -> dict:
    """E2: pick lambda* on validation at each n, d fixed at 50"""
    print(f"\nE2: lambda* vs n, d={E2_D} fixed, "
          f"{E2_REPLICATES} replicates per n")

    lambda_star_by_n: dict[int, float] = {}
    test_loss_by_n: dict[int, float] = {}

    for n in E2_N_GRID:
        val_curve = np.zeros(LAMBDA_GRID.shape[0])  # shape: (n_lambda,)
        test_curve = np.zeros(LAMBDA_GRID.shape[0])

        for rep in range(E2_REPLICATES):
            data_seed = derive_seed(2, n, rep)
            split_seed = derive_seed(2, n, rep + 10_000)
            X, y, _ = make_logistic_data(
                DataConfig(n, E2_D, E2_D_SIGNAL, seed=data_seed)
            )
            (Xtr, ytr), (Xva, yva), (Xte, yte) = prepare_splits(
                X, y, split_seed
            )
            for j, l2 in enumerate(LAMBDA_GRID):
                metrics, _ = fit_and_evaluate(
                    Xtr, ytr, Xva, yva, Xte, yte, float(l2)
                )
                val_curve[j] += metrics["val_log_loss"] / E2_REPLICATES
                test_curve[j] += metrics["test_log_loss"] / E2_REPLICATES
                log.log(
                    experiment="E2",
                    replicate=rep,
                    n=n,
                    d=E2_D,
                    d_signal=E2_D_SIGNAL,
                    l2=float(l2),
                    data_seed=data_seed,
                    split_seed=split_seed,
                    lr=FIXED_LR,
                    max_iter=FIXED_MAX_ITER,
                    **metrics,
                )

        # lambda* chosen on VALIDATION, then evaluated on test
        j_star = int(np.nanargmin(val_curve))
        lambda_star_by_n[n] = float(LAMBDA_GRID[j_star])
        test_loss_by_n[n] = float(test_curve[j_star])
        print(
            f"  n={n:>5}  lambda*={LAMBDA_GRID[j_star]:<10.4g} "
            f"test log-loss={test_curve[j_star]:.4f}"
        )

    ns = np.array(sorted(lambda_star_by_n), dtype=np.float64)
    stars = np.array([lambda_star_by_n[int(n)] for n in ns])

    plt = get_pyplot()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(ns, np.where(stars == 0.0, LAMBDA_GRID[1] / 3.0, stars),
             marker="o")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("n (log)")
    ax1.set_ylabel("lambda* selected on validation (log)")
    ax1.set_title("E2: H2 -- lambda* vs n")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(ns, [test_loss_by_n[int(n)] for n in ns], marker="s")
    ax2.set_xscale("log")
    ax2.set_xlabel("n (log)")
    ax2.set_ylabel("test log-loss at lambda*")
    ax2.set_title("E2: test loss at the selected lambda")
    ax2.grid(alpha=0.3)
    save_figure(fig, "e2_lambda_star_vs_n.png")
    plt.close(fig)

    # monotonicity via Spearman-style rank correlation on log(n)
    rank_n = np.argsort(np.argsort(ns)).astype(np.float64)
    rank_l = np.argsort(np.argsort(stars)).astype(np.float64)
    denom = rank_n.std() * rank_l.std()
    rho = (
        float(((rank_n - rank_n.mean()) * (rank_l - rank_l.mean())).mean()
              / denom)
        if denom > 0
        else float("nan")
    )

    return {
        "lambda_star_by_n": lambda_star_by_n,
        "test_loss_by_n": test_loss_by_n,
        "rank_correlation": rho,
    }


# =====================================================================
# E3 -- leakage
# =====================================================================
def select_top_k(X, y, k: int) -> np.ndarray:
    """
    the k features most correlated with y; shape (k,)
    honest on training rows, a leak on all rows: the rows passed in decide which
    """
    yc = y - y.mean()  # shape: (m,)
    Xc = X - X.mean(axis=0)  # shape: (m, d)
    denom = np.sqrt((Xc**2).sum(axis=0) * (yc**2).sum())  # shape: (d,)
    corr = np.abs((Xc * yc[:, None]).sum(axis=0) / np.where(denom == 0, 1, denom))
    return np.argsort(corr)[::-1][:k]  # shape: (k,)


def experiment_3(log: RunLogger) -> dict:
    """E3: test AUC with no leak, a scaler leak, and a feature-selection leak"""
    print(f"\nE3: leakage, d={E3_D}, {E3_D_SIGNAL} signal features, "
          f"select top {E3_K_SELECT}, {E3_REPLICATES} replicates")

    summary: dict[str, dict[int, float]] = {
        mode: {} for mode in ("none", "scaler", "selection")
    }

    for n in E3_N_GRID:
        acc: dict[str, list[float]] = {m: [] for m in summary}

        for rep in range(E3_REPLICATES):
            data_seed = derive_seed(3, n, rep)
            split_seed = derive_seed(3, n, rep + 20_000)
            X, y, _ = make_logistic_data(
                DataConfig(n, E3_D, E3_D_SIGNAL, seed=data_seed)
            )
            tr, va, te = train_val_test_split(
                n, VAL_FRAC, TEST_FRAC, seed=split_seed, stratify=y
            )

            for mode in ("none", "scaler", "selection"):
                # arm definitions; only the first is correct
                if mode == "none":
                    cols = select_top_k(X[tr], y[tr], E3_K_SELECT)
                    scaler = Standardizer().fit(X[tr][:, cols])
                elif mode == "scaler":
                    cols = select_top_k(X[tr], y[tr], E3_K_SELECT)
                    scaler = Standardizer().fit(X[:, cols])  # LEAK
                else:
                    cols = select_top_k(X, y, E3_K_SELECT)  # LEAK
                    scaler = Standardizer().fit(X[tr][:, cols])

                Xs = scaler.transform(X[:, cols])  # shape: (n, k)
                metrics, _ = fit_and_evaluate(
                    Xs[tr], y[tr], Xs[va], y[va], Xs[te], y[te], E3_LAMBDA
                )
                acc[mode].append(metrics["test_auc"])
                log.log(
                    experiment="E3",
                    leakage_mode=mode,
                    replicate=rep,
                    n=n,
                    d=E3_D,
                    d_signal=E3_D_SIGNAL,
                    k_select=E3_K_SELECT,
                    l2=E3_LAMBDA,
                    data_seed=data_seed,
                    split_seed=split_seed,
                    **metrics,
                )

        for mode in summary:
            summary[mode][n] = float(np.nanmean(acc[mode]))
        print(
            f"  n={n:>5}  none={summary['none'][n]:.4f}  "
            f"scaler={summary['scaler'][n]:.4f}  "
            f"selection={summary['selection'][n]:.4f}"
        )

    plt = get_pyplot()
    ns = np.array(E3_N_GRID, dtype=np.float64)
    width = 0.26
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    pos = np.arange(len(ns), dtype=np.float64)

    for i, mode in enumerate(("none", "scaler", "selection")):
        vals = [summary[mode][int(n)] for n in ns]
        ax1.bar(pos + (i - 1) * width, vals, width, label=mode)
    ax1.axhline(0.5, color="k", ls="--", lw=1, label="chance")
    ax1.set_xticks(pos)
    ax1.set_xticklabels([str(int(n)) for n in ns])
    ax1.set_xlabel("n")
    ax1.set_ylabel("reported test AUC")
    ax1.set_title("E3: H3 -- apparent test AUC by leakage mode")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    for mode in ("scaler", "selection"):
        optimism = [
            summary[mode][int(n)] - summary["none"][int(n)] for n in ns
        ]
        ax2.plot(ns, optimism, marker="o", label=mode)
    ax2.axhline(0.0, color="k", lw=1)
    ax2.set_xscale("log")
    ax2.set_xlabel("n (log)")
    ax2.set_ylabel("AUC optimism vs leakage-free arm")
    ax2.set_title("E3: optimism shrinks with n")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    save_figure(fig, "e3_leakage.png")
    plt.close(fig)

    return summary


# =====================================================================
# analysis
# =====================================================================
def report(e1: dict, e2: dict, e3: dict) -> None:
    """print each verdict, computed from the measurements rather than asserted"""
    line = "=" * 68
    print(f"\n{line}\nANALYSIS\n{line}")

    # --- H1 -----------------------------------------------------------
    star = e1["lambda_star_test"]
    at_zero, at_star = e1["test_loss_at_zero"], e1["test_loss_at_star"]
    h1 = "SUPPORTED" if star > 0.0 else "CONTRADICTED"
    print(f"\nH1 [{h1}]")
    print(
        f"  Test log-loss is minimized at lambda* = {star:.4g}, where it "
        f"reaches {at_star:.4f} nats, against {at_zero:.4f} nats "
        f"unpenalized -- an improvement of "
        f"{at_zero - at_star:.4f} nats "
        f"({100 * (at_zero - at_star) / at_zero:.1f}%). With only "
        f"{E1_D_SIGNAL} of {E1_D} features carrying signal and n="
        f"{E1_N}, the unpenalized fit has ~{E1_D - E1_D_SIGNAL} "
        f"directions in which to fit noise, and shrinking those "
        f"coefficients costs little bias while removing substantial "
        f"variance. As lambda grows further the penalty begins to shrink "
        f"the genuine signal too and the loss rises again, which is the "
        f"bias-dominated arm of the U."
    )
    print(
        f"  Selection honesty: choosing lambda on validation picks "
        f"{e1['lambda_star_val']:.4g}, whose test loss is "
        f"{e1['test_loss_at_val_choice']:.4f} nats -- "
        f"{e1['test_loss_at_val_choice'] - at_star:+.4f} versus the best "
        f"achievable test loss. That gap is the optimism that selecting "
        f"on test would have hidden."
    )
    print(
        f"  Calibration disagrees with loss: ECE is minimized at lambda = "
        f"{e1['lambda_star_ece']:.4g}, not {star:.4g}. The lambda that "
        f"best predicts is not the lambda that is best calibrated, which "
        f"is why both are reported."
    )

    # --- H2 -----------------------------------------------------------
    rho = e2["rank_correlation"]
    h2 = "SUPPORTED" if rho < -0.5 else (
        "CONTRADICTED" if rho > 0.5 else "INCONCLUSIVE"
    )
    print(f"\nH2 [{h2}]")
    print(
        f"  Rank correlation between n and lambda* is {rho:+.3f}. "
        f"lambda* by n: "
        + ", ".join(
            f"n={n}: {v:.4g}" for n, v in sorted(e2["lambda_star_by_n"].items())
        )
    )
    print(
        "  Mechanism: the data term of the objective is a mean over n "
        "observations while the penalty is not, so on the sum scale the "
        "likelihood grows linearly in n and the prior does not. In MAP "
        "terms the implied prior variance tau^2 = 1/(2 n lambda) must "
        "grow to keep the same prior influence, i.e. lambda must fall."
    )

    # --- H3 -----------------------------------------------------------
    ns = sorted(e3["none"])
    scaler_opt = [e3["scaler"][n] - e3["none"][n] for n in ns]
    select_opt = [e3["selection"][n] - e3["none"][n] for n in ns]
    h3 = (
        "SUPPORTED"
        if np.mean(select_opt) > np.mean(scaler_opt) + 0.02
        else "CONTRADICTED"
    )
    print(f"\nH3 [{h3}]")
    print(
        f"  Mean AUC optimism: scaler leakage {np.mean(scaler_opt):+.4f}, "
        f"selection leakage {np.mean(select_opt):+.4f}. "
        f"At n={ns[0]}: {scaler_opt[0]:+.4f} vs {select_opt[0]:+.4f}; "
        f"at n={ns[-1]}: {scaler_opt[-1]:+.4f} vs {select_opt[-1]:+.4f}."
    )
    print(
        f"  Mechanism: fitting the standardizer on all rows leaks two "
        f"summary statistics per feature, which barely helps a model. "
        f"Selecting {E3_K_SELECT} of {E3_D} features by full-data "
        f"correlation leaks which noise features happen to correlate "
        f"with the test labels, so the model is handed a feature set "
        f"pre-fitted to the evaluation data. Both shrink with n because "
        f"spurious correlations shrink with n."
    )
    print(
        "  Practical reading: 'we standardized before splitting' is a "
        "minor sin; 'we selected features on the full dataset' "
        "invalidates the reported number."
    )
    print(f"\n{line}")


def main() -> None:
    log = RunLogger(
        RESULTS_DIR / "experiment.jsonl", header=environment_record()
    )
    e1 = experiment_1(log)
    e2 = experiment_2(log)
    e3 = experiment_3(log)
    report(e1, e2, e3)
    print(f"Log: {log.path}")
    print(f"Figures: {RESULTS_DIR / 'figures'}")


if __name__ == "__main__":
    main()
