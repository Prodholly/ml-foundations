# run from 6_grad_check:  python3 examples/figures.py
# figures for the readme, built on top of grad_check (nothing in here is part of the task code)
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from grad_check import numerical_gradient, forward, cross_entropy_loss, analytical_gradient_W
from run_all import setup   # same X, W, y_true as run_all (seed 42)

FIGURES = os.path.join(os.path.dirname(__file__), "..", "figures")
RED, GREEN, BLUE, GRAY = "#d62828", "#2a9d3f", "steelblue", "#8d99ae"


def rel_error(a, n):
    return np.abs(a - n) / (np.maximum(np.abs(a), np.abs(n)) + 1e-8)


def both_grads(X, W, y_true, h=1e-5):
    D, C = W.shape

    def loss_fn(W_flat):
        probs, _ = forward(X, W_flat.reshape(D, C))
        return cross_entropy_loss(probs, y_true)

    num = numerical_gradient(loss_fn, W.flatten(), h=h).reshape(D, C)
    probs, _ = forward(X, W)
    ana = analytical_gradient_W(X, probs, y_true)
    return ana, num


# 1. what probs - y_true actually says, for one sample
def gradient_intuition(X, W, y_true, i=0):
    probs, z = forward(X, W)
    z, p, y = z[i], probs[i], y_true[i]
    true_c = int(y.argmax())
    classes = [f"class {c}" for c in range(len(z))]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    panels = [
        (z, "1. logits  z = X[i] @ W", "score"),
        (p, "2. softmax  probs", "probability"),
        (p - y, "3. gradient  probs - y_true", "dL/dz"),
    ]
    for ax, (vals, title, ylabel) in zip(axes, panels):
        colors = [GRAY] * len(vals)
        colors[true_c] = BLUE
        if "gradient" in title:   # sign is what matters here
            colors = [GREEN if v < 0 else RED for v in vals]
        ax.bar(classes, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.axhline(0, color="black", linewidth=0.8)
        for k, v in enumerate(vals):
            ax.annotate(f"{v:+.2f}" if "gradient" in title else f"{v:.2f}",
                        xy=(k, v), xytext=(0, 4 if v >= 0 else -12),
                        textcoords="offset points", ha="center", fontsize=9)
        ax.margins(y=0.18)   # room for the value labels
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)

    axes[1].set_ylim(0, 1.05)
    axes[1].text(true_c, 1.0, "true class", ha="center", color=BLUE, fontsize=9)
    # how gradient descent reads the last panel
    axes[2].text(0.03, 0.96, "green < 0: descent pushes this score up\nred > 0: descent pushes it down",
                 transform=axes[2].transAxes, fontsize=8.5, color="#333", va="top")

    fig.suptitle(f"sample {i}: the true class is class {true_c}", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "grad_intuition.png"), dpi=150)
    plt.close(fig)


# 2. did the check pass? every entry of dL/dW, analytical vs numerical
def check_result(X, W, y_true):
    ana, num = both_grads(X, W, y_true)
    err = rel_error(ana, num)

    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))

    # left: all 12 entries should sit on y = x
    lim = 1.1 * max(np.abs(ana).max(), np.abs(num).max())
    left.plot([-lim, lim], [-lim, lim], "--", color=GRAY, linewidth=1, label="y = x (perfect match)")
    left.scatter(ana.ravel(), num.ravel(), s=60, color=BLUE, edgecolor="black", zorder=3, label="entries of dL/dW")
    left.set_xlim(-lim, lim)
    left.set_ylim(-lim, lim)
    left.set_aspect("equal")
    left.set_xlabel("analytical  X.T @ (probs - y) / N")
    left.set_ylabel("numerical  centered difference")
    left.set_title("12 gradient entries, two ways")
    left.legend(loc="upper left", fontsize=9)
    left.grid(True, alpha=0.3)

    # right: relative error per entry, colored on the cs231n scale
    im = right.imshow(err, cmap="RdYlGn_r", norm=LogNorm(vmin=1e-10, vmax=1e-2))
    for (r, c), v in np.ndenumerate(err):
        right.text(c, r, f"{v:.0e}", ha="center", va="center", fontsize=9)
    right.set_xticks(range(err.shape[1]), [f"class {c}" for c in range(err.shape[1])])
    right.set_yticks(range(err.shape[0]), [f"feature {d}" for d in range(err.shape[0])])
    right.set_title(f"relative error per entry  (max {err.max():.1e})")
    bar = fig.colorbar(im, ax=right, fraction=0.046)
    bar.set_ticks([1e-9, 1e-7, 1e-5, 1e-3])
    bar.set_ticklabels(["1e-9", "1e-7 excellent", "1e-5 target", "1e-3 bug"])
    for t in (1e-7, 1e-5, 1e-3):
        bar.ax.axhline(t, color="black", linewidth=1)

    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "grad_check_result.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# 3. choosing h: sweep it on the real softmax loss
def h_sweep(X, W, y_true):
    hs = np.logspace(-1, -12, 45)
    errs = [rel_error(*both_grads(X, W, y_true, h=h)).max() for h in hs]

    fig, ax = plt.subplots(figsize=(8.5, 5))
    # cs231n zones
    ax.axhspan(1e-20, 1e-7, color=GREEN, alpha=0.08)
    ax.axhspan(1e-7, 1e-5, color="#f4d35e", alpha=0.15)
    ax.axhspan(1e-3, 1e3, color=RED, alpha=0.08)
    zone = ax.get_yaxis_transform()   # x in axes coords, y in data
    ax.text(0.02, 3e-9, "excellent  < 1e-7", transform=zone, fontsize=9, color=GREEN)
    ax.text(0.02, 2.2e-6, "acceptable", transform=zone, fontsize=9, color="#a07c00")
    ax.text(0.02, 2e-1, "almost certainly a bug  > 1e-3", transform=zone, fontsize=9, color=RED)

    ax.loglog(hs, errs, marker="o", markersize=4, linewidth=2, color=BLUE)
    ax.axvline(1e-5, color="black", linestyle="--", linewidth=1)
    ax.text(8e-6, 3e-4, "default h = 1e-5", fontsize=9)
    ax.text(1e-4, 4e-9, "floor ~1e-8: the +1e-9 inside log()", fontsize=8.5, color="#333")
    ax.annotate("too big: the curve bends\naway from the secant", xy=(hs[2], errs[2]), xytext=(1.5e-3, 3e-2),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#333"))
    ax.annotate("too small: float round-off\ntakes over", xy=(hs[-8], errs[-8]), xytext=(3e-10, 3e-8),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#333"))

    ax.invert_xaxis()   # read left to right as h shrinks
    ax.set_ylim(1e-9, 1)
    ax.set_xlabel("step size h (shrinking ->)")
    ax.set_ylabel("max relative error over dL/dW")
    ax.set_title("gradient check vs step size, on the real softmax loss")
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "h_sweep.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIGURES, exist_ok=True)
    X, W, y_true = setup()
    for make in (gradient_intuition, check_result, h_sweep):
        make(X, W, y_true)
        print("done:", make.__name__)
