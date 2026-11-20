from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .activations import gelu, gelu_backward, relu, relu_backward, sigmoid, swish, swish_backward

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)


def _save(fig: plt.Figure, name: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_activations() -> Path:
    x = np.linspace(-3, 3, 601)
    ones = np.ones_like(x)
    p = sigmoid(x)

    panels = [
        ("ReLU", relu(x), relu_backward(x, ones), "gradient is exactly 0 for x < 0:\na unit stuck here never learns again"),
        ("GELU (tanh approx)", gelu(x), gelu_backward(x, ones), "small negative slope for x < 0,\nthe gradient never fully dies"),
        ("Swish  x·σ(x)", swish(x), swish_backward(x, ones), "smooth, dips below 0 near x = -1.3,\ngradient keeps flowing"),
        ("Softmax on [x, 0]", p, p * (1 - p), "derivative p(1 - p) peaks at 0.25,\nfades when one class dominates"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharex=True)
    for ax, (title, f, df, note) in zip(axes, panels):
        _style(ax)
        ax.plot(x, f, color=BLUE, linewidth=2, label="f(x)")
        ax.plot(x, df, color=ORANGE, linewidth=2, linestyle="--", label="f'(x)")
        ax.axhline(0, color=MUTED, linewidth=0.8)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_title(title, color=INK, fontsize=12)
        ax.set_xlabel("x", color=MUTED)
        ax.text(0.03, 0.97, note, transform=ax.transAxes, va="top", fontsize=8.5, color=MUTED)

    axes[0].axvspan(-3, 0, color=ORANGE, alpha=0.08, linewidth=0)
    axes[0].legend(loc="center left", frameon=False, fontsize=9)
    fig.suptitle("activations (solid) and their derivatives (dashed)", color=INK, fontsize=13, y=1.03)
    return _save(fig, "activations.png")


def plot_init_comparison(results: dict[str, dict[str, list]]) -> Path:
    colors = {"gaussian": ORANGE, "xavier": AQUA, "he": BLUE, "lecun": YELLOW}
    schemes = list(results)
    n_layers = len(results[schemes[0]]["grad_norms"][0])

    fig, axes = plt.subplots(1, n_layers + 1, figsize=(4.2 * (n_layers + 1), 4))
    for layer in range(n_layers):
        ax = axes[layer]
        _style(ax)
        for s in schemes:
            norms = np.array(results[s]["grad_norms"])
            epochs = np.arange(norms.shape[0])
            ax.plot(epochs, norms[:, layer], color=colors.get(s, MUTED), linewidth=2, marker="o", markersize=5, label=s)
        ax.set_yscale("log")
        ax.set_title(f"layer {layer + 1}: mean ||dW||", color=INK, fontsize=11)
        ax.set_xlabel("epoch (0 = initialization)", color=MUTED)
        ax.set_xticks(epochs)

    ax = axes[-1]
    _style(ax)
    for s in schemes:
        accs = results[s]["val_accs"]
        ax.plot(np.arange(1, len(accs) + 1), accs, color=colors.get(s, MUTED), linewidth=2, marker="o", markersize=5, label=s)
    ax.set_title("val accuracy", color=INK, fontsize=11)
    ax.set_xlabel("epoch", color=MUTED)
    ax.set_xticks(np.arange(1, len(accs) + 1))
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    fig.suptitle("same network, same data, same SGD: only the initialization changes", color=INK, fontsize=13, y=1.03)
    return _save(fig, "init_comparison.png")


def plot_training_curves(
    train_losses: list[float],
    val_losses: list[float],
    train_accs: list[float],
    val_accs: list[float],
    best_epoch: int | None = None,
    name: str = "numpy_training_curves.png",
) -> Path:
    epochs = np.arange(1, len(train_losses) + 1)
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(12, 4.2))

    for ax, tr, va, title in ((ax_l, train_losses, val_losses, "cross-entropy loss"), (ax_a, train_accs, val_accs, "accuracy")):
        _style(ax)
        ax.plot(epochs, tr, color=BLUE, linewidth=2, label="train")
        ax.plot(epochs, va, color=ORANGE, linewidth=2, label="val")
        if best_epoch is not None:
            ax.axvline(best_epoch, color=MUTED, linewidth=1, linestyle=":")
        ax.set_title(title, color=INK, fontsize=12)
        ax.set_xlabel("epoch", color=MUTED)

    if best_epoch is not None:
        near_end = best_epoch > 0.75 * len(train_losses)
        ax_l.text(best_epoch, ax_l.get_ylim()[1], f"kept: epoch {best_epoch} " if near_end else f" kept: epoch {best_epoch}",
                  ha="right" if near_end else "left", va="top", fontsize=9, color=MUTED)
    ax_l.legend(frameon=False, fontsize=9)
    return _save(fig, name)


if __name__ == "__main__":
    print("saved", plot_activations())
