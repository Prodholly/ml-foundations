import math
from functools import partial
from pathlib import Path

import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, CosineAnnealingWarmRestarts, LambdaLR, OneCycleLR, StepLR,
)

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
SCHEDULERS = ("step", "cosine", "warm_restarts", "one_cycle", "warmup_cosine")


def warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(progress, 1.0)
    return 0.5 * (1 + math.cos(math.pi * progress))


def build_scheduler(name: str, optimizer: torch.optim.Optimizer, epochs: int, steps_per_epoch: int = 1,
                    max_lr: float | None = None, warmup_fraction: float = 0.05):
    total = epochs * steps_per_epoch
    base_lr = optimizer.param_groups[0]["lr"]

    if name == "step":
        return StepLR(optimizer, step_size=30, gamma=0.1)
    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs)
    if name == "warm_restarts":
        return CosineAnnealingWarmRestarts(optimizer, T_0=10)
    if name == "one_cycle":
        return OneCycleLR(optimizer, max_lr=max_lr or base_lr, total_steps=total, pct_start=0.3)
    if name == "warmup_cosine":
        fn = partial(warmup_cosine, warmup_steps=max(1, int(warmup_fraction * total)), total_steps=total)
        return LambdaLR(optimizer, lr_lambda=fn)
    raise ValueError(f"unknown scheduler {name!r}, pick one of {SCHEDULERS}")


def simulate_schedule(name: str, epochs: int = 100, base_lr: float = 0.1) -> list[float]:
    p = torch.zeros(1, requires_grad=True)
    opt = torch.optim.SGD([p], lr=base_lr)

    sched = build_scheduler(name, opt, epochs, steps_per_epoch=1, max_lr=base_lr)

    lrs = []
    for _ in range(epochs):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()

    return lrs


def plot_scheduler_comparison(epochs: int = 100, base_lr: float = 0.1) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    titles = {
        "step": "StepLR(30, 0.1)",
        "cosine": "CosineAnnealingLR(T_max=100)",
        "warm_restarts": "CosineAnnealingWarmRestarts(T_0=10)",
        "one_cycle": "OneCycleLR(pct_start=0.3)",
        "warmup_cosine": "LambdaLR warmup (5%) + cosine",
    }
    fig, axes = plt.subplots(1, len(SCHEDULERS), figsize=(4 * len(SCHEDULERS), 3.4), sharey=True)
    for ax, name in zip(axes, SCHEDULERS):
        lrs = simulate_schedule(name, epochs, base_lr)
        ax.plot(range(epochs), lrs, lw=2, color="#2a78d6")
        ax.set_title(titles[name], fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(True, color="#e4e3df")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel(f"lr (base {base_lr})")
    fig.suptitle("every schedule over 100 epochs, one step per epoch", fontsize=12, y=1.03)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "scheduler_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", plot_scheduler_comparison())
