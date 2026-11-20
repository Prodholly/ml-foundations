from pathlib import Path

import torch
from torch import nn

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


def get_weights(model: nn.Module) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def random_direction(weights: list[torch.Tensor], generator: torch.Generator | None = None) -> list[torch.Tensor]:
    return [torch.randn(w.shape, generator=generator, dtype=w.dtype).to(w.device) for w in weights]


def filter_normalize(direction: list[torch.Tensor], weights: list[torch.Tensor]) -> None:
    for d, w in zip(direction, weights):
        if w.dim() <= 1:
            d.zero_()
        else:
            d_rows = d.view(d.shape[0], -1)
            w_rows = w.view(w.shape[0], -1)
            d_rows.mul_(w_rows.norm(dim=1, keepdim=True) / (d_rows.norm(dim=1, keepdim=True) + 1e-10))


def set_weights(model: nn.Module, base, d1, d2, alpha: float, beta: float) -> None:
    with torch.no_grad():
        for p, w, x, y in zip(model.parameters(), base, d1, d2):
            p.copy_(w + alpha * x + beta * y)


def loss_surface(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion,
                 steps: int = 25, span: float = 1.0, seed: int = 0):
    model.eval()
    base = get_weights(model)

    g = torch.Generator().manual_seed(seed)
    d1, d2 = random_direction(base, g), random_direction(base, g)
    filter_normalize(d1, base)
    filter_normalize(d2, base)

    alphas = torch.linspace(-span, span, steps)
    betas = torch.linspace(-span, span, steps)
    Z = torch.zeros(steps, steps)

    with torch.no_grad():
        for i, a in enumerate(alphas):
            for j, b in enumerate(betas):
                set_weights(model, base, d1, d2, a.item(), b.item())
                Z[i, j] = criterion(model(X), y).item()

    set_weights(model, base, d1, d2, 0.0, 0.0)
    return alphas, betas, Z


def plot_landscape(alphas, betas, Z, name: str = "loss_landscape_nn.png", title: str = "") -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    A, B = np.meshgrid(alphas.numpy(), betas.numpy(), indexing="ij")
    Zn = Z.numpy()

    fig = plt.figure(figsize=(12, 5))

    ax3 = fig.add_subplot(1, 2, 1, projection="3d")
    ax3.plot_surface(A, B, Zn, cmap="viridis", linewidth=0, antialiased=True)
    ax3.set_xlabel("alpha (d1)")
    ax3.set_ylabel("beta (d2)")
    ax3.set_zlabel("loss")

    ax = fig.add_subplot(1, 2, 2)
    levels = np.geomspace(max(Zn.min(), 1e-3), Zn.max(), 20)
    cs = ax.contour(A, B, Zn, levels=levels, cmap="viridis")
    ax.clabel(cs, fontsize=7, fmt="%.2f")
    ax.plot(0, 0, marker="*", ms=12, color="#eb6834")
    ax.set_xlabel("alpha (d1)")
    ax.set_ylabel("beta (d2)")
    ax.set_aspect("equal")
    fig.suptitle(title or "loss around the trained weights, filter-normalized directions", fontsize=12)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
