from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"


class ReLU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x)


class GELU(nn.Module):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__()
        self.approximate = approximate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.silu(x)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.leaky_relu(x, self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class ELU(nn.Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.elu(x, self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


ACTIVATIONS = {"relu": ReLU, "gelu": GELU, "swish": Swish, "leaky_relu": LeakyReLU, "elu": ELU}


def get_activation(name: str, **kwargs) -> nn.Module:
    if name not in ACTIVATIONS:
        raise ValueError(f"unknown activation {name!r}, pick one of {list(ACTIVATIONS)}")
    return ACTIVATIONS[name](**kwargs)


def activation_and_grad(module: nn.Module, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.detach().clone().requires_grad_()

    y = module(x)

    y.sum().backward()

    return y.detach(), x.grad


def plot_nn_activations() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from numpy_mlp import activations as np_act

    x = torch.linspace(-3, 3, 601, dtype=torch.float64)
    xn = x.numpy()
    ones = np.ones_like(xn)

    panels = [
        ("ReLU", ReLU(), (np_act.relu(xn), np_act.relu_backward(xn, ones))),
        ("GELU (tanh)", GELU("tanh"), (np_act.gelu(xn), np_act.gelu_backward(xn, ones))),
        ("Swish / SiLU", Swish(), (np_act.swish(xn), np_act.swish_backward(xn, ones))),
        ("LeakyReLU (0.1)", LeakyReLU(0.1), None),
        ("ELU (alpha 1)", ELU(1.0), None),
    ]

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 3.8), sharex=True)
    for ax, (title, module, ref) in zip(axes, panels):
        y, dy = activation_and_grad(module, x)
        ax.plot(xn, y.numpy(), color="#2a78d6", lw=2, label="f(x)  F.*")
        ax.plot(xn, dy.numpy(), color="#eb6834", lw=2, ls="--", label="f'(x)  autograd")

        if ref is not None:
            s = slice(0, None, 30)
            ax.plot(xn[s], ref[0][s], "o", ms=3.5, color="#0b0b0b", label="phase 1 numpy")
            ax.plot(xn[s], ref[1][s], "o", ms=3.5, color="#0b0b0b")

        ax.axhline(0, color="#52514e", lw=0.8)
        ax.axvline(0, color="#52514e", lw=0.8)
        ax.grid(True, color="#e4e3df")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("x")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle("nn.functional activations (solid) and autograd derivatives (dashed), dots = phase 1 numpy",
                 fontsize=12, y=1.03)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "nn_activations.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print("saved", plot_nn_activations())
