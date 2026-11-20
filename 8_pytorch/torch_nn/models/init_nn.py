import json
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
SCHEMES = ("normal_0.01", "xavier_uniform", "kaiming_normal", "kaiming_normal_fan_in", "orthogonal")


def initialize_model(model: nn.Module, scheme: str) -> nn.Module:
    for module in model.modules():
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            if scheme == "xavier_uniform":
                nn.init.xavier_uniform_(module.weight)
            elif scheme == "kaiming_normal":
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif scheme == "kaiming_normal_fan_in":
                nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
            elif scheme == "orthogonal":
                nn.init.orthogonal_(module.weight)
            elif scheme == "normal_0.01":
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            else:
                raise ValueError(f"unknown init scheme {scheme!r}")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


def linear_grad_norms(model: nn.Module) -> list[float]:
    return [m.weight.grad.norm().item() for m in model.modules() if isinstance(m, nn.Linear)]


def compare_init_gradient_flow(schemes: tuple[str, ...] = SCHEMES, epochs: int = 5, batch_size: int = 256,
                               lr: float = 0.01, seed: int = 0) -> dict[str, dict[str, list]]:
    from ..data import load_cifar10_tensors, tensor_batches
    from ..training.reproducibility_nn import set_seed
    from ..training.trainer_nn import get_device
    from .mlp import MLP

    device = get_device()
    data = load_cifar10_tensors(device, seed)
    results: dict[str, dict[str, list]] = {}

    for scheme in schemes:
        set_seed(seed)

        model = initialize_model(MLP(activation="relu"), scheme).to(device)

        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        model.zero_grad()
        F.cross_entropy(model(data["X_train"][:batch_size]), data["y_train"][:batch_size]).backward()
        grad_norms = [linear_grad_norms(model)]
        val_accs = []

        g = torch.Generator().manual_seed(seed)
        for epoch in range(1, epochs + 1):
            model.train()
            sums, steps = None, 0
            for Xb, yb in tensor_batches(data["X_train"], data["y_train"], batch_size, g):
                opt.zero_grad(set_to_none=True)
                F.cross_entropy(model(Xb), yb).backward()
                norms = torch.stack([m.weight.grad.norm() for m in model.modules() if isinstance(m, nn.Linear)])
                sums = norms if sums is None else sums + norms
                steps += 1
                opt.step()
            grad_norms.append((sums / steps).tolist())

            model.eval()
            with torch.no_grad():
                acc = (model(data["X_val"]).argmax(1) == data["y_val"]).float().mean().item()
            val_accs.append(acc)
            print(f"{scheme:15s} epoch {epoch} | val_acc {acc:.4f} | grad norms {[round(n, 4) for n in grad_norms[-1]]}")

        results[scheme] = {"grad_norms": grad_norms, "val_accs": val_accs}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "init_nn_comparison.json").write_text(json.dumps(results, indent=2))
    plot_init_comparison(results)
    return results


def plot_init_comparison(results: dict[str, dict[str, list]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    colors = {"normal_0.01": "#eb6834", "xavier_uniform": "#1baf7a", "kaiming_normal": "#2a78d6", "kaiming_normal_fan_in": "#8a4fd6", "orthogonal": "#eda100"}
    schemes = list(results)
    n_layers = len(results[schemes[0]]["grad_norms"][0])

    fig, axes = plt.subplots(1, n_layers + 1, figsize=(3.6 * (n_layers + 1), 3.8))
    for layer in range(n_layers):
        ax = axes[layer]
        for s in schemes:
            norms = np.array(results[s]["grad_norms"])
            ax.plot(np.arange(len(norms)), norms[:, layer], marker="o", ms=4, lw=2, color=colors.get(s), label=s)
        ax.set_yscale("log")
        ax.set_title(f"linear {layer + 1}: mean ||dW||", fontsize=10)
        ax.set_xlabel("epoch (0 = init)")
        ax.grid(True, color="#e4e3df")

    ax = axes[-1]
    for s in schemes:
        accs = results[s]["val_accs"]
        ax.plot(np.arange(1, len(accs) + 1), accs, marker="o", ms=4, lw=2, color=colors.get(s), label=s)
    ax.set_title("val accuracy", fontsize=10)
    ax.set_xlabel("epoch")
    ax.grid(True, color="#e4e3df")
    ax.legend(frameon=False, fontsize=8)
    fig.suptitle("CIFAR-10 mlp, SGD momentum, only nn.init changes", fontsize=12, y=1.02)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "init_nn_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    compare_init_gradient_flow()
