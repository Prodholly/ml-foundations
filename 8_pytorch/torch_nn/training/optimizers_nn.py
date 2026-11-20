import json
from pathlib import Path

import torch
from torch import nn

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

OPTIMIZER_CONFIGS = {
    "sgd_nesterov": dict(lr=0.1, momentum=0.9, nesterov=True),
    "adam": dict(lr=1e-3, betas=(0.9, 0.999), eps=1e-8),
    "adamw": dict(lr=1e-3, weight_decay=0.01),
    "rmsprop": dict(lr=1e-3, alpha=0.99),
    "lbfgs": dict(lr=1, max_iter=20),
}

OPTIMIZER_CLASSES = {
    "sgd_nesterov": torch.optim.SGD,
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "rmsprop": torch.optim.RMSprop,
    "lbfgs": torch.optim.LBFGS,
}


def build_optimizer(name: str, params) -> torch.optim.Optimizer:
    if name not in OPTIMIZER_CLASSES:
        raise ValueError(f"unknown optimizer {name!r}, pick one of {list(OPTIMIZER_CLASSES)}")
    return OPTIMIZER_CLASSES[name](params, **OPTIMIZER_CONFIGS[name])


def train_step(model: nn.Module, optimizer: torch.optim.Optimizer, criterion, X, y) -> float:
    if isinstance(optimizer, torch.optim.LBFGS):
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            return loss

        loss = optimizer.step(closure)
    else:
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    return loss.item()


def run_comparison(epochs: int = 5, batch_size: int = 256, seed: int = 0, lbfgs_steps: int = 120,
                   log_every: int = 20) -> dict[str, dict[str, list]]:
    from ..data import load_cifar10_tensors, tensor_batches
    from ..models.init_nn import initialize_model
    from ..models.mlp import MLP
    from .reproducibility_nn import set_seed
    from .trainer_nn import get_device

    device = get_device()
    data = load_cifar10_tensors(device, seed)
    criterion = nn.CrossEntropyLoss()
    results: dict[str, dict[str, list]] = {}

    for name in OPTIMIZER_CONFIGS:
        set_seed(seed)

        model = initialize_model(MLP(use_batchnorm=True), "kaiming_normal_fan_in").to(device)

        opt = build_optimizer(name, model.parameters())

        g = torch.Generator().manual_seed(seed)
        steps, losses, window = [], [], []
        step = 0
        budget = lbfgs_steps if name == "lbfgs" else None
        done = False
        for _ in range(epochs):
            model.train()
            for Xb, yb in tensor_batches(data["X_train"], data["y_train"], batch_size, g):
                window.append(train_step(model, opt, criterion, Xb, yb))
                step += 1
                if step % log_every == 0:
                    steps.append(step)
                    losses.append(sum(window) / len(window))
                    window = []
                if budget is not None and step >= budget:
                    done = True
                    break
            if done:
                break

        model.eval()
        with torch.no_grad():
            val_acc = (model(data["X_val"]).argmax(1) == data["y_val"]).float().mean().item()
        results[name] = {"steps": steps, "losses": losses, "val_acc": val_acc}
        print(f"{name:13s} final train loss {losses[-1]:.4f} | val_acc {val_acc:.4f} | {step} steps")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "optimizer_comparison.json").write_text(json.dumps(results, indent=2))
    plot_optimizer_comparison(results)
    return results


def plot_optimizer_comparison(results: dict[str, dict[str, list]]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sgd_nesterov": "#2a78d6", "adam": "#eb6834", "adamw": "#1baf7a", "rmsprop": "#eda100",
              "lbfgs": "#8a4fd6"}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, r in results.items():
        ax.plot(r["steps"], r["losses"], lw=2, color=colors.get(name),
                label=f"{name}  (val acc {r['val_acc']:.3f})")
    ax.set_xlabel("step (batch 256)")
    ax.set_ylabel("train cross-entropy, mean over 20 steps")
    ax.set_title("same mlp, same init, same batches: only the optimizer changes", fontsize=11)
    ax.grid(True, color="#e4e3df")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=9)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / "optimizer_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    run_comparison()
