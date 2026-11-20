import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from numpy_mlp.figures import plot_training_curves

from .advanced.custom_losses import LabelSmoothingCE
from .data import CLASSES, build_loaders
from .debugging.loss_landscape import loss_surface, plot_landscape
from .models.cnn import CNN
from .models.init_nn import initialize_model
from .models.mlp import MLP
from .training.reproducibility_nn import set_seed
from .training.trainer_nn import CHECKPOINT_DIR, Trainer, get_device

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

CONFIG = {
    "model": "cnn",
    "cnn_widths": (32, 64, 128, 256),
    "hidden_sizes": (1024, 512, 256),
    "dropout_p": 0.3,
    "use_batchnorm": True,
    "lr": 1e-3,
    "weight_decay": 0.01,
    "max_lr": 3e-3,
    "epochs": 30,
    "batch_size": 256,
    "max_norm": 1.0,
    "label_smoothing": 0.1,
    "patience": 8,
    "seed": 0,
    "num_workers": 4,
}


def build_model(c: dict) -> nn.Module:
    if c["model"] == "cnn":
        model = CNN(widths=c["cnn_widths"], dropout_p=c["dropout_p"])
    else:
        model = MLP(3 * 32 * 32, c["hidden_sizes"], 10, "relu", c["dropout_p"], c["use_batchnorm"])
    return initialize_model(model, "kaiming_normal_fan_in")


def plot_confusion_matrix(cm: np.ndarray, name: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frac = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(10):
        for j in range(10):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=7.5,
                    color="white" if frac[i, j] > 0.5 else "#0b0b0b")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"CIFAR-10 test confusion matrix, acc {np.trace(cm) / cm.sum():.4f}", fontsize=11)
    fig.colorbar(im, ax=ax, label="fraction of the true class")
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def main(model_name: str = "cnn") -> dict:
    c = {**CONFIG, "model": model_name}
    prefix = "cifar10" if model_name == "cnn" else "cifar10_mlp"

    set_seed(c["seed"])
    device = torch.device(get_device())
    print("device:", device)

    train_loader, val_loader, test_loader = build_loaders(c["batch_size"], c["seed"], device.type, c["num_workers"])
    print(train_loader.dataset, val_loader.dataset, test_loader.dataset)

    model = build_model(c).to(device)
    print(f"{model_name}: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=c["max_lr"], total_steps=c["epochs"] * len(train_loader), pct_start=0.3)
    scaler = torch.amp.GradScaler(device.type)
    criterion = LabelSmoothingCE(c["label_smoothing"])

    trainer = Trainer(model, optimizer, scheduler, criterion, scaler, device, config={
        **c, "scheduler_per_batch": True, "use_amp": True,
        "checkpoint_path": CHECKPOINT_DIR / f"{prefix}_best.pt",
    })
    history = trainer.fit(train_loader, val_loader)

    results = trainer.evaluate(test_loader)
    acc = results["acc"]
    print(f"\ntest loss {results['loss']:.4f}  acc {acc:.4f}  {'> 85% target met' if acc > 0.85 else 'below the 85% target'}")
    print(f"mean epoch time {np.mean(history.epoch_time):.1f}s")

    best_epoch = int(np.argmin(history.val_loss)) + 1
    plot_training_curves(history.train_loss, history.val_loss, history.train_acc, history.val_acc,
                         best_epoch=best_epoch, name=f"{prefix}_training_curves.png")
    plot_confusion_matrix(results["confusion_matrix"], f"{prefix}_confusion_matrix.png")

    Xb, yb = next(iter(val_loader))
    alphas, betas, Z = loss_surface(model, Xb[:256].to(device), yb[:256].to(device), nn.CrossEntropyLoss(),
                                    steps=25, span=1.0)
    plot_landscape(alphas, betas, Z, name=f"{prefix}_loss_landscape.png",
                   title=f"CIFAR-10 {model_name}: loss around the trained weights (filter-normalized)")

    summary = {
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()},
        "device": device.type, "best_epoch": best_epoch,
        "history": {k: v for k, v in history.__dict__.items()},
        "test": {"loss": results["loss"], "acc": acc}, "confusion_matrix": results["confusion_matrix"].tolist(),
        "loss_landscape": {"min": Z.min().item(), "center": Z[12, 12].item(), "max": Z.max().item()},
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{prefix}_run.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "cnn")
