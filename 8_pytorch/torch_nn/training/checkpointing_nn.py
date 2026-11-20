import random
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch import nn


def save_checkpoint(path: Path, epoch: int, model, optimizer, scheduler, scaler,
                    best_val_loss: float, train_losses: list, val_losses: list, config: dict,
                    generator: torch.Generator | None = None, extra: dict | None = None) -> None:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "config": config,
        "torch_rng": torch.get_rng_state(),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "generator_state": generator.get_state() if generator is not None else None,
        **(extra or {}),
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Path, model, optimizer=None, scheduler=None, scaler=None,
                    map_location: str = "cpu", generator: torch.Generator | None = None) -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)

    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler is not None and ckpt["scheduler_state_dict"] is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler is not None and ckpt["scaler_state_dict"] is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])

    torch.set_rng_state(ckpt["torch_rng"])
    np.random.set_state(ckpt["numpy_rng"])
    random.setstate(ckpt["python_rng"])
    if ckpt.get("cuda_rng") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
    if generator is not None and ckpt.get("generator_state") is not None:
        generator.set_state(ckpt["generator_state"])

    return ckpt


def _build(seed: int):
    from .reproducibility_nn import set_seed

    set_seed(seed)
    X = torch.randn(512, 20)
    y = (X[:, :3].sum(dim=1) > 0).long()
    model = nn.Sequential(nn.Linear(20, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    g = torch.Generator().manual_seed(seed)
    return model, opt, sched, scaler, X, y, g


def _run_epochs(model, opt, sched, X, y, g, epochs: int, losses: list) -> None:
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        total = 0.0
        for s in range(0, len(X), 64):
            idx = perm[s:s + 64]
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(X[idx]), y[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        sched.step()
        losses.append(total / len(X))


def verify_resume(epochs_total: int = 10, split_at: int = 5, seed: int = 0) -> bool:
    model, opt, sched, scaler, X, y, g = _build(seed)
    losses_a: list[float] = []
    _run_epochs(model, opt, sched, X, y, g, epochs_total, losses_a)

    model, opt, sched, scaler, X, y, g = _build(seed)
    losses_b: list[float] = []
    _run_epochs(model, opt, sched, X, y, g, split_at, losses_b)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "ckpt.pt"
        save_checkpoint(path, split_at, model, opt, sched, scaler, min(losses_b), losses_b, [], {"seed": seed},
                        generator=g)

        model, opt, sched, scaler, X2, y2, g = _build(seed + 1)
        ckpt = load_checkpoint(path, model, opt, sched, scaler, generator=g)
    losses_b = list(ckpt["train_losses"])
    _run_epochs(model, opt, sched, X, y, g, epochs_total - ckpt["epoch"], losses_b)

    print(f"uninterrupted: {losses_a[-1]:.10f}\nresumed:       {losses_b[-1]:.10f}")
    return losses_a == losses_b


if __name__ == "__main__":
    print("identical curves after resume:", verify_resume())
