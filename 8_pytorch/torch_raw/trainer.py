import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

from numpy_mlp.figures import plot_training_curves as plot_curves_phase1

from .model import RawMLP, cross_entropy

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def memory_in_use_mb(device: str) -> float | None:
    if device == "cuda":
        return torch.cuda.memory_allocated() / 1e6
    if device == "mps":
        return torch.mps.current_allocated_memory() / 1e6
    return None


def attach_grad_norm_hooks(model: RawMLP, log: list[list]) -> list:
    handles = []
    for i, layer in enumerate(model.layers):
        log.append([])
        h = layer.W.register_hook(lambda grad, i=i: log[i].append(grad.detach().norm()))
        handles.append(h)
    return handles


def evaluate(model: RawMLP, X: torch.Tensor, y: torch.Tensor, batch_size: int = 2048) -> tuple[float, float]:
    model.eval()
    N = X.shape[0]
    total_loss = torch.zeros((), device=X.device)
    correct = torch.zeros((), device=X.device, dtype=torch.long)

    with torch.no_grad():
        for start in range(0, N, batch_size):
            Xb, yb = X[start:start + batch_size], y[start:start + batch_size]
            logits = model(Xb)
            total_loss += cross_entropy(logits, yb) * Xb.shape[0]
            correct += (logits.argmax(dim=1) == yb).sum()

    model.train()
    return total_loss.item() / N, correct.item() / N


def predict(model: RawMLP, X: torch.Tensor, batch_size: int = 2048) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        preds = torch.cat([model(X[s:s + batch_size]).argmax(dim=1) for s in range(0, X.shape[0], batch_size)])
    model.train()
    return preds


def train(
    model: RawMLP, optimizer, X_train: torch.Tensor, y_train: torch.Tensor,
    X_val: torch.Tensor, y_val: torch.Tensor, epochs: int, batch_size: int, patience: int,
    l2_lambda: float = 0.0, seed: int = 0, verbose: bool = True,
) -> dict[str, list]:
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")

    device = X_train.device
    N = X_train.shape[0]
    steps_per_epoch = math.ceil(N / batch_size)

    grad_log: list[list] = []
    handles = attach_grad_norm_hooks(model, grad_log)

    g = torch.Generator().manual_seed(seed)

    history: dict[str, list] = {k: [] for k in
                                ("train_losses", "val_losses", "train_accs", "val_accs", "epoch_times", "memory_mb")}
    best_val_loss = math.inf
    best_epoch = 0
    best = [p.detach().clone() for p in model.parameters()]
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()

        perm = torch.randperm(N, generator=g).to(device)
        loss_sum = torch.zeros((), device=device)
        correct = torch.zeros((), device=device, dtype=torch.long)

        for start in range(0, N, batch_size):
            idx = perm[start:start + batch_size]
            Xb, yb = X_train[idx], y_train[idx]
            B = Xb.shape[0]

            optimizer.zero_grad()
            logits = model(Xb)
            data_loss = cross_entropy(logits, yb)
            loss = data_loss
            if l2_lambda:
                loss = loss + l2_lambda * sum((layer.W ** 2).sum() for layer in model.layers)
            loss.backward()
            optimizer.step()

            loss_sum += data_loss.detach() * B
            correct += (logits.argmax(dim=1) == yb).sum()

        val_loss, val_acc = evaluate(model, X_val, y_val)
        train_loss = loss_sum.item() / N
        elapsed = time.perf_counter() - t0

        if not math.isfinite(train_loss):
            raise FloatingPointError(f"loss became {train_loss} at epoch {epoch}, lower the lr")

        history["train_losses"].append(train_loss)
        history["train_accs"].append(correct.item() / N)
        history["val_losses"].append(val_loss)
        history["val_accs"].append(val_acc)
        history["epoch_times"].append(elapsed)
        history["memory_mb"].append(memory_in_use_mb(device.type))

        if verbose:
            mem = history["memory_mb"][-1]
            mem_s = f" | mem {mem:.0f}MB" if mem is not None else ""
            print(
                f"epoch {epoch:3d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
                f"train_acc {history['train_accs'][-1]:.4f} | val_acc {val_acc:.4f} | {elapsed:.1f}s{mem_s}"
            )

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best = [p.detach().clone() for p in model.parameters()]
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"early stop: no val improvement for {patience} epochs, best was epoch {best_epoch}")
                break

    with torch.no_grad():
        for p, saved in zip(model.parameters(), best):
            p.copy_(saved)

    for h in handles:
        h.remove()

    history["grad_norms"] = [torch.stack(layer_log).cpu().tolist() for layer_log in grad_log]
    history["steps_per_epoch"] = steps_per_epoch
    history["best_epoch"] = best_epoch
    return history


def plot_grad_norm_heatmap(grad_norms: list[list[float]], steps_per_epoch: int,
                           name: str = "grad_norms_raw.png") -> Path:
    epochs = len(grad_norms[0]) // steps_per_epoch
    per_epoch = [np.asarray(g[: epochs * steps_per_epoch]).reshape(epochs, steps_per_epoch).mean(axis=1)
                 for g in grad_norms]

    M = np.stack(per_epoch)

    fig, ax = plt.subplots(figsize=(max(6, 0.45 * epochs + 2), 2.6))
    im = ax.imshow(M, aspect="auto", cmap="viridis", norm=LogNorm(vmin=M.min(), vmax=M.max()))
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white")

    ax.set_yticks(range(M.shape[0]), [f"layer {i + 1}" for i in range(M.shape[0])])
    ax.set_xticks(range(epochs), [str(e + 1) for e in range(epochs)])
    ax.set_xlabel("epoch")
    ax.set_title("mean ||dL/dW|| per layer per epoch (hooks on W)", fontsize=11)
    fig.colorbar(im, ax=ax, label="grad norm (log)")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def plot_training_curves(history: dict[str, list]) -> Path:
    return plot_curves_phase1(
        history["train_losses"], history["val_losses"], history["train_accs"], history["val_accs"],
        best_epoch=history.get("best_epoch"), name="torch_raw_training.png",
    )
