from __future__ import annotations

import math
import time

import numpy as np

from .activations import softmax
from .data import batch_generator
from .losses import cross_entropy, softmax_cross_entropy
from .model import NumpyMLP
from .optimizers import Optimizer
from .regularization import add_penalty_grads, penalty


def accuracy(logits_or_probs: np.ndarray, Y_true: np.ndarray) -> float:
    return float(np.mean(logits_or_probs.argmax(axis=1) == Y_true.argmax(axis=1)))


def evaluate(model: NumpyMLP, X: np.ndarray, Y: np.ndarray, batch_size: int = 2048) -> tuple[float, float]:
    was_training = model.training
    model.eval()
    total_loss, total_correct = 0.0, 0
    try:
        for start in range(0, X.shape[0], batch_size):
            Xb, Yb = X[start:start + batch_size], Y[start:start + batch_size]
            probs = softmax(model.forward(Xb))
            total_loss += cross_entropy(probs, Yb) * Xb.shape[0]
            total_correct += int(np.sum(probs.argmax(axis=1) == Yb.argmax(axis=1)))
    finally:
        model.training = was_training
    return total_loss / X.shape[0], total_correct / X.shape[0]


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 10) -> np.ndarray:
    counts = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(counts, (y_true, y_pred), 1)
    return counts


def train(
    model: NumpyMLP,
    optimizer: Optimizer,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    epochs: int,
    batch_size: int,
    patience: int,
    *,
    l2_lambda: float = 0.0,
    l1_lambda: float = 0.0,
    seed: int = 0,
    verbose: bool = True,
    epoch_times: list[float] | None = None,
    grad_norms: list[list[float]] | None = None,
) -> tuple[list[float], list[float], list[float], list[float]]:
    if patience < 1:
        raise ValueError(f"patience must be >= 1, got {patience}")

    steps_per_epoch = math.ceil(X_train.shape[0] / batch_size)
    batches = batch_generator(X_train, Y_train, batch_size, shuffle=True, seed=seed)

    train_losses: list[float] = []
    val_losses: list[float] = []
    train_accs: list[float] = []
    val_accs: list[float] = []

    best_val_loss = math.inf
    best_state = model.get_state()
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        norm_sums = np.zeros(len(model.layers))

        for _ in range(steps_per_epoch):
            Xb, Yb = next(batches)
            B = Xb.shape[0]

            model.zero_grad()
            logits = model.forward(Xb)
            data_loss, dL_dZ = softmax_cross_entropy(logits, Yb)
            model.backward(dL_dZ)

            if grad_norms is not None:
                norm_sums += [np.linalg.norm(layer.dW) for layer in model.layers]

            add_penalty_grads(model, l2_lambda, l1_lambda)

            optimizer.step(model)

            if not np.isfinite(data_loss):
                raise FloatingPointError(f"loss became {data_loss} at epoch {epoch}, lower the lr")

            loss_sum += data_loss * B
            correct += int(np.sum(logits.argmax(axis=1) == Yb.argmax(axis=1)))
            seen += B

        val_loss, val_acc = evaluate(model, X_val, Y_val)
        elapsed = time.perf_counter() - t0

        train_losses.append(loss_sum / seen)
        train_accs.append(correct / seen)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        if epoch_times is not None:
            epoch_times.append(elapsed)
        if grad_norms is not None:
            grad_norms.append((norm_sums / steps_per_epoch).tolist())

        if verbose:
            reg = penalty(model, l2_lambda, l1_lambda)
            print(
                f"epoch {epoch:3d} | train_loss {train_losses[-1]:.4f} | val_loss {val_loss:.4f} | "
                f"train_acc {train_accs[-1]:.4f} | val_acc {val_acc:.4f} | penalty {reg:.4f} | {elapsed:.1f}s"
            )

        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            best_state = model.get_state()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                if verbose:
                    print(f"early stop: no val improvement for {patience} epochs, best was epoch {best_epoch}")
                break

    model.set_state(best_state)
    return train_losses, val_losses, train_accs, val_accs
