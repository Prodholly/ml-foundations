from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import prepare_mnist
from .figures import plot_init_comparison
from .init_schemes import INIT_SCHEMES
from .losses import softmax_cross_entropy
from .model import NumpyMLP
from .optimizers import SGDMomentum
from .trainer import train

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
LAYER_SIZES = [784, 256, 128, 10]
SEED = 8


def step0_grad_norms(model: NumpyMLP, X: np.ndarray, Y: np.ndarray) -> list[float]:
    model.zero_grad()
    _, dL_dZ = softmax_cross_entropy(model.forward(X), Y)
    model.backward(dL_dZ)
    norms = [float(np.linalg.norm(layer.dW)) for layer in model.layers]
    model.zero_grad()
    return norms


def run(epochs: int = 5, batch_size: int = 128) -> dict[str, dict[str, list]]:
    data = prepare_mnist(seed=SEED)
    results: dict[str, dict[str, list]] = {}

    for scheme in INIT_SCHEMES:
        print(f"\n== init: {scheme} ==")
        model = NumpyMLP(LAYER_SIZES, activation="relu", init_scheme=scheme, seed=SEED)
        grad_norms: list[list[float]] = [step0_grad_norms(model, data.X_train[:batch_size], data.Y_train[:batch_size])]
        _, val_losses, _, val_accs = train(
            model, SGDMomentum(lr=0.01, beta=0.9),
            data.X_train, data.Y_train, data.X_val, data.Y_val,
            epochs=epochs, batch_size=batch_size, patience=epochs,
            seed=SEED, grad_norms=grad_norms,
        )
        results[scheme] = {"grad_norms": grad_norms, "val_losses": val_losses, "val_accs": val_accs}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "init_comparison.json").write_text(json.dumps(results, indent=2))
    print("\nsaved", plot_init_comparison(results))
    return results


if __name__ == "__main__":
    run()
