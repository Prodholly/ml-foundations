import json
from pathlib import Path

import numpy as np
import torch

from numpy_mlp.data import prepare_mnist
from numpy_mlp.trainer import confusion_matrix

from .model import RawMLP
from .optimizers import Adam
from .trainer import evaluate, get_device, plot_grad_norm_heatmap, plot_training_curves, predict, train

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

CONFIG = {
    "layer_sizes": [784, 256, 128, 10],
    "lr": 1e-3,
    "l2_lambda": 1e-4,
    "epochs": 30,
    "batch_size": 128,
    "patience": 5,
    "seed": 8,
}


def to_tensors(X: np.ndarray, Y: np.ndarray, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    X_t = torch.from_numpy(X.astype(np.float32)).to(device)
    y_t = torch.from_numpy(Y.argmax(axis=1).astype(np.int64)).to(device)
    return X_t, y_t


def main(device: str | None = None) -> dict:
    c = CONFIG

    device = device or get_device()
    torch.manual_seed(c["seed"])
    print("device:", device)

    data = prepare_mnist(seed=c["seed"])

    X_tr, y_tr = to_tensors(data.X_train, data.Y_train, device)
    X_va, y_va = to_tensors(data.X_val, data.Y_val, device)
    X_te, y_te = to_tensors(data.X_test, data.Y_test, device)
    X_fi, y_fi = to_tensors(data.X_final, data.Y_final, device)

    model = RawMLP(c["layer_sizes"], device=device, seed=c["seed"])
    optimizer = Adam(model.parameters(), lr=c["lr"])
    print(model)

    history = train(model, optimizer, X_tr, y_tr, X_va, y_va, epochs=c["epochs"], batch_size=c["batch_size"],
                    patience=c["patience"], l2_lambda=c["l2_lambda"], seed=c["seed"])

    test_loss, test_acc = evaluate(model, X_te, y_te)
    final_loss, final_acc = evaluate(model, X_fi, y_fi)
    mean_time = float(np.mean(history["epoch_times"]))
    print(f"\nbest epoch {history['best_epoch']} of {len(history['val_losses'])}, mean epoch time {mean_time:.2f}s")
    print(f"test split (6k)    loss {test_loss:.4f}  acc {test_acc:.4f}")
    print(f"held-out 10k test  loss {final_loss:.4f}  acc {final_acc:.4f}  "
          f"{'> 97% target met' if final_acc > 0.97 else 'below the 97% target'}")

    cm = confusion_matrix(y_fi.cpu().numpy(), predict(model, X_fi).cpu().numpy())
    print("\nconfusion matrix on the 10k test set (rows true, cols predicted)")
    print("      " + " ".join(f"{j:5d}" for j in range(10)))
    for i, row in enumerate(cm):
        print(f"{i:5d} " + " ".join(f"{v:5d}" for v in row))

    plot_training_curves(history)
    plot_grad_norm_heatmap(history["grad_norms"], history["steps_per_epoch"])

    summary = {
        "config": c, "device": device,
        "epochs_run": len(history["val_losses"]), "best_epoch": history["best_epoch"],
        "epoch_times_s": history["epoch_times"], "mean_epoch_time_s": mean_time,
        "train_losses": history["train_losses"], "val_losses": history["val_losses"],
        "train_accs": history["train_accs"], "val_accs": history["val_accs"],
        "memory_mb": history["memory_mb"],
        "test_split": {"loss": test_loss, "acc": test_acc},
        "final_test": {"loss": final_loss, "acc": final_acc},
        "confusion_matrix": cm.tolist(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"torch_raw_run_{device}.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else None)
