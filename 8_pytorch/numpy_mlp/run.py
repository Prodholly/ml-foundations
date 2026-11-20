from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .data import prepare_mnist
from .figures import plot_training_curves
from .model import NumpyMLP
from .optimizers import Adam
from .trainer import confusion_matrix, evaluate, train

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

CONFIG = {
    "layer_sizes": [784, 256, 128, 10],
    "activation": "relu",
    "init_scheme": "he",
    "lr": 1e-3,
    "l2_lambda": 1e-4,
    "epochs": 30,
    "batch_size": 128,
    "patience": 5,
    "seed": 8,
}


def main() -> dict:
    c = CONFIG
    data = prepare_mnist(seed=c["seed"])
    print(f"train {data.X_train.shape}, val {data.X_val.shape}, test {data.X_test.shape}, final {data.X_final.shape}")
    print(f"normalizer from train only: mean {data.mean:.3f}, std {data.std:.3f}")

    model = NumpyMLP(c["layer_sizes"], activation=c["activation"], init_scheme=c["init_scheme"], seed=c["seed"])
    print(model)

    epoch_times: list[float] = []
    train_losses, val_losses, train_accs, val_accs = train(
        model, Adam(lr=c["lr"]),
        data.X_train, data.Y_train, data.X_val, data.Y_val,
        epochs=c["epochs"], batch_size=c["batch_size"], patience=c["patience"],
        l2_lambda=c["l2_lambda"], seed=c["seed"], epoch_times=epoch_times,
    )
    best_epoch = int(np.argmin(val_losses)) + 1

    test_loss, test_acc = evaluate(model, data.X_test, data.Y_test)
    final_loss, final_acc = evaluate(model, data.X_final, data.Y_final)
    cm = confusion_matrix(data.Y_final.argmax(axis=1), model.predict(data.X_final))

    print(f"\nbest epoch {best_epoch} of {len(val_losses)}, mean epoch time {np.mean(epoch_times):.2f}s")
    print(f"test split (6k)    loss {test_loss:.4f}  acc {test_acc:.4f}")
    print(f"held-out 10k test  loss {final_loss:.4f}  acc {final_acc:.4f}  {'> 97% target met' if final_acc > 0.97 else 'below the 97% target'}")
    print("\nconfusion matrix on the 10k test set (rows true, cols predicted)")
    print("      " + " ".join(f"{j:5d}" for j in range(10)))
    for i, row in enumerate(cm):
        print(f"{i:5d} " + " ".join(f"{v:5d}" for v in row))

    plot_training_curves(train_losses, val_losses, train_accs, val_accs, best_epoch=best_epoch)

    summary = {
        "config": c,
        "epochs_run": len(val_losses),
        "best_epoch": best_epoch,
        "epoch_times_s": epoch_times,
        "train_losses": train_losses, "val_losses": val_losses,
        "train_accs": train_accs, "val_accs": val_accs,
        "test_split": {"loss": test_loss, "acc": test_acc},
        "final_test": {"loss": final_loss, "acc": final_acc},
        "confusion_matrix": cm.tolist(),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "numpy_run.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main()
