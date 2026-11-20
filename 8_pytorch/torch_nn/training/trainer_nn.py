import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import checkpointing_nn

CHECKPOINT_DIR = Path(__file__).resolve().parents[2] / "checkpoints"


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)
    lr: list[float] = field(default_factory=list)
    epoch_time: list[float] = field(default_factory=list)


class Trainer:
    def __init__(self, model: nn.Module, optimizer, scheduler, criterion, scaler, device: torch.device,
                 config: dict) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.config = config

        self.epochs = config.get("epochs", 10)
        self.patience = config.get("patience", math.inf)
        self.max_norm = config.get("max_norm")
        self.use_amp = config.get("use_amp", False)
        self.amp_dtype = config.get("amp_dtype", torch.float16)
        self.scheduler_per_batch = config.get("scheduler_per_batch", False)
        self.checkpoint_path = Path(config.get("checkpoint_path", CHECKPOINT_DIR / "best.pt"))
        self.verbose = config.get("verbose", True)

        self.scaler = scaler if scaler is not None else torch.amp.GradScaler(self.device.type, enabled=False)

        self.history = History()
        self.best_val_loss = math.inf
        self.epoch = 0

    def _autocast(self):
        return torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.use_amp)

    def train_epoch(self, loader) -> dict:
        self.model.train()
        loss_sum = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device, dtype=torch.long)
        n = 0

        for X, y in loader:
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with self._autocast():
                logits = self.model(X)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            if self.max_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None and self.scheduler_per_batch:
                self.scheduler.step()

            loss_sum += loss.detach().float() * len(y)
            correct += (logits.argmax(dim=1) == y).sum()
            n += len(y)

        return {"loss": loss_sum.item() / n, "acc": correct.item() / n}

    @torch.no_grad()
    def val_epoch(self, loader, collect: bool = False) -> dict:
        self.model.eval()
        loss_sum = torch.zeros((), device=self.device)
        correct = torch.zeros((), device=self.device, dtype=torch.long)
        n = 0
        preds, labels = [], []

        for X, y in loader:
            X, y = X.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            with self._autocast():
                logits = self.model(X)
                loss = self.criterion(logits, y)
            pred = logits.argmax(dim=1)
            loss_sum += loss.float() * len(y)
            correct += (pred == y).sum()
            n += len(y)
            if collect:
                preds.append(pred.cpu())
                labels.append(y.cpu())

        out = {"loss": loss_sum.item() / n, "acc": correct.item() / n}
        if collect:
            out["preds"], out["labels"] = torch.cat(preds).numpy(), torch.cat(labels).numpy()
        return out

    def fit(self, train_loader, val_loader) -> History:
        bad_epochs = 0

        for epoch in range(self.epoch + 1, self.epochs + 1):
            t0 = time.perf_counter()
            lr = self.optimizer.param_groups[0]["lr"]
            t = self.train_epoch(train_loader)
            v = self.val_epoch(val_loader)
            if self.scheduler is not None and not self.scheduler_per_batch:
                self.scheduler.step()
            elapsed = time.perf_counter() - t0
            self.epoch = epoch

            h = self.history
            h.train_loss.append(t["loss"])
            h.train_acc.append(t["acc"])
            h.val_loss.append(v["loss"])
            h.val_acc.append(v["acc"])
            h.lr.append(lr)
            h.epoch_time.append(elapsed)

            improved = v["loss"] < self.best_val_loss
            if improved:
                self.best_val_loss = v["loss"]
                self.save_checkpoint(self.checkpoint_path)
                bad_epochs = 0
            else:
                bad_epochs += 1

            if self.verbose:
                print(f"epoch {epoch:3d} | train_loss {t['loss']:.4f} | val_loss {v['loss']:.4f} | "
                      f"train_acc {t['acc']:.4f} | val_acc {v['acc']:.4f} | lr {lr:.2e} | {elapsed:.1f}s"
                      f"{' *' if improved else ''}")

            if bad_epochs >= self.patience:
                if self.verbose:
                    print(f"early stop: no val improvement for {self.patience} epochs")
                break

        if self.checkpoint_path.exists():
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            if self.verbose:
                print(f"restored best weights from epoch {ckpt['epoch']} (val_loss {ckpt['best_val_loss']:.4f})")
        return self.history

    def evaluate(self, test_loader, num_classes: int = 10) -> dict:
        r = self.val_epoch(test_loader, collect=True)
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        np.add.at(cm, (r["labels"], r["preds"]), 1)
        return {"loss": r["loss"], "acc": r["acc"], "confusion_matrix": cm}

    def save_checkpoint(self, path) -> None:
        checkpointing_nn.save_checkpoint(
            path, self.epoch, self.model, self.optimizer, self.scheduler, self.scaler,
            self.best_val_loss, self.history.train_loss, self.history.val_loss,
            {k: v for k, v in self.config.items() if isinstance(v, (int, float, str, bool, tuple, list, type(None)))},
            extra={"history": asdict(self.history)},
        )

    def load_checkpoint(self, path) -> None:
        ckpt = checkpointing_nn.load_checkpoint(path, self.model, self.optimizer, self.scheduler, self.scaler,
                                                map_location=str(self.device))
        self.epoch = ckpt["epoch"]
        self.best_val_loss = ckpt["best_val_loss"]
        if "history" in ckpt:
            self.history = History(**ckpt["history"])
