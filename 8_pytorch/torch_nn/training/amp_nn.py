import time

import torch
from torch import nn


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def train_epoch_amp(model: nn.Module, loader, optimizer, criterion, scaler, device: torch.device,
                    use_amp: bool = True, max_norm: float = 1.0) -> float:
    model.train()
    total = torch.zeros((), device=device)
    n = 0
    for X, y in loader:
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(X)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        scaler.step(optimizer)
        scaler.update()

        total += loss.detach() * len(X)
        n += len(X)
    return total.item() / n


def peak_memory_mb(device: torch.device) -> float | None:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / 1e6
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 1e6
    return None


def benchmark_amp(build_model, loader_fn, device: torch.device, epochs: int = 1, seed: int = 0) -> dict[str, dict[str, float]]:
    from .reproducibility_nn import set_seed

    results = {}
    for use_amp in (False, True):
        set_seed(seed)
        model = build_model().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        criterion = nn.CrossEntropyLoss()

        train_epoch_amp(model, [next(iter(loader_fn()))], optimizer, criterion, scaler, device, use_amp)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        elif device.type == "mps":
            torch.mps.empty_cache()
        sync(device)
        t0 = time.perf_counter()
        losses = [train_epoch_amp(model, loader_fn(), optimizer, criterion, scaler, device, use_amp)
                  for _ in range(epochs)]
        sync(device)
        sec = (time.perf_counter() - t0) / epochs

        key = "fp16_autocast" if use_amp else "fp32"
        results[key] = {"s_per_epoch": sec, "peak_mb": peak_memory_mb(device), "final_loss": losses[-1]}
        print(f"{key:14s} {sec:6.2f} s/epoch | memory {results[key]['peak_mb']} MB | loss {losses[-1]:.4f}")
    return results


if __name__ == "__main__":
    from ..data import load_cifar10_tensors, tensor_batches
    from ..models.cnn import CNN
    from .trainer_nn import get_device

    device = torch.device(get_device())
    data = load_cifar10_tensors(device.type)
    g = torch.Generator().manual_seed(0)
    print("device:", device)
    benchmark_amp(lambda: CNN(), lambda: tensor_batches(data["X_train"], data["y_train"], 256, g), device)
