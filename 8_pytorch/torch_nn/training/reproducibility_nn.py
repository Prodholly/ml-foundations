import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def tiny_run(seed: int, epochs: int = 3) -> list[float]:
    from torch import nn

    set_seed(seed)
    X = torch.randn(512, 20)
    y = (X[:, :2].sum(dim=1) > 0).long()
    model = nn.Sequential(nn.Linear(20, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, 2))
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, y), batch_size=64, shuffle=True,
                                         generator=torch.Generator().manual_seed(seed))
    losses = []
    for _ in range(epochs):
        total = 0.0
        for Xb, yb in loader:
            opt.zero_grad()
            loss = nn.functional.cross_entropy(model(Xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(Xb)
        losses.append(total / len(X))
    return losses


def verify_reproducibility(run_fn=tiny_run, epochs: int = 3) -> tuple[list[float], list[float], list[float]]:
    a = run_fn(seed=0, epochs=epochs)
    b = run_fn(seed=0, epochs=epochs)
    c = run_fn(seed=1, epochs=epochs)
    assert a == b, "same seed gave different curves"
    assert a != c, "different seeds gave the same curve"
    print(f"seed 0: {[round(v, 5) for v in a]}\nseed 0: {[round(v, 5) for v in b]}\nseed 1: {[round(v, 5) for v in c]}")
    return a, b, c


if __name__ == "__main__":
    verify_reproducibility()
