import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "6_grad_check"))
from grad_check.numerical import numerical_gradient


def rel_error(a: np.ndarray, n: np.ndarray) -> float:
    return float(np.max(np.abs(a - n) / np.maximum(np.abs(a) + np.abs(n), 1e-12)))


def numerical_check_module(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion) -> dict[str, float]:
    model.double().eval()
    X = X.double()

    model.zero_grad()
    criterion(model(X), y).backward()
    grads = {n: p.grad.detach().clone().numpy() for n, p in model.named_parameters()}

    errors = {}
    for name, p in model.named_parameters():
        def f(values: np.ndarray, p=p) -> float:
            with torch.no_grad():
                p.copy_(torch.from_numpy(values))
                return criterion(model(X), y).item()

        original = p.detach().numpy().copy()
        num = numerical_gradient(f, original.copy())
        with torch.no_grad():
            p.copy_(torch.from_numpy(original))
        errors[name] = rel_error(grads[name], num)

    return errors


def gradcheck_module(model: nn.Module, X: torch.Tensor, y: torch.Tensor, criterion) -> bool:
    model.double().eval()

    names = [n for n, _ in model.named_parameters()]
    params = tuple(p.detach().clone().requires_grad_() for p in model.parameters())

    def f(*ps):
        out = torch.func.functional_call(model, dict(zip(names, ps)), (X.double(),))
        return criterion(out, y)

    return torch.autograd.gradcheck(f, params, eps=1e-6, atol=1e-4)


def tiny_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(6, 5), nn.SiLU(), nn.Linear(5, 3))


if __name__ == "__main__":
    torch.manual_seed(0)
    X, y = torch.randn(4, 6), torch.randint(0, 3, (4,))
    criterion = nn.CrossEntropyLoss()

    errs = numerical_check_module(tiny_model(), X, y, criterion)
    for name, e in errs.items():
        print(f"  day 6 checker  {name:10s} rel error {e:.2e}")
    print("all < 1e-5:", all(e < 1e-5 for e in errs.values()))
    print("torch.autograd.gradcheck:", gradcheck_module(tiny_model(), X, y, criterion))
