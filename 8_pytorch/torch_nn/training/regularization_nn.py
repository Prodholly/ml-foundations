import torch
from torch import nn


def dropout_demo() -> None:
    drop = nn.Dropout(p=0.5)
    x = torch.ones(4, 8)

    drop.train()
    print("dropout train:\n", drop(x))

    drop.eval()
    print("dropout eval, equal to x:", torch.equal(drop(x), x))


def batchnorm_demo() -> None:
    bn = nn.BatchNorm1d(4)
    print("bn start   mean", bn.running_mean.tolist(), "var", bn.running_var.tolist(),
          "batches", bn.num_batches_tracked.item())

    bn.train()
    bn(torch.randn(32, 4) * 3 + 5)
    print("bn train   mean", [round(v, 2) for v in bn.running_mean.tolist()],
          "var", [round(v, 2) for v in bn.running_var.tolist()], "batches", bn.num_batches_tracked.item())

    before = bn.running_mean.clone()
    bn.eval()
    bn(torch.randn(32, 4) * 10)
    print("bn eval    stats frozen:", torch.equal(bn.running_mean, before), "batches", bn.num_batches_tracked.item())


def layernorm_vs_batchnorm() -> None:
    x = torch.randn(1, 16)

    try:
        nn.BatchNorm1d(16).train()(x)
    except ValueError as e:
        print("batchnorm on 1 sample:", e)

    y = nn.LayerNorm(16)(x)
    print(f"layernorm on 1 sample: mean {y.mean().item():.3f}, std {y.std(unbiased=False).item():.3f}")


def dropout2d_demo() -> None:
    x = torch.ones(2, 8, 4, 4)
    y = nn.Dropout2d(0.5).train()(x)

    per_channel = y.flatten(2)
    whole = bool(((per_channel == 0).all(-1) | (per_channel == 2).all(-1)).all())
    print("dropout2d channel values:", per_channel[:, :, 0].tolist(), "| whole channels only:", whole)


def _penalized_weights(model: nn.Module):
    return [p for name, p in model.named_parameters() if name.endswith("weight") and p.dim() > 1]


def l1_penalty(model: nn.Module, lam: float) -> torch.Tensor:
    return lam * sum(p.abs().sum() for p in _penalized_weights(model))


def l2_penalty(model: nn.Module, lam: float) -> torch.Tensor:
    return lam * sum((p ** 2).sum() for p in _penalized_weights(model))


def clip_gradients(model: nn.Module, max_norm: float = 1.0) -> float:
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return total.item()


def total_grad_norm(model: nn.Module) -> float:
    return torch.norm(torch.stack([p.grad.norm() for p in model.parameters() if p.grad is not None])).item()


if __name__ == "__main__":
    torch.manual_seed(0)
    dropout_demo()
    batchnorm_demo()
    layernorm_vs_batchnorm()
    dropout2d_demo()

    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
    x = torch.randn(16, 10)
    loss = model(x).pow(2).sum() * 1000 + l2_penalty(model, 1e-4) + l1_penalty(model, 1e-5)
    loss.backward()
    before = clip_gradients(model, 1.0)
    print(f"clip: norm before {before:.1f}, after {total_grad_norm(model):.4f}")
