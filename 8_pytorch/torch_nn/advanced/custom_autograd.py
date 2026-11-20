import torch
from torch import nn


class ClippedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, clip_val: float) -> torch.Tensor:
        ctx.save_for_backward(x)
        ctx.clip_val = clip_val
        return x.clamp(0, clip_val)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        x, = ctx.saved_tensors
        grad = grad_output * ((x > 0) & (x < ctx.clip_val)).to(grad_output.dtype)
        return grad, None


def clipped_relu(x: torch.Tensor, clip_val: float = 6.0) -> torch.Tensor:
    return ClippedReLU.apply(x, clip_val)


class ClippedReLUModule(nn.Module):
    def __init__(self, clip_val: float = 6.0) -> None:
        super().__init__()
        self.clip_val = clip_val

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return clipped_relu(x, self.clip_val)

    def extra_repr(self) -> str:
        return f"clip_val={self.clip_val}"


def verify_gradcheck(clip_val: float = 6.0, seed: int = 0) -> bool:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(50, dtype=torch.float64, generator=g) * 4

    x = x[(x.abs() > 1e-2) & ((x - clip_val).abs() > 1e-2)]

    x.requires_grad_()

    return torch.autograd.gradcheck(lambda t: ClippedReLU.apply(t, clip_val), (x,), eps=1e-6, atol=1e-4)


if __name__ == "__main__":
    print("gradcheck:", verify_gradcheck())
    x = torch.linspace(-2, 8, 6, requires_grad=True)
    y = clipped_relu(x, 6.0)
    y.sum().backward()
    print("x     ", x.detach().tolist())
    print("y     ", y.detach().tolist())
    print("dy/dx ", x.grad.tolist(), "| grad_fn:", y.grad_fn.__class__.__name__)
    print("same as F.hardtanh(x, 0, 6):", torch.equal(y, nn.functional.hardtanh(x, 0, 6)))
