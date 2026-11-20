import torch
from torch import nn

from .activations_nn import get_activation


class MLP(nn.Module):
    def __init__(
        self,
        in_features: int = 3 * 32 * 32,
        hidden_sizes: tuple[int, ...] = (1024, 512, 256),
        out_features: int = 10,
        activation: str = "relu",
        dropout_p: float = 0.0,
        use_batchnorm: bool = False,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.flatten = nn.Flatten()

        blocks = []
        prev = in_features
        for h in hidden_sizes:
            parts: list[nn.Module] = [nn.Linear(prev, h, bias=not use_batchnorm)]
            if use_batchnorm:
                parts.append(nn.BatchNorm1d(h))
            parts.append(get_activation(activation))
            if dropout_p > 0:
                parts.append(nn.Dropout(dropout_p))
            blocks.append(nn.Sequential(*parts))
            prev = h

        self.layers = nn.ModuleList(blocks)

        self.head = nn.Linear(prev, out_features)

        self.register_buffer("input_mean", torch.zeros(in_features))
        self.momentum = 0.1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)

        if self.training:
            with torch.no_grad():
                self.input_mean.mul_(1 - self.momentum).add_(self.momentum * x.mean(dim=0).to(self.input_mean.dtype))

        for block in self.layers:
            x = block(x)

        return self.head(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def expected_parameter_count(in_features: int, hidden_sizes: tuple[int, ...], out_features: int,
                             use_batchnorm: bool) -> int:
    sizes = [in_features, *hidden_sizes, out_features]

    weights = sum(i * o for i, o in zip(sizes[:-1], sizes[1:]))
    biases = out_features + (0 if use_batchnorm else sum(hidden_sizes))

    bn = 2 * sum(hidden_sizes) if use_batchnorm else 0
    return weights + biases + bn


def demo_module_patterns(model: MLP) -> None:
    print("== named_parameters")
    for name, p in model.named_parameters():
        print(f"  {name:22s} {tuple(p.shape)!s:14s} requires_grad={p.requires_grad}")

    print("== named_modules")
    for name, m in model.named_modules():
        print(f"  {name or '(root)':12s} {m.__class__.__name__}")

    print("== children:", [c.__class__.__name__ for c in model.children()])

    print("== named_buffers")
    for name, b in model.named_buffers():
        print(f"  {name:28s} {tuple(b.shape)}")

    model.train()
    print("== train():", model.training, model.layers[0][-1].training)
    model.eval()
    print("== eval(): ", model.training, model.layers[0][-1].training)

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print("== to:", next(model.parameters()).device)
    model.half()
    print("== half:", next(model.parameters()).dtype, "| buffer:", model.input_mean.dtype)
    model.float()
    print("== float:", next(model.parameters()).dtype)
    model.to("cpu")


if __name__ == "__main__":
    demo_module_patterns(MLP(hidden_sizes=(64, 32), dropout_p=0.2, use_batchnorm=True))
