import math
from typing import Callable

import torch

from .layers import RawLinear
from .regularization import dropout


def cross_entropy(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    N = logits.shape[0]

    log_probs = logits - torch.logsumexp(logits, dim=1, keepdim=True)

    true_class_log_probs = log_probs[torch.arange(N, device=logits.device), y]

    return -true_class_log_probs.mean()


ACTIVATIONS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "relu": lambda z: z.clamp(min=0),
    "gelu": lambda z: 0.5 * z * (1 + torch.tanh(math.sqrt(2 / math.pi) * (z + 0.044715 * z**3))),
    "swish": lambda z: z * torch.sigmoid(z),
}


class RawMLP:
    def __init__(
        self,
        layer_sizes: list[int],
        activation: str = "relu",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        dropout_p: float = 0.0,
        seed: int = 0,
    ) -> None:
        if activation not in ACTIVATIONS:
            raise ValueError(f"unknown activation {activation!r}, pick one of {list(ACTIVATIONS)}")

        torch.manual_seed(seed)

        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.device = device

        self.layers = [RawLinear(i, o, device, dtype) for i, o in zip(layer_sizes[:-1], layer_sizes[1:])]

        self.activation = ACTIVATIONS[activation]

        self.dropout_p = dropout_p

        self.training = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = x
        for layer in self.layers[:-1]:
            z = layer(a)
            a = self.activation(z)
            if self.dropout_p > 0:
                a = dropout(a, self.dropout_p, training=self.training)
        return self.layers[-1](a)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def parameters(self) -> list[torch.Tensor]:
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self) -> None:
        for layer in self.layers:
            layer.zero_grad()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def train(self) -> "RawMLP":
        self.training = True
        return self

    def eval(self) -> "RawMLP":
        self.training = False
        return self

    def sgd_step(self, lr: float) -> None:
        with torch.no_grad():
            for p in self.parameters():
                p -= lr * p.grad

    def __repr__(self) -> str:
        arrows = " -> ".join(str(s) for s in self.layer_sizes)
        return f"RawMLP({arrows}, {self.activation_name}, dropout={self.dropout_p}, params={self.num_params():,})"
