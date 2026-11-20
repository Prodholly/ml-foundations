import torch
from torch import nn

from .activations_nn import get_activation


def conv_bn(c_in: int, c_out: int, activation: str = "relu") -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(c_out),
        get_activation(activation),
    )


class Residual(nn.Module):
    def __init__(self, channels: int, activation: str = "relu") -> None:
        super().__init__()
        self.body = nn.Sequential(conv_bn(channels, channels, activation), conv_bn(channels, channels, activation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class CNN(nn.Module):
    def __init__(self, num_classes: int = 10, widths: tuple[int, int, int, int] = (64, 128, 256, 512),
                 activation: str = "relu", dropout_p: float = 0.0) -> None:
        super().__init__()
        w0, w1, w2, w3 = widths

        self.prep = conv_bn(3, w0, activation)
        self.layer1 = nn.Sequential(conv_bn(w0, w1, activation), nn.MaxPool2d(2), Residual(w1, activation))
        self.layer2 = nn.Sequential(conv_bn(w1, w2, activation), nn.MaxPool2d(2))
        self.layer3 = nn.Sequential(conv_bn(w2, w3, activation), nn.MaxPool2d(2), Residual(w3, activation))
        self.pool = nn.AdaptiveMaxPool2d(1)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_p)
        self.head = nn.Linear(w3, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.flatten(self.pool(x))
        return self.head(self.dropout(x))
