import torch


def dropout(x: torch.Tensor, p: float, training: bool) -> torch.Tensor:
    if not 0 <= p < 1:
        raise ValueError(f"dropout p must be in [0, 1), got {p}")

    if not training or p == 0:
        return x

    mask = torch.bernoulli(torch.full(x.shape, 1 - p, device=x.device, dtype=x.dtype))

    return x * mask / (1 - p)


class RawBatchNorm1d:
    def __init__(self, num_features: int, momentum: float = 0.1, eps: float = 1e-5,
                 device: str = "cpu", dtype: torch.dtype = torch.float32) -> None:
        D = num_features

        self.gamma = torch.ones(D, device=device, dtype=dtype, requires_grad=True)
        self.beta = torch.zeros(D, device=device, dtype=dtype, requires_grad=True)

        self.running_mean = torch.zeros(D, device=device, dtype=dtype)
        self.running_var = torch.ones(D, device=device, dtype=dtype)

        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)

            x_hat = (x - mean) / torch.sqrt(var + self.eps)

            with torch.no_grad():
                self.running_mean.mul_(1 - self.momentum).add_(self.momentum * mean)
                self.running_var.mul_(1 - self.momentum).add_(self.momentum * x.var(dim=0, unbiased=True))
        else:
            x_hat = (x - self.running_mean) / torch.sqrt(self.running_var + self.eps)

        return self.gamma * x_hat + self.beta

    def __call__(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return self.forward(x, training)

    def parameters(self) -> list[torch.Tensor]:
        return [self.gamma, self.beta]
