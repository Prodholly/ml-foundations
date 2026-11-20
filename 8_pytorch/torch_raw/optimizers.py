import torch


class RawOptimizer:
    def __init__(self, params: list[torch.Tensor], lr: float) -> None:
        self.params = list(params)
        self.lr = lr

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    def step(self) -> None:
        raise NotImplementedError


class SGDMomentum(RawOptimizer):
    def __init__(self, params: list[torch.Tensor], lr: float = 0.01, beta: float = 0.9) -> None:
        super().__init__(params, lr)
        self.beta = beta
        self.velocity = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def step(self) -> None:
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            v.mul_(self.beta).sub_(self.lr * p.grad)
            p.add_(v)


class Adam(RawOptimizer):
    def __init__(
        self, params: list[torch.Tensor], lr: float = 1e-3,
        beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    @torch.no_grad()
    def _adam_update(self) -> None:
        self.t += 1

        bias1 = 1 - self.beta1 ** self.t
        bias2 = 1 - self.beta2 ** self.t

        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)
            m_hat = m / bias1
            v_hat = v / bias2
            p.sub_(self.lr * m_hat / (v_hat.sqrt() + self.eps))

    def step(self) -> None:
        self._adam_update()


class AdamW(Adam):
    def __init__(self, params: list[torch.Tensor], lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.01) -> None:
        super().__init__(params, lr, beta1, beta2, eps)
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self) -> None:
        for p in self.params:
            p.mul_(1 - self.lr * self.weight_decay)

        self._adam_update()
