import torch
import torch.nn.functional as F
from torch import nn


class LabelSmoothingCE(nn.Module):
    def __init__(self, eps: float = 0.1) -> None:
        super().__init__()
        if not 0 <= eps < 1:
            raise ValueError(f"eps must be in [0, 1), got {eps}")
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        C = logits.shape[1]
        log_probs = F.log_softmax(logits, dim=1)

        smooth = torch.full_like(log_probs, self.eps / C)
        smooth.scatter_(1, targets.unsqueeze(1), 1 - self.eps + self.eps / C)

        return -(smooth * log_probs).sum(dim=1).mean()

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("alpha", alpha)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_p = F.log_softmax(logits, dim=1)

        log_pt = log_p.gather(1, targets.unsqueeze(1)).squeeze(1)

        pt = log_pt.exp()

        loss = -((1 - pt) ** self.gamma) * log_pt

        if self.alpha is not None:
            loss = loss * self.alpha[targets]

        return loss.mean()

    def extra_repr(self) -> str:
        return f"gamma={self.gamma}"


if __name__ == "__main__":
    torch.manual_seed(0)
    logits, y = torch.randn(32, 10), torch.randint(0, 10, (32,))
    ls = LabelSmoothingCE(0.1)(logits, y)
    print(f"label smoothing  mine {ls.item():.6f} | F.cross_entropy(label_smoothing=0.1) "
          f"{F.cross_entropy(logits, y, label_smoothing=0.1).item():.6f}")
    print(f"focal gamma=0    mine {FocalLoss(0.0)(logits, y).item():.6f} | cross_entropy {F.cross_entropy(logits, y).item():.6f}")
    print(f"focal gamma=2    {FocalLoss(2.0)(logits, y).item():.6f}  (smaller: confident examples count less)")
