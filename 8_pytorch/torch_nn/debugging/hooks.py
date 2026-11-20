import torch
from torch import nn


def make_forward_hook(name: str, store: dict):
    def hook(module, inputs, output):
        store.setdefault(name, []).append((output.mean().item(), output.std().item()))
    return hook


def make_backward_hook(name: str, store: dict):
    def hook(module, grad_input, grad_output):
        store.setdefault(name, []).append(grad_output[0].norm().item())
    return hook


def attach_hooks(model: nn.Module, module_types=(nn.Linear,)) -> tuple[list, dict, dict]:
    fwd: dict[str, list] = {}
    bwd: dict[str, list] = {}
    handles = []
    for name, m in model.named_modules():
        if isinstance(m, module_types):
            handles.append(m.register_forward_hook(make_forward_hook(name, fwd)))
            handles.append(m.register_full_backward_hook(make_backward_hook(name, bwd)))
    return handles, fwd, bwd


def remove_hooks(handles: list) -> None:
    for h in handles:
        h.remove()


def anomaly_demo() -> str:
    x = torch.tensor([0.0], requires_grad=True)
    try:
        with torch.autograd.detect_anomaly():
            loss = (x.sqrt() * 0).sum()
            loss.backward()
    except RuntimeError as e:
        msg = str(e).splitlines()[0]
        print("anomaly caught:", msg)
        return msg
    return ""


if __name__ == "__main__":
    import warnings

    from ..models.mlp import MLP

    torch.manual_seed(0)
    model = MLP(hidden_sizes=(256, 128), use_batchnorm=False)
    handles, fwd, bwd = attach_hooks(model)

    x = torch.randn(64, 3, 32, 32, requires_grad=True)
    loss = nn.functional.cross_entropy(model(x), torch.randint(0, 10, (64,)))
    loss.backward()
    remove_hooks(handles)

    print("forward  (activation mean, std) per Linear:")
    for name, stats in fwd.items():
        print(f"  {name:12s} mean={stats[-1][0]:+.3f} std={stats[-1][1]:.3f}")
    print("backward ||dL/d output|| per Linear (fires in reverse order):")
    for name, norms in bwd.items():
        print(f"  {name:12s} norm={norms[-1]:.4f}")

    model(torch.randn(4, 3, 32, 32))
    print("calls logged after remove:", sum(len(v) for v in fwd.values()), "(unchanged)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        anomaly_demo()
