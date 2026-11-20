import numpy as np
import torch

from numpy_mlp.layers import LinearLayer

from .layers import RawLinear

TOL = 1e-5


def manual_backward(
    X: torch.Tensor, W: torch.Tensor, dL_dZ: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        dW = dL_dZ.T @ X

        db = dL_dZ.sum(dim=0)

        dX = dL_dZ @ W

    return dW, db, dX


def verify_layer(N: int = 8, in_features: int = 5, out_features: int = 3, seed: int = 0) -> dict[str, float]:
    torch.manual_seed(seed)
    layer = RawLinear(in_features, out_features, device="cpu", dtype=torch.float64)
    
    with torch.no_grad():
        layer.b.normal_()
    
    X = torch.randn(N, in_features, dtype=torch.float64, requires_grad=True)
    Z = layer(X)
    upstream = torch.randn(N, out_features, dtype=torch.float64)
    loss = (upstream * Z).sum()
    loss.backward()
    dW, db, dX = manual_backward(X, layer.W, upstream)
    
    err_W = (layer.W.grad - dW).abs().max().item()
    err_b = (layer.b.grad - db).abs().max().item()
    err_X = (X.grad - dX).abs().max().item()
    
    return {
        "W": err_W,
        "b": err_b,
        "X": err_X
    }


def verify_against_numpy_layer(N: int = 8, in_features: int = 5, out_features: int = 3, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    np_layer = LinearLayer(in_features, out_features, init_scheme="he", rng=rng)
    layer = RawLinear(in_features, out_features, "cpu", dtype=torch.float64)
    layer.W = torch.from_numpy(np_layer.W.copy()).requires_grad_()
    layer.b = torch.from_numpy(np_layer.b.copy()).requires_grad_()
    X_np = rng.normal(size=(N, in_features)); 
    up_np = rng.normal(size=(N, out_features))
    np_layer.forward(X_np)
    dX_np = np_layer.backward(up_np)
    X = torch.from_numpy(X_np).requires_grad_()
    loss = (torch.from_numpy(up_np) * layer(X)).sum()
    loss.backward()

    err_W = np.abs(layer.W.grad.numpy() - np_layer.dW).max()
    err_b = np.abs(layer.b.grad.numpy() - np_layer.db).max()
    err_X = np.abs(X.grad.numpy() - dX_np).max()

    return {
        "W": float(err_W),
        "b": float(err_b),
        "X": float(err_X)
    }


if __name__ == "__main__":
    checks = [("manual formulas", verify_layer), ("phase 1 LinearLayer", verify_against_numpy_layer)]

    for name, check in checks:
        errs = check()
        print(f"{name}:")
        for param, err in errs.items():
            print(f"    d{param}  max abs error {err:.2e}")
        assert all(e < TOL for e in errs.values()), f"{name} disagrees with autograd: {errs}"

    print(f"phase 1's math matches autograd (every error < {TOL})")
