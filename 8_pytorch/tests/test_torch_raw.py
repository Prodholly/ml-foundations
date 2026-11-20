import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F
from torch import nn

from torch_raw.layers import RawLinear
from torch_raw.model import RawMLP, cross_entropy
from torch_raw.optimizers import Adam, AdamW, SGDMomentum
from torch_raw.regularization import RawBatchNorm1d, dropout
from torch_raw.verify_autograd import TOL, verify_against_numpy_layer, verify_layer


def test_raw_linear_shapes_and_leaves():
    layer = RawLinear(5, 3, "cpu")
    assert layer.W.shape == (3, 5) and layer.b.shape == (3,)
    assert layer.W.is_leaf and layer.W.requires_grad and layer.b.is_leaf
    assert layer(torch.randn(4, 5)).shape == (4, 3)
    with pytest.raises(ValueError):
        layer(torch.randn(4, 6))


@pytest.mark.parametrize("check", [verify_layer, verify_against_numpy_layer])
def test_autograd_matches_phase1_math(check):
    assert all(e < TOL for e in check().values())


def test_raw_mlp_forward_and_loss():
    model = RawMLP([20, 16, 3], activation="swish")
    logits = model(torch.randn(8, 20))
    assert logits.shape == (8, 3)
    y = torch.randint(0, 3, (8,))
    torch.testing.assert_close(cross_entropy(logits, y), F.cross_entropy(logits, y))
    assert len(model.parameters()) == 4


def _run(opt_raw, opt_ref, steps=25):
    torch.manual_seed(0)
    p1 = torch.randn(6, 4, requires_grad=True)
    p2 = p1.detach().clone().requires_grad_()
    o1, o2 = opt_raw([p1]), opt_ref([p2])
    for _ in range(steps):
        for p, o in ((p1, o1), (p2, o2)):
            o.zero_grad()
            (p ** 2).sum().backward()
            o.step()
    return p1, p2


@pytest.mark.parametrize("raw, ref", [
    (lambda ps: SGDMomentum(ps, lr=0.01, beta=0.9), lambda ps: torch.optim.SGD(ps, lr=0.01, momentum=0.9)),
    (lambda ps: Adam(ps, lr=0.01), lambda ps: torch.optim.Adam(ps, lr=0.01)),
    (lambda ps: AdamW(ps, lr=0.01, weight_decay=0.1), lambda ps: torch.optim.AdamW(ps, lr=0.01, weight_decay=0.1)),
])
def test_raw_optimizers_match_torch_optim(raw, ref):
    p1, p2 = _run(raw, ref)
    torch.testing.assert_close(p1, p2, atol=1e-6, rtol=1e-5)


def test_raw_dropout():
    torch.manual_seed(0)
    x = torch.ones(200, 200)
    y = dropout(x, 0.3, training=True)
    assert 0.28 < (y == 0).float().mean().item() < 0.32
    torch.testing.assert_close(y.max(), torch.tensor(1 / 0.7))
    assert torch.equal(dropout(x, 0.3, training=False), x)


def test_raw_batchnorm_matches_nn():
    torch.manual_seed(0)
    mine, ref = RawBatchNorm1d(4), nn.BatchNorm1d(4)
    for _ in range(3):
        x = torch.randn(32, 4) * 3 + 5
        torch.testing.assert_close(mine(x, training=True), ref(x), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(mine.running_mean, ref.running_mean)
    torch.testing.assert_close(mine.running_var, ref.running_var)
    ref.eval()
    x = torch.randn(8, 4)
    torch.testing.assert_close(mine(x, training=False), ref(x), atol=1e-5, rtol=1e-5)


def test_clipped_relu_gradcheck():
    from torch_nn.advanced.custom_autograd import verify_gradcheck

    assert verify_gradcheck()


def test_custom_losses_match_builtins():
    from torch_nn.advanced.custom_losses import FocalLoss, LabelSmoothingCE

    torch.manual_seed(0)
    logits, y = torch.randn(16, 10), torch.randint(0, 10, (16,))
    torch.testing.assert_close(LabelSmoothingCE(0.1)(logits, y), F.cross_entropy(logits, y, label_smoothing=0.1))
    torch.testing.assert_close(FocalLoss(0.0)(logits, y), F.cross_entropy(logits, y))
    assert FocalLoss(2.0)(logits, y) < F.cross_entropy(logits, y)


def test_warmup_cosine_shape():
    from torch_nn.training.schedulers_nn import warmup_cosine

    assert warmup_cosine(0, 10, 100) == 0.0
    assert warmup_cosine(5, 10, 100) == 0.5
    assert warmup_cosine(10, 10, 100) == 1.0
    assert warmup_cosine(100, 10, 100) == pytest.approx(0.0)


def test_checkpoint_resume_is_exact():
    from torch_nn.training.checkpointing_nn import verify_resume

    assert verify_resume(epochs_total=4, split_at=2)


def test_same_seed_same_curve():
    from torch_nn.training.reproducibility_nn import tiny_run

    assert tiny_run(0, 2) == tiny_run(0, 2)
    assert tiny_run(0, 2) != tiny_run(1, 2)


def test_module_grad_check_with_day6_checker():
    from torch_nn.debugging.grad_check import gradcheck_module, numerical_check_module, tiny_model

    torch.manual_seed(0)
    X, y = torch.randn(4, 6), torch.randint(0, 3, (4,))
    errs = numerical_check_module(tiny_model(), X, y, nn.CrossEntropyLoss())
    assert all(e < 1e-5 for e in errs.values())
    assert gradcheck_module(tiny_model(), X, y, nn.CrossEntropyLoss())
