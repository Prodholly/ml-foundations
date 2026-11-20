import pytest

torch = pytest.importorskip("torch")

from torch import nn

from torch_nn.training.regularization_nn import (
    clip_gradients, l1_penalty, l2_penalty, total_grad_norm,
)


def test_dropout_inactive_in_eval():
    drop = nn.Dropout(0.5).eval()
    x = torch.randn(100, 100)
    assert torch.equal(drop(x), x)


def test_dropout_active_in_train():
    torch.manual_seed(0)
    y = nn.Dropout(0.5).train()(torch.ones(100, 100))
    assert 0.45 < (y == 0).float().mean().item() < 0.55
    assert set(y.unique().tolist()) == {0.0, 2.0}


def test_batchnorm_running_stats_update_then_freeze():
    bn = nn.BatchNorm1d(4)
    before = bn.running_mean.clone()
    bn.train()
    bn(torch.randn(32, 4) + 5)
    assert not torch.equal(bn.running_mean, before)
    assert bn.num_batches_tracked.item() == 1

    frozen = bn.running_mean.clone()
    bn.eval()
    bn(torch.randn(32, 4))
    assert torch.equal(bn.running_mean, frozen)
    assert bn.num_batches_tracked.item() == 1


def test_grad_clip_bounds_the_norm():
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.Linear(20, 1))
    loss = model(torch.randn(16, 10)).pow(2).sum() * 1000
    loss.backward()
    before = clip_gradients(model, max_norm=1.0)
    assert before > 1.0
    assert total_grad_norm(model) <= 1.0 + 1e-6
    assert max(p.grad.norm().item() for p in model.parameters()) <= 1.0 + 1e-6


def test_penalties_skip_biases():
    model = nn.Sequential(nn.Linear(3, 2))
    with torch.no_grad():
        model[0].weight.fill_(2.0)
        model[0].bias.fill_(100.0)
    assert l2_penalty(model, 0.5).item() == pytest.approx(0.5 * 6 * 4)
    assert l1_penalty(model, 0.5).item() == pytest.approx(0.5 * 6 * 2)


def test_l2_penalty_grad_is_2_lambda_w():
    model = nn.Sequential(nn.Linear(3, 2))
    l2_penalty(model, 0.1).backward()
    torch.testing.assert_close(model[0].weight.grad, 2 * 0.1 * model[0].weight.detach())
