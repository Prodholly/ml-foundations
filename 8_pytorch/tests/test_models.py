import pytest

torch = pytest.importorskip("torch")

from torch import nn

from torch_nn.models.cnn import CNN
from torch_nn.models.init_nn import initialize_model
from torch_nn.models.mlp import MLP, count_parameters, expected_parameter_count


@pytest.mark.parametrize("B", [1, 7, 64])
def test_output_shape(B):
    model = MLP(use_batchnorm=False).eval()
    assert model(torch.randn(B, 3, 32, 32)).shape == (B, 10)


@pytest.mark.parametrize("B", [2, 16])
def test_cnn_output_shape(B):
    model = CNN(widths=(8, 16, 32, 64))
    assert model(torch.randn(B, 3, 32, 32)).shape == (B, 10)


@pytest.mark.parametrize("use_bn", [False, True])
def test_parameter_count_matches_formula(use_bn):
    model = MLP(hidden_sizes=(64, 32), use_batchnorm=use_bn)
    assert count_parameters(model) == expected_parameter_count(3072, (64, 32), 10, use_bn)


def test_eval_is_deterministic():
    model = MLP(dropout_p=0.5, use_batchnorm=True).eval()
    x = torch.randn(8, 3, 32, 32)
    torch.testing.assert_close(model(x), model(x))


def test_train_mode_dropout_is_random():
    torch.manual_seed(0)
    model = MLP(hidden_sizes=(64,), dropout_p=0.5).train()
    x = torch.randn(8, 3, 32, 32)
    assert not torch.equal(model(x), model(x))


def test_buffer_is_saved_but_not_a_parameter():
    model = MLP(hidden_sizes=(16,))
    assert "input_mean" in model.state_dict()
    assert "input_mean" not in dict(model.named_parameters())
    model.train()(torch.ones(4, 3, 32, 32))
    assert torch.allclose(model.input_mean, torch.full((3072,), 0.1))
    model.eval()(torch.zeros(4, 3, 32, 32))
    assert torch.allclose(model.input_mean, torch.full((3072,), 0.1))


@pytest.mark.parametrize("scheme", ["xavier_uniform", "kaiming_normal", "orthogonal"])
def test_initialize_model_zeroes_biases_and_keeps_leaves(scheme):
    model = initialize_model(MLP(hidden_sizes=(64, 32)), scheme)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            assert torch.count_nonzero(m.bias) == 0
            assert m.weight.is_leaf and m.weight.requires_grad


def test_kaiming_fan_out_variance():
    torch.manual_seed(0)
    model = initialize_model(MLP(hidden_sizes=(1024,)), "kaiming_normal")
    w = model.layers[0][0].weight
    assert abs(w.var().item() - 2 / 1024) / (2 / 1024) < 0.05
