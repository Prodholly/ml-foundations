from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch.utils.data import DataLoader

from torch_nn.data import CIFARDataset, build_transforms, make_weighted_sampler, split_indices

CIFAR_DIR = Path(__file__).resolve().parents[1] / "data" / "cifar-10-batches-py"


def fake_cifar(n: int = 64, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    images = rng.integers(0, 256, (n, 32, 32, 3), dtype=np.uint8)
    labels = np.arange(n, dtype=np.int64) % 10
    return images, labels


def test_extreme_pixels_normalized_within_minus3_plus3():
    images = np.stack([np.zeros((32, 32, 3), np.uint8), np.full((32, 32, 3), 255, np.uint8)])
    ds = CIFARDataset(images, np.array([0, 1]), build_transforms(train=False))
    for i in range(2):
        x, _ = ds[i]
        assert x.shape == (3, 32, 32)
        assert x.min() >= -3 and x.max() <= 3


@pytest.mark.skipif(not CIFAR_DIR.exists(), reason="CIFAR-10 not downloaded")
def test_real_images_normalized_within_minus3_plus3():
    from torch_nn.data import load_cifar10

    X_tr, y_tr, _, _ = load_cifar10()
    ds = CIFARDataset(X_tr[:256], y_tr[:256], build_transforms(train=False))
    x = torch.stack([ds[i][0] for i in range(len(ds))])
    assert x.min() >= -3 and x.max() <= 3
    assert x.mean(dim=(0, 2, 3)).abs().max() < 0.3


def test_loader_batch_shapes():
    images, labels = fake_cifar(64)
    loader = DataLoader(CIFARDataset(images, labels, build_transforms(train=True)), batch_size=16, num_workers=0)
    X, y = next(iter(loader))
    assert X.shape == (16, 3, 32, 32) and X.dtype == torch.float32
    assert y.shape == (16,) and y.dtype == torch.int64


def test_dataset_len_and_repr():
    images, labels = fake_cifar(20)
    ds = CIFARDataset(images, labels, build_transforms(False), "eval")
    assert len(ds) == 20
    assert repr(ds) == "CIFARDataset(n=20, classes=10, transform=eval)"


def test_split_indices_disjoint():
    tr, va = split_indices(50000, 0.1, seed=0)
    assert set(tr.tolist()).isdisjoint(va.tolist())
    assert len(tr) + len(va) == 50000 and len(va) == 5000
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(50000))


def test_weighted_sampler_balances_an_imbalanced_set():
    labels = np.array([0] * 900 + [1] * 100)
    drawn = np.array(list(make_weighted_sampler(labels, seed=0)))
    frac_class1 = (labels[drawn] == 1).mean()
    assert 0.45 < frac_class1 < 0.55
