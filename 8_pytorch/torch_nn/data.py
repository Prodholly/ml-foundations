from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms

from .training.reproducibility_nn import seed_worker

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)
CLASSES = ("airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def build_transforms(train: bool) -> transforms.Compose:
    to_tensor = [transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)]
    if not train:
        return transforms.Compose(to_tensor)
    augment = [
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
    ]
    return transforms.Compose(augment + to_tensor)


def load_cifar10(root: Path = DATA_DIR) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torchvision

    train = torchvision.datasets.CIFAR10(str(root), train=True, download=True)
    test = torchvision.datasets.CIFAR10(str(root), train=False, download=True)

    X_train, y_train = train.data, np.array(train.targets, dtype=np.int64)
    X_test, y_test = test.data, np.array(test.targets, dtype=np.int64)

    return X_train, y_train, X_test, y_test


class CIFARDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None, name: str = "") -> None:
        if len(images) != len(labels):
            raise ValueError(f"{len(images)} images but {len(labels)} labels")
        self.images = images
        self.labels = labels
        self.transform = transform
        self.name = name

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img = Image.fromarray(self.images[idx])

        if self.transform is not None:
            img = self.transform(img)

        return img, int(self.labels[idx])

    def __repr__(self) -> str:
        return (f"CIFARDataset(n={len(self)}, classes={len(np.unique(self.labels))}, "
                f"transform={self.name or ('yes' if self.transform else 'none')})")


def split_indices(n: int, val_fraction: float = 0.1, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(n * val_fraction)
    return perm[n_val:], perm[:n_val]


def make_weighted_sampler(labels: np.ndarray, seed: int = 0) -> WeightedRandomSampler:
    counts = np.bincount(labels)

    sample_weights = 1.0 / counts[labels]

    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(labels), replacement=True,
        generator=torch.Generator().manual_seed(seed),
    )


def load_cifar10_tensors(device: str = "cpu", seed: int = 0) -> dict[str, torch.Tensor]:
    X_tr, y_tr, X_te, y_te = load_cifar10()
    train_idx, val_idx = split_indices(len(X_tr), 0.1, seed)
    mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)

    def prep(images: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        return ((x - mean) / std).to(device)

    return {
        "X_train": prep(X_tr[train_idx]), "y_train": torch.from_numpy(y_tr[train_idx]).to(device),
        "X_val": prep(X_tr[val_idx]), "y_val": torch.from_numpy(y_tr[val_idx]).to(device),
        "X_test": prep(X_te), "y_test": torch.from_numpy(y_te).to(device),
    }


def tensor_batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, generator: torch.Generator):
    perm = torch.randperm(len(X), generator=generator).to(X.device)
    for s in range(0, len(X), batch_size):
        idx = perm[s:s + batch_size]
        yield X[idx], y[idx]


def build_loaders(batch_size: int = 128, seed: int = 0, device: str = "cpu", num_workers: int = 4,
                  balanced: bool = False) -> tuple[DataLoader, DataLoader, DataLoader]:
    X_tr, y_tr, X_te, y_te = load_cifar10()

    train_idx, val_idx = split_indices(len(X_tr), 0.1, seed)

    train_ds = CIFARDataset(X_tr[train_idx], y_tr[train_idx], build_transforms(train=True), "train")
    val_ds = CIFARDataset(X_tr[val_idx], y_tr[val_idx], build_transforms(train=False), "eval")
    test_ds = CIFARDataset(X_te, y_te, build_transforms(train=False), "eval")

    kw = dict(num_workers=num_workers, persistent_workers=num_workers > 0, pin_memory=(device == "cuda"),
              worker_init_fn=seed_worker)

    if balanced:
        train_loader = DataLoader(train_ds, batch_size, sampler=make_weighted_sampler(y_tr[train_idx], seed), **kw)
    else:
        train_loader = DataLoader(train_ds, batch_size, shuffle=True,
                                  generator=torch.Generator().manual_seed(seed), **kw)
    val_loader = DataLoader(val_ds, 512, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, 512, shuffle=False, **kw)

    return train_loader, val_loader, test_loader
