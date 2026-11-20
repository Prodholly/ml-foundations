import itertools

import numpy as np
import pytest

from numpy_mlp.data import (
    DATA_DIR, batch_generator, fit_normalizer, label_one_hot, load_mnist, normalize, split_train_val_test,
)

MNIST_ON_DISK = (DATA_DIR / "MNIST" / "raw").exists()


def test_batches_have_correct_shape(rng):
    X, Y = rng.normal(size=(1000, 784)), label_one_hot(rng.integers(0, 10, 1000))
    gen = batch_generator(X, Y, batch_size=128, seed=0)

    first_pass = list(itertools.islice(gen, 8))
    for Xb, Yb in first_pass[:-1]:
        assert Xb.shape == (128, 784) and Yb.shape == (128, 10)
    assert first_pass[-1][0].shape == (104, 784)


def test_batch_generator_is_infinite_and_covers_every_row(rng):
    X = np.arange(10, dtype=float).reshape(10, 1)
    Y = label_one_hot(np.arange(10) % 10)
    gen = batch_generator(X, Y, batch_size=4, seed=0)

    seen = np.concatenate([next(gen)[0] for _ in range(3)]).ravel()
    assert sorted(seen) == list(range(10))
    for _ in range(20):
        next(gen)


def test_no_shuffle_keeps_order():
    X = np.arange(6, dtype=float).reshape(6, 1)
    Xb, _ = next(batch_generator(X, label_one_hot(np.zeros(6, int)), batch_size=6, shuffle=False))
    assert Xb.ravel().tolist() == [0, 1, 2, 3, 4, 5]


def test_normalization_mean_zero_std_one_on_train(rng):
    X_train = rng.uniform(0, 255, size=(2000, 784))
    mean, std = fit_normalizer(X_train)
    Xn = normalize(X_train, mean, std)
    assert abs(Xn.mean()) < 1e-4
    assert abs(Xn.std() - 1.0) < 1e-4


def test_normalizer_uses_train_stats_only(rng):
    X_train, X_val = rng.uniform(0, 1, (500, 4)), rng.uniform(5, 6, (500, 4))
    mean, std = fit_normalizer(X_train)
    assert normalize(X_val, mean, std).mean() > 5


def test_one_hot_rows_sum_to_one():
    Y = label_one_hot(np.array([0, 3, 9, 3]))
    assert Y.shape == (4, 10)
    np.testing.assert_array_equal(Y.sum(axis=1), np.ones(4))
    assert Y.argmax(axis=1).tolist() == [0, 3, 9, 3]


def test_one_hot_rejects_out_of_range():
    with pytest.raises(ValueError):
        label_one_hot(np.array([0, 10]))


def test_split_80_10_10_is_disjoint_and_complete(rng):
    X = np.arange(60000, dtype=float).reshape(-1, 1)
    (Xtr, _), (Xva, _), (Xte, _) = split_train_val_test(X, np.zeros(60000, int), seed=0)
    assert (len(Xtr), len(Xva), len(Xte)) == (48000, 6000, 6000)
    ids = [set(a.ravel()) for a in (Xtr, Xva, Xte)]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert len(ids[0] | ids[1] | ids[2]) == 60000


@pytest.mark.skipif(not MNIST_ON_DISK, reason="MNIST not downloaded yet (python -m numpy_mlp.run fetches it)")
def test_real_mnist_shapes():
    X_train, y_train, X_test, y_test = load_mnist()
    assert X_train.shape == (60000, 784) and y_train.shape == (60000,)
    assert X_test.shape == (10000, 784) and y_test.shape == (10000,)
    assert X_train.min() == 0 and X_train.max() == 255
    assert set(np.unique(y_train)) == set(range(10))
