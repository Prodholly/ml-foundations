from __future__ import annotations

from typing import Callable

import numpy as np
import pytest
from grad_check.numerical import numerical_gradient


def rel_error(a: np.ndarray, n: np.ndarray) -> float:
    return float(np.max(np.abs(a - n) / np.maximum(np.abs(a) + np.abs(n), 1e-12)))


@pytest.fixture
def grad_check() -> Callable[[Callable[[np.ndarray], float], np.ndarray, np.ndarray], float]:
    def _check(f: Callable[[np.ndarray], float], x: np.ndarray, analytical: np.ndarray) -> float:
        numerical = numerical_gradient(f, x.copy())
        return rel_error(analytical, numerical)

    return _check


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(8)
