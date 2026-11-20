from __future__ import annotations

from typing import Callable

import numpy as np

_STD: dict[str, Callable[[int, int], float]] = {
    "gaussian": lambda fan_in, fan_out: 0.01,
    "xavier": lambda fan_in, fan_out: np.sqrt(2.0 / (fan_in + fan_out)),
    "he": lambda fan_in, fan_out: np.sqrt(2.0 / fan_in),
    "lecun": lambda fan_in, fan_out: np.sqrt(1.0 / fan_in),
}

INIT_SCHEMES: tuple[str, ...] = tuple(_STD)


def init_std(scheme: str, fan_in: int, fan_out: int) -> float:
    try:
        return float(_STD[scheme.lower()](fan_in, fan_out))
    except KeyError:
        raise ValueError(f"unknown init scheme {scheme!r}, choose from {list(INIT_SCHEMES)}") from None


def init_weights(
    scheme: str,
    fan_in: int,
    fan_out: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if fan_in < 1 or fan_out < 1:
        raise ValueError(f"fan_in and fan_out must be >= 1, got {fan_in}, {fan_out}")
    rng = np.random.default_rng() if rng is None else rng
    std = init_std(scheme, fan_in, fan_out)
    return rng.normal(loc=0.0, scale=std, size=(fan_out, fan_in))
