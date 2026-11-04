from typing import Any, Optional

import numpy as np

# Function          | Time   | Space | Notes
# batch_items       | O(n)   | O(n)  | n = number of items
# check_shape       | O(1)   | O(1)  | shape is stored metadata
# safe_parse_int    | O(1)   | O(1)  | single value conversion



def batch_items(items: list[Any], batch_size: int) -> list[list[Any]]:

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if len(items) == 0:
        return []

    batches = []

    for start in range(0, len(items), batch_size):
        batch = items[start: start + batch_size]
        batches.append(batch)

    return batches


def check_shape(tensor: np.ndarray, expected_shape: tuple[int, ...]) -> bool:

    if tensor is None:
        raise TypeError("tensor cannot be None")

    if not isinstance(tensor, np.ndarray):
        raise TypeError(f"tensor must be a NumPy array, got {type(tensor).__name__}")

    if tensor.shape != expected_shape:
        raise ValueError(f"Shape mismatch: expected {expected_shape}, but got {tensor.shape}")

    return True


def safe_parse_int(value: Any, default: Optional[int] = None) -> Optional[int]:

    if value is None:
        return default

    try:
        return int(value)
    except (ValueError, TypeError, OverflowError):
        return default