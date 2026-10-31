# Day 1: DataLoader

A minimal, typed DataLoader built from scratch, with batching, optional shuffling with a seeded RNG, optional
drop of incomplete final batches, and a pluggable collate function.

Note: everything here is written as though I was a complete beginner.

## Why this exists

Most ML practitioners use PyTorch's DataLoader without knowing what it actually does.
This day builds it from scratch, line by line, with full type hints, tests, and documentation.
My goal is not to replace using PyTorch for this kind of task, but to understand it deeply enough that using it becomes trivial.

## Project Structure

```
1_dataloader
├── dataloader
│   ├── __init__.py          # public API surface
│   └── core.py              # DataLoader implementation
├── examples
│   └── quickstart.py        # runnable usage demo
├── research-log
│   └── 2026-10-21.md        # daily build notes
├── tests
│   └── test_dataloader.py   # pytest test suite
└── README.md
```

The full reasoning behind this build is in the [research log](research-log/2026-10-21.md).

## Installation

Run this from the repo root. It requires Python 3.10 or higher.

```bash
git clone https://github.com/Prodholly/ml-foundations.git
cd ml-foundations
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

```python
import numpy as np
from dataloader import DataLoader

# a toy dataset: 10 samples, each a 4-element vector
dataset = [np.array([i, i + 1, i + 2, i + 3]) for i in range(10)]

loader = DataLoader(dataset, batch_size=3, shuffle=True, seed=42)

for batch in loader:
    print(batch.shape)  # (3, 4)
```

Or run the included example from inside `1_dataloader`:

```bash
cd 1_dataloader
python -m examples.quickstart
```

## What it does

Wraps any sequence dataset and yields fixed-size batches one at a time.
This uses a Python generator so only one batch ever lives in memory, regardless of dataset size.
This is particularly a good engineering practice for memory efficiency. In computer vision tasks
with thousands of high-resolution images, loading the entire dataset at once would exhaust RAM
before training even begins. The generator loads only what is needed, when it is needed.

## API

```python
DataLoader(
    dataset,           # any Sequence[T] (list, tuple, custom class)
    batch_size,        # int, must be positive
    shuffle=False,     # bool, reshuffle indices at the start of each epoch
    drop_last=False,   # bool, discard final batch if smaller than batch_size
    collate_fn=None,   # Callable[[list[T]], np.ndarray], defaults to np.stack
    seed=None,         # Optional[int], seeds the RNG for reproducible shuffling
)
```

## Key behaviors

**Batch count**

With `drop_last=False`, the number of batches is `ceil(n / batch_size)`.
With `drop_last=True`, it is `n // batch_size`.

For a dataset of 10 samples and `batch_size=3`:
- `drop_last=False` gives 4 batches (last batch has 1 sample)
- `drop_last=True` gives 3 batches (last sample is discarded)

**Reproducible shuffling**

Passing `seed=42` guarantees the same index order every run, on any machine, as long as the Python version and standard library version match.
Two loaders created with the same seed will produce identical batches:

```python
loader1 = DataLoader(dataset, batch_size=4, shuffle=True, seed=42)
loader2 = DataLoader(dataset, batch_size=4, shuffle=True, seed=42)
# loader1 and loader2 yield identical batches
```

**Custom collation**

By default, batches are assembled with `np.stack`. Pass your own function to override:

```python
def pad_and_stack(samples):
    max_len = max(len(s) for s in samples)
    padded = [np.pad(s, (0, max_len - len(s))) for s in samples]
    return np.stack(padded)

loader = DataLoader(variable_length_dataset, batch_size=8, collate_fn=pad_and_stack)
```

## Design decisions

The main decisions are below. The [research log](research-log/2026-10-21.md) has the full reasoning.

| Decision | Why |
|---|---|
| Use `Generic[T]` instead of hardcoding `np.ndarray` | The DataLoader works on any sequence type, and the type system catches mismatches in the editor |
| Use `random.Random(seed)` instead of `random.seed(seed)` | It creates an isolated RNG, so global random state doesn't bleed into other code and two loaders never interfere |
| Use `yield` instead of returning a list of batches | A generator loads one batch at a time, so a large dataset doesn't exhaust RAM before training starts |
| Shuffle indices instead of the dataset | The user's original data never moves, because the shuffled integers just point into it |

## Running the Tests

From the repo root:

```bash
pytest 1_dataloader/tests
```

Expected output:

```
6 passed in 0.xxs
```

Plain `pytest` from the repo root runs the tests for every day.

## Concepts Covered

| Concept | Location |
|---|---|
| Python generics (`Generic[T]`, `TypeVar`) | `core.py` class definition |
| Generator functions (`yield`) | `DataLoader.__iter__` |
| Dunder methods (`__init__`, `__len__`, `__iter__`) | `core.py` throughout |
| Seeded RNG for reproducibility | `random.Random(seed)` in `__init__` |
| Callable type hints | `collate_fn` parameter signature |
| Editable package installs | root `pyproject.toml` and `pip install -e ".[dev]"` |
| Defensive validation | `ValueError` on bad `batch_size` or empty dataset |
| pytest basics | `tests/test_dataloader.py` |

## Roadmap

This is Day 1 of a 115-day ML engineering curriculum.
Upcoming builds include a training loop, autograd engine, attention mechanism, and transformer from scratch.

## Author

Emmanuel Adeloju — [github.com/Prodholly](https://github.com/Prodholly)

## License

MIT
