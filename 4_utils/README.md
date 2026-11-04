# Day 4: Python Data Structures & Complexity

Three typed utility functions for everyday ML engineering jobs: splitting data into batches, checking an
array's shape, and turning messy values into integers. Each one has an explicit input/output contract,
handles its edge cases, comes with a complexity analysis, and is covered by pytest tests.


## Why this exists

ML code rarely fails on the happy path. It fails on an empty list, a batch with its axes swapped, or a
`"N/A"` in a column that was supposed to be numbers. Production code deals with this by stating exactly
what each function accepts, what it returns, and what happens at the edges, and then testing those edges.

This day practises that habit on three small functions, plus the other half of writing them well: knowing
what they cost in time and memory as the input grows.

## Project Structure

```
4_utils
├── examples
│   └── demo.py              # the three utilities on ML-style jobs
├── research-log
│   └── 2026-11-04.md        # daily build notes
├── tests
│   └── test_ml_utils.py     # pytest tests for every function
├── utils
│   ├── __init__.py          # exports the three functions
│   └── ml_utils.py          # all utility functions live here
└── README.md
```

The full reasoning behind this build is in the [research log](research-log/2026-11-04.md).

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
from utils import batch_items, check_shape, safe_parse_int

batch_items([1, 2, 3, 4, 5], 2)          # [[1, 2], [3, 4], [5]]
check_shape(np.zeros((3, 4)), (3, 4))    # True
safe_parse_int("abc", default=-1)        # -1
```

Or run the included demo from inside `4_utils`:

```bash
cd 4_utils
python -m examples.demo
```

## What each function does

### `batch_items(items, batch_size)`

Splits a flat list into batches of a given size. The last batch may be smaller.

| | |
|---|---|
| Input | `items: list[Any]`, `batch_size: int` |
| Output | `list[list[Any]]` |
| Empty list | returns `[]` |
| `batch_size <= 0` | raises `ValueError` |
| `batch_size > len(items)` | one batch with all the items |

```python
batch_items([1, 2, 3, 4, 5], 2)   # [[1, 2], [3, 4], [5]]
batch_items([], 3)                # []
batch_items([1, 2, 3], 10)        # [[1, 2, 3]]
```

### `check_shape(tensor, expected_shape)`

Checks that a NumPy array has the expected shape. It returns `True`, or raises an error that shows the
actual shape next to the expected one.

| | |
|---|---|
| Input | `tensor: np.ndarray`, `expected_shape: tuple[int, ...]` |
| Output | `bool`, and only ever `True` |
| `None` | raises `TypeError` |
| Not a NumPy array | raises `TypeError` |
| Wrong number of dimensions, or wrong sizes | raises `ValueError` |

```python
check_shape(np.zeros((3, 4)), (3, 4))   # True
check_shape(np.zeros((3, 4)), (4, 3))   # ValueError: Shape mismatch: expected (4, 3), but got (3, 4)
check_shape(None, (3, 4))               # TypeError: tensor cannot be None
```

### `safe_parse_int(value, default=None)`

Converts a value to an integer without crashing. If it can't, it returns `default`.

| | |
|---|---|
| Input | `value: Any`, `default: Optional[int] = None` |
| Output | `Optional[int]` |
| `None` | returns `default` |
| Float | truncates toward zero, so `3.9` becomes `3` and `-3.9` becomes `-3` |
| `"abc"`, empty string, `inf`, `nan` | returns `default` |

```python
safe_parse_int("42")         # 42
safe_parse_int(3.9)          # 3
safe_parse_int("abc")        # None
safe_parse_int("abc", -1)    # -1
safe_parse_int(None)         # None
```

## Complexity

| Function | Time | Space | Notes |
|---|---|---|---|
| `batch_items` | O(n) | O(n) | n = number of items. Every item is copied into a batch once |
| `check_shape` | O(1) | O(1) | the shape is stored metadata, so the array's data is never read |
| `safe_parse_int` | O(1) | O(1) | one conversion of one value |

The same table sits at the top of `ml_utils.py`. The research log explains each row, including where
"O(1)" is really "constant for any realistic input".

## Design decisions

The main decisions are below. The [research log](research-log/2026-11-04.md) has the full reasoning.

| Decision | Why |
|---|---|
| Check `batch_size` before checking for an empty list | A bad `batch_size` is a bug in the caller, so `batch_items([], 0)` raises rather than quietly returning `[]` |
| Return a list of batches, not a generator | That's the contract, and it suits small lists. Day 1's DataLoader uses a generator because it has to handle datasets too big for memory |
| Keep the smaller last batch | Nothing is thrown away. Dropping it is the caller's choice, like Day 1's `drop_last` |
| `check_shape` returns `True` or raises, never `False` | A `False` is easy to ignore by accident, and an error stops the program at the line where the shape went wrong |
| Show expected and actual shapes in the error | `expected (4, 3), but got (3, 4)` already tells you the batch is transposed |
| Use `raise`, not `assert` | Python skips `assert` statements when run with `-O`, so a check written as an `assert` can silently disappear |
| `TypeError` for the wrong kind of input, `ValueError` for the wrong value | `None` or a list isn't an array at all, but a `(3, 4)` array is the right kind of thing with the wrong shape |
| `safe_parse_int` defaults to `None` | `None` means "couldn't parse", which stays distinct from a real `0`. The caller can pass `-1` or `0` when they want a number |
| Truncate floats with `int()` | It's what the contract asks for, and it means `-3.9` becomes `-3`, not `-4` as with rounding down |
| Catch specific exceptions, never a bare `except:` | A bare `except:` would also swallow things like Ctrl+C, and would hide real bugs |
| Add runtime checks even with type hints | Python doesn't enforce type hints. They help readers and editors, but only the checks stop bad input |
| Test error messages with `match=` | It proves the error came from my check, not from somewhere else that raises the same type |

## Challenges and fixes

| Challenge | Fix |
|---|---|
| Two tests were called `test_check_shape_wrong`, so the second silently replaced the first | Gave each test its own name. The first one asserted nothing anyway |
| There was no test for the `None` → `TypeError` case | Added `test_check_shape_none` |
| `safe_parse_int(float("inf"))` crashed with `OverflowError` | Added `OverflowError` to the exceptions it catches |
| A plain list passed to `check_shape` crashed with a confusing `AttributeError` | Added an `isinstance` check that raises a clear `TypeError` |
| The `batch_size=0` test passed even with my check removed, because `range()` raises its own `ValueError` for a step of 0 | Checked the message with `match="batch_size must be positive"` |
| The new `isinstance` check also catches `None`, so the `None` test passed without the `None` check | Checked the message there too |

## Running the Tests

From the repo root:

```bash
pytest 4_utils/tests
```

Or from inside `4_utils`, with each test listed:

```bash
cd 4_utils
pytest tests/ -v
```

Expected output:

```
20 passed in 0.xxs
```

Plain `pytest` from the repo root runs the tests for every day.

## Known limitations

- `safe_parse_int("3.9")` returns the default. Only real floats are truncated, because `int()` won't
  parse a string with a decimal point.
- `int()` is more forgiving than it looks: `" 42 "`, `"4_2"`, `True` and `b"7"` all become integers.
- A number string longer than 4,300 digits returns the default. That's Python's safety limit on
  converting long strings to integers.
- `check_shape` needs a tuple. Passing `[3, 4]` fails with `expected [3, 4], but got (3, 4)`, which
  looks like a match but isn't, because a list never equals a tuple.
- `check_shape` has no wildcard. There's no way to say "any batch size, but 3 features".
- `batch_items` builds every batch up front, so it holds a second copy of the list in memory.
- `utils` is a common package name, so it could clash with another installed package called `utils`.

## Concepts Covered

| Concept | Location |
|---|---|
| Type hints: `list[Any]`, `tuple[int, ...]`, `Optional` | `ml_utils.py` signatures |
| Big-O time and space complexity | complexity table in `ml_utils.py` |
| List slicing and stepping through a range | `batch_items` |
| `raise` vs `assert`, and `TypeError` vs `ValueError` | `check_shape` |
| Specific exception handling with `try`/`except` | `safe_parse_int` |
| How `int()` treats floats, strings and other types | `safe_parse_int` |
| Exporting functions from a package's `__init__.py` | `utils/__init__.py` |
| `pytest.mark.parametrize` and `pytest.raises(match=...)` | `tests/test_ml_utils.py` |
| Checking that tests can actually fail | research log |

## Roadmap

This is Day 4 of a 115-day ML engineering curriculum.
Upcoming builds include a training loop, autograd engine, attention mechanism, and transformer from scratch.

## Author

Emmanuel Adeloju — [github.com/Prodholly](https://github.com/Prodholly)

## License

MIT
