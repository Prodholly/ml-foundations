# Day 2: Linear Algebra for ML

Matrix multiplication, softmax and cross-entropy, built from scratch with NumPy arrays and plain Python loops,
then rebuilt with `np.einsum`, then timed against NumPy itself.

Note: like Day 1, everything here is written as though I was a complete beginner.

## Why this exists

Almost every line of a neural network is one of three things: multiply some matrices, turn scores into
probabilities, and measure how wrong those probabilities are. In PyTorch that is `x @ W.T`, `softmax`, and
`cross_entropy`, which are three function calls you never look inside.

This day opens them up. I wrote the matrix multiply as three nested loops so every multiply-add is visible,
then rewrote it as a single `einsum` line, then benchmarked both against `np.matmul` to see what
"use the library" is actually buying us.

## Project Structure

```
2_linalg
├── examples
│   └── demo.py              # runnable walkthrough of everything below
├── linalg
│   ├── __init__.py
│   ├── ops.py               # matmul with three nested loops
│   ├── einsum_matmul.py     # the same matmul in one einsum line
│   ├── softmax.py           # softmax with the max-subtraction trick
│   ├── einsum_softmax.py    # row-wise softmax using einsum
│   ├── cross_entropy.py     # cross-entropy loss on one-hot targets
│   └── benchmark.py         # loop vs einsum vs NumPy timing
├── linalg_test
│   └── test_linalg.py       # pytest test suite
└── README.md
```

The day's build notes are in [`research-log/2026-10-25.md`](../research-log/2026-10-25.md) at the repo root.

## Installation

Same setup as Day 1. The only dependency is NumPy.

```bash
git clone https://github.com/Prodholly/ml-foundations.git
cd ml-foundations
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`linalg` is not installed as a package. The root `pyproject.toml` adds `2_linalg` to pytest's import
path, which is all the tests need.

## Quick Start

```python
import numpy as np
from linalg.ops import matmul
from linalg.einsum_softmax import softmax_einsum
from linalg.cross_entropy import cross_entropy

X = np.random.rand(3, 4)        # 3 samples, 4 features each
w = np.random.rand(3, 4)        # 3 classes, one weight vector per class

z = matmul(X, w.T)              # logits, shape (3, 3)
probs = softmax_einsum(z)       # each row now sums to 1
loss = cross_entropy(np.eye(3), probs)
```

Or run the included demo from inside `2_linalg`:

```bash
cd 2_linalg
python -m examples.demo
```

The `-m` matters. It runs the demo as a module with `2_linalg` as the starting folder, so
`from linalg...` imports resolve. Running `python examples/demo.py` would look for `linalg` inside
`examples/` and fail.

Every file in `linalg/` can also be run on its own to see its demo, for example
`python -m linalg.benchmark`. The demo code sits under `if __name__ == "__main__":`, so importing the
functions (as the tests do) does not run it.

## What each piece does

### Matrix multiply (`ops.py`, `einsum_matmul.py`)

Multiplying an `(m, n)` matrix by an `(n, p)` matrix gives an `(m, p)` matrix. Each output cell `C[i, j]`
is the dot product of row `i` of the first matrix and column `j` of the second:

```python
for i in range(rows_X):          # each row of the output
    for j in range(cols_Y):      # each column of the output
        for k in range(inner):   # walk along the shared dimension
            C[i, j] += X[i, k] * Y[k, j]
```

The einsum version does the same thing in one line:

```python
np.einsum('ik,kj->ij', X, Y)
```

Read it as: `X` has axes `i, k`, `Y` has axes `k, j`, and the output keeps `i, j`. Because `k` appears
in the inputs but not in the output, einsum sums over it. That is the innermost loop above. The letters
before the arrow are the loop variables, and the letters missing after the arrow are the ones being summed.

### Softmax (`softmax.py`, `einsum_softmax.py`)

Turns a row of raw scores (logits) into probabilities that are all positive and sum to 1:

```
softmax(z)_j = exp(z_j) / sum_k exp(z_k)
```

Before exponentiating, the max is subtracted. `exp(1000)` overflows to `inf`, but softmax gives the same
answer if you shift every score by the same amount, so subtracting the max keeps the largest exponent
at `exp(0) = 1`.

`softmax_einsum` works on a `(samples, classes)` matrix and normalizes each row separately, so you get
one probability distribution per sample.

### Cross-entropy (`cross_entropy.py`)

Measures how much probability the model gave to the correct class:

```python
-np.mean(np.sum(y_true * np.log(y_pred + 1e-9), axis=1))
```

With one-hot targets, `y_true * log(y_pred)` zeros out every class except the correct one, so each
sample's loss is just `-log(probability of the right answer)`. The `1e-9` stops `log(0)` from
returning `-inf`.

### Benchmark (`benchmark.py`)

Times a 500x500 matrix multiply three ways. Results on my MacBook Air:

| Method | Time | Relative to NumPy |
|---|---|---|
| Three-loop Python `matmul` | 39.94 s | ~14,659x slower |
| `np.einsum('ik,kj->ij', ...)` | 0.0204 s | ~7.5x slower |
| `np.matmul` | 0.0027 s | 1x |

Your numbers will differ by machine, but the order of magnitude will not. See the research log for why
the gap is this large and why einsum lands in the middle.

## Running the Tests

From the repo root:

```bash
pytest 2_linalg/linalg_test
```

Expected output:

```
6 passed in 0.xxs
```

Plain `pytest` from the repo root runs both Day 1 and Day 2 tests (12 total).

## Known issues

These are documented, not fixed yet. See the research log for detail.

- `softmax_einsum` subtracts the max of the whole matrix instead of each row's max. Rows with very
  different scales can produce `nan`.
- `softmax` in `softmax.py` normalizes over the whole array, so on a 2D input it is not a per-sample
  softmax. The standalone demo in `cross_entropy.py` uses it, which inflates that demo's loss.
- `matmul` does not check that the inner dimensions match. `(3, 4) @ (2, 3)` silently returns a wrong
  `(3, 3)` result instead of raising an error.

## Concepts Covered

| Concept | Location |
|---|---|
| Matrix shapes and the inner-dimension rule | `ops.py` |
| Einstein summation notation | `einsum_matmul.py`, `einsum_softmax.py` |
| Numerical stability (max subtraction, epsilon in log) | `softmax.py`, `cross_entropy.py` |
| Logits, probabilities and one-hot targets | `einsum_softmax.py`, `cross_entropy.py` |
| Weights stored as `(classes, features)` and the `w.T` transpose | `einsum_softmax.py` |
| Interpreted loops vs compiled BLAS | `benchmark.py` |
| `if __name__ == "__main__":` guards | every file in `linalg/` |
| pytest `pythonpath` and `testpaths` config | root `pyproject.toml` |

## Roadmap

This is Day 2 of a 115-day ML engineering curriculum. The matmul, softmax and cross-entropy built here are
the forward pass and loss of a classifier, which the training loop builds on next.

## Author

Emmanuel Adeloju — [github.com/Prodholly](https://github.com/Prodholly)

## License

MIT
