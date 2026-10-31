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
├── research-log
│   └── 2026-10-25.md        # daily build notes
├── tests
│   └── test_linalg.py       # pytest test suite
└── README.md
```

The full reasoning behind this build is in the [research log](research-log/2026-10-25.md).

## Installation

Run this from the repo root. It requires Python 3.10 or higher.

```bash
git clone https://github.com/Prodholly/ml-foundations.git
cd ml-foundations
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
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

`matmul` checks the inner dimensions first and raises a `ValueError` if they don't match, the same way
`np.matmul` does. Without that check, `(3, 4) @ (2, 3)` would quietly use only part of `X` and return
wrong numbers.

### Softmax (`softmax.py`, `einsum_softmax.py`)

Turns a row of raw scores (logits) into probabilities that are all positive and sum to 1:

```
softmax(z)_j = exp(z_j) / sum_k exp(z_k)
```

Before exponentiating, the max is subtracted. `exp(1000)` overflows to `inf`, but softmax gives the same
answer if you shift every score by the same amount, so subtracting the max keeps the largest exponent
at `exp(0) = 1`.

Both versions work on a `(samples, classes)` matrix and treat each row as its own sample. The max and the
sum are taken per row (`axis=1, keepdims=True`), so you get one probability distribution per sample, and
a row on a very different scale from the others can't underflow to `0 / 0`. `softmax` also accepts a
single sample's 1D vector of class scores.

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

## Design decisions

The main decisions are below. The [research log](research-log/2026-10-25.md) has the full reasoning.

| Decision | Why |
|---|---|
| Write matmul as three loops first | Every multiply-add is visible, and the shape rule `(m, n) @ (n, p) -> (m, p)` follows from the loops |
| Rewrite matmul with einsum | `'ik,kj->ij'` maps onto the three loops in one line, and the same notation carries through to batched matmul and attention |
| Use einsum in softmax | Practice with a case where nothing is summed. Plain broadcasting is just as simple here |
| Subtract the max before `exp` | `exp(1000)` overflows, and shifting every score by the same amount doesn't change softmax |
| Move from a `(1, 4)` weight vector to a `(3, 4)` matrix | Classification needs one score per class for each sample, not one score per sample |
| Store weights as `(classes, features)` and use `w.T` | Each row is one class's weights, which is also the convention in PyTorch's `nn.Linear` |
| Add `1e-9` inside the log | It stops `log(0)` giving `-inf`, capping the worst case at about 20.7 |
| Average the loss over the batch | With a sum, the loss and a good learning rate would both depend on batch size |
| Benchmark on 500x500 matrices | Big enough that the arithmetic outweighs timer noise, small enough that the loop finishes in under a minute |

## Challenges and fixes

| Challenge | Fix |
|---|---|
| `softmax_einsum` subtracted the whole matrix's max, so rows on very different scales produced `nan` | Took each row's max with `axis=1, keepdims=True` |
| `softmax` summed the whole array, so on a 2D input the samples' probabilities competed with each other and inflated the loss in `cross_entropy.py`'s demo | Took the max and sum over `axis=-1`, which works on 1D and 2D input |
| `matmul` never checked the inner dimensions, so `(3, 4) @ (2, 3)` silently returned a wrong `(3, 3)` result | Raised a `ValueError` naming both shapes |
| The loop `matmul` and `softmax` were copied between files, so each copy had the same bug | Kept one version of each and imported it |
| The tests weren't seeded, so a failure couldn't be reproduced | Drew test inputs from `np.random.default_rng(0)` |
| Importing a module ran its demo code | Put each demo under `if __name__ == "__main__":` |
| Tests only passed from inside the folder | Added `pythonpath` and `testpaths` to the root `pyproject.toml` |

Each of the three logic bugs now has a test that fails on the old code. The research log explains them in detail.

## Running the Tests

From the repo root:

```bash
pytest 2_linalg/tests
```

Expected output:

```
11 passed in 0.xxs
```

Plain `pytest` from the repo root runs the tests for every day.

The tests use a seeded random generator, so a failure can be reproduced.

## Concepts Covered

| Concept | Location |
|---|---|
| Matrix shapes and the inner-dimension rule | `ops.py` |
| Defensive validation | `ValueError` on mismatched shapes in `matmul` |
| Per-row reductions with `axis` and `keepdims` | `softmax.py`, `einsum_softmax.py` |
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
