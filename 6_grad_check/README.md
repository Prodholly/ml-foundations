# Day 6: Gradient Checking & Vectorization Benchmarks

Gradient checking answers one question: is the gradient formula I derived actually right? It answers it with
nothing but the definition of a derivative. So before any code, here's that definition, built up from pictures.

## Start here: what a derivative really is

This is the note I made while watching 3Blue1Brown's *Essence of Calculus*:

![my note on the area under x squared](calc_note1.jpg)

If `A(x)` is the area under `y = x²`, then pushing `x` a tiny bit adds a thin sliver of area, `δA ≈ x² · δx`.
Rearranged, `δA / δx ≈ x²`. **The derivative is the ratio of a tiny change in output to the tiny change in input
that caused it.**

### 1. A circle is a stack of rings

![circle area](figures/circle_area.gif)

Cut a circle into thin rings and straighten each one out. A ring at radius `r` becomes a strip about `2πr` long
and `dr` thick. Laid side by side, the strips form a triangle with area `½ · R · 2πR = πR²`.

Read it the other way: growing the radius by a tiny `dr` adds one more ring, `2πr · dr` of area. So the
derivative of `A = πr²` is `2πr`, the circumference.

**Why the one-sided estimate is off by `πh`.** Grow the radius from `r` to `r + h` and measure the new ring:

$$\frac{A(r+h) - A(r)}{h} = \frac{\pi(r+h)^2 - \pi r^2}{h} = \frac{2\pi r h + \pi h^2}{h} = 2\pi r + \pi h$$

The extra `πh` exists because a real ring isn't a perfectly thin strip. Its outer edge is longer than its
inner edge, so it holds a little more area than `2πr · h`. That leftover is what `πh` measures.

Notice the error is proportional to `h`: halve the step and you only halve the error. Getting close takes a very
small `h`, and that's where floating-point round-off starts to bite.

**The centered fix.** Instead, nudge both ways and measure the band from `r − h` to `r + h`:

$$\frac{A(r+h) - A(r-h)}{2h} = \frac{4\pi r h}{2h} = 2\pi r$$

The `πh²` terms cancel, because the band's outer half has a little too much area and its inner half a little
too little. For a circle this is exactly `2πr` at any `h`. That's why `numerical_gradient` uses the centered
formula.

### 2. My note, animated

![area under x squared](figures/area_under_x2.gif)

The green region is `A(x)`, the area under the curve up to `x` (here `x = 1.5`). Push `x` a bit to the right and
a thin sliver gets added. That sliver is almost a rectangle, `x²` tall and `h` wide, plus a small red curved
piece on top that the rectangle misses, because the curve keeps rising across the sliver.

As `h` shrinks, that red piece shrinks faster than the rectangle, so `sliver ÷ h` gets closer and closer to `x²`.
The panel on the right plots this estimate as `h` shrinks.

**Where the numbers come from.** The exact area under `x²` is `A(x) = x³/3`, so the one-sided estimate works out
to:

$$\frac{A(x+h) - A(x)}{h} = x^2 + xh + \frac{h^2}{3}$$

The `xh` term is the red piece, and it's large. At `x = 1.5` and `h = 0.1`, the estimate is about 2.40 instead
of 2.25.

**The centered version.** Take the sliver from `x − h` to `x + h` instead. Now the rectangle misses a piece on
the right (red), where the curve rises above it, and over-counts a piece on the left (purple), where the curve
dips below it. Those two pieces are almost the same size, so they almost cancel:

$$\frac{A(x+h) - A(x-h)}{2h} = \frac{(x+h)^3 - (x-h)^3}{6h} = \frac{6x^2h + 2h^3}{6h} = x^2 + \frac{h^2}{3}$$

The `xh` terms cancel exactly. What's left, `h²/3`, comes from the curve bending: it's slightly steeper on the
right than on the left, so the red piece is a little bigger than the purple one. At `h = 0.1` that's `0.0033`,
so the estimate is 2.2533 against 2.40 for one-sided. Shrink `h` by 10× and the one-sided error drops 10×, but
the centered error drops 100×.

## The build

```
6_grad_check/
├── grad_check/
│   ├── numerical.py      # numerical_gradient: centered difference, any shape
│   └── analytical.py     # softmax, cross_entropy_loss, forward, analytical_gradient_W
├── examples/
│   ├── run_all.py        # the five checks + benchmark, prints everything
│   ├── figures.py        # the readme figures below
│   └── calculus_intuition.py
├── tests/test_grad_check.py
└── figures/
```

Run from `6_grad_check`:

```bash
python3 examples/run_all.py        # checks, benchmark, speedup_curve.png
python3 examples/figures.py        # grad_intuition, grad_check_result, h_sweep
python -m examples.calculus_intuition
pytest tests -v
```

### 1. The numerical gradient

`numerical_gradient(f, x, h=1e-5)` nudges one element of `x` up by `h`, then down by `h`, and takes
`(f(x+h) − f(x−h)) / 2h`. That's the centered formula from above, once per element. It walks the array with
`np.nditer`, so the same loop works for a vector, a matrix or a 0-d scalar.

It starts with `x = x.astype(float)`. My first run printed `[50000, 150000, 250000, 350000]` for `d/dx sum(x²)`
instead of `[2, 4, 6, 8]`: `np.arange(1, 5)` is an int array, and `1 − 0.00001` got stored as `0`. The cast also
makes a copy, so the caller's array is never touched.

### 2. The analytical gradient

For one sample with true class `k`, the loss is `−log(p_k)`, and its derivative with respect to each logit is
`p_j − 1` for the true class and `p_j` for the others. In one line: `dL/dz = probs − y_true`. Averaged over the
batch and pushed back through `z = X @ W`, that becomes `dL/dW = X.T @ (probs − y_true) / N`.

![what probs - y_true says](figures/grad_intuition.png)

This sample is badly wrong: it puts 0.90 on class 2 when the answer is class 0. The gradient says exactly how
to fix that: a big negative push on class 0 (raise its score) and a big positive push on class 2 (lower it).
Class 1 was already near zero, so it barely moves.

### 3. Do they agree?

![analytical vs numerical](figures/grad_check_result.png)

All 12 entries of `dL/dW` sit on the line `y = x`. The max relative error is **1.4e-8**, against a target
of 1e-5 (CS231n's thresholds: below 1e-7 is excellent, above 1e-3 is almost certainly a bug).

### 4. Picking h, on the real loss

![h sweep](figures/h_sweep.png)

The same U shape as the calculus section, now measured on the softmax loss. Anything from about `1e-4` to
`1e-6` works, and `1e-5` sits right in the middle. The flat floor at 1e-8 isn't round-off: it's the `+ 1e-9`
inside `log()` in my loss, which the analytical formula doesn't know about.

### 5. Loop vs NumPy

![speedup curve](figures/speedup_curve.png)

| n | loop | numpy | speedup |
|---:|---:|---:|---:|
| 100 | 14 µs | 0.3 µs | ~45x |
| 10,000 | 1.4 ms | 2.7 µs | ~510x |
| 1,000,000 | 145 ms | 0.7 ms | ~200x |

The loop is a straight line: 10× the elements, 10× the time. NumPy is nearly flat until about 1,000 elements,
because the fixed cost of the call is the whole cost. The speedup peaks near 100k and drops at 1M, where NumPy's
own line gets steeper.

### Tests

8 tests: softmax rows sum to 1 (including logits ×1000), loss ≈ 0 for perfect predictions, loss = `log(C)`
for uniform ones, gradient shape matches `W`, the gradient check itself, and three edge cases for the
numerical gradient: int input, a 0-d scalar, and the input array left unchanged.

## Credit

The ring and sliver pictures are the ones Grant Sanderson uses in the first two chapters of
[**3Blue1Brown, *Essence of Calculus***](https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr&si=guWmcdNLRhyTxTgw).
The animations here are my own versions, but the way of seeing it is his. Watch the originals.
