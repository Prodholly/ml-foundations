# Day 6: Gradient Checking & Vectorization Benchmarks

Gradient checking answers one question: is the gradient formula I derived actually right? It answers it with
nothing but the definition of a derivative. So before any code, here's that definition, built up from pictures.

## Start here: what a derivative really is

This is the note I made while watching 3Blue1Brown's *Essence of Calculus*:

![my note on the area under x squared](calc_note1.jpg)

If `A(x)` is the area under `y = x²`, then pushing `x` a tiny bit adds a thin sliver of area, `δA ≈ x² · δx`.
Rearranged, `δA / δx ≈ x²`. **The derivative is the ratio of a tiny change in output to the tiny change in input
that caused it.** Everything in today's build is that sentence turned into code.

### 1. A circle is a stack of rings

![circle area](figures/circle_area.gif)

Cut a circle into thin rings, then cut each ring at the bottom and straighten it out. A ring at radius `r` becomes
a strip about `2πr` tall and `dr` wide. Laid side by side, the strips form a triangle: `½ · R · 2πR = πR²`.

Read the other way, adding one more ring adds `2πr · dr` of area, so `dA/dr = 2πr`. The last part of the
animation nudges `r` both ways and measures the band between `r − h` and `r + h`:

$$\frac{A(r+h) - A(r-h)}{2h}$$

For a circle this is exactly `2πr` at every `h`. The one-sided version, `(A(r+h) − A(r)) / h`, is off by `πh`.

### 2. My note, animated

![area under x squared](figures/area_under_x2.gif)

The sliver is almost a rectangle `x²` tall. The red bit on top is what the rectangle misses, and it shrinks like
`δx²`, faster than the sliver, so `δA / δx → x²`. The panel on the right tracks the estimate as `h` shrinks.

Then the centered version. The rectangle now misses a bit on the right (red) but counts a bit too much on the
left (purple), and the two nearly cancel. The one-sided estimate crawls in along a straight line, with error
about `x·h`. The centered one hugs the true value from the start, with error `h²/3`.

### 3. Why the h² disappears

![x squared as a square](figures/square_x2.gif)

`x²` is the area of a square. Grow the side by `h` and you add two strips plus a pink corner `h²`. That corner is
the whole error of the one-sided formula: `((x+h)² − x²) / h = 2x + h`.

Now grow by `h` and shrink by `h`. The difference is two strips `2h` wide, one `x + h` long and one `x − h` long.
Cut the extra `h` off the long one and it fills the gap in the short one exactly. Both become `2h × x`, the area
is `4xh`, and dividing by `2h` gives `2x` with nothing left over.

### 4. From derivative to gradient

![gradient as a hike](figures/gradient_2d.gif)

A loss depends on many inputs at once, like the height of a hill depends on east and north. To find uphill,
stand still and take one tiny step east and west, then one north and south. Each pair of steps gives one partial
derivative, and together they make the gradient arrow. That loop is `numerical_gradient`, and its answer matches
the analytical gradient `(−x, −3y)` to about `1e-11`.

Follow the arrow step by step and you reach the summit. Flip its sign and you walk downhill instead, which is
gradient descent, the thing that trains every model.

### 5. How small should h be?

![error vs h](figures/error_vs_h.png)

Smaller isn't always better. If `h` is too big, the curve bends away from the secant line. If it's too small,
below about `1e-8`, floating-point round-off swamps the estimate. The centered formula bottoms out near
`h = 1e-5`, which is why that's the default step size in `numerical_gradient`.

To redraw these figures, run this from `6_grad_check`:

```bash
python -m examples.calculus_intuition
```

## Credit

The ring, sliver and square pictures are the ones Grant Sanderson uses in the first two chapters of
[**3Blue1Brown, *Essence of Calculus***](https://youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr&si=guWmcdNLRhyTxTgw).
The animations here are my own versions, drawn with matplotlib on graph paper like my notes, but the way of seeing it is his. Watch the
originals.

---

*The gradient checker, the softmax + cross-entropy check, and the vectorization benchmark are next.*
