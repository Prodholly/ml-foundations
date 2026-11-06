"""Draws the calculus intuition figures for Day 6: what a derivative is, and why the centered difference works.

Run from 6_grad_check:  python -m examples.calculus_intuition
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to files, no window

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle, Wedge

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# >> style: dark board, bright chalk

BG = "#0f1117"
FG = "#e8e8e8"
MUTED = "#7d8594"
BLUE = "#58c4dd"
YELLOW = "#f4d35e"
GREEN = "#83c167"
RED = "#fc6255"
PURPLE = "#b189d6"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "cm",
    "text.color": FG,
    "axes.labelcolor": FG,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
})


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)  # smoothstep: slow start, slow finish


def shrink(t, start, end):
    # move from start to end on a log scale, so small h gets as much screen time as large h
    return float(np.exp(np.log(start) + (np.log(end) - np.log(start)) * ease(t)))


def board(ax, equal=False):
    ax.set_facecolor(BG)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    if equal:
        ax.set_aspect("equal")


def new_figure(width, height):
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor(BG)
    return fig


def phases(**lengths):
    # turns phase lengths into (name, local t in [0, 1]) for any frame number
    names, bounds, total = [], [], 0
    for name, n in lengths.items():
        names.append(name)
        bounds.append((total, total + n))
        total += n

    def which(f):
        for name, (a, b) in zip(names, bounds):
            if f < b:
                return name, (f - a) / max(b - a - 1, 1)
        return names[-1], 1.0

    return which, total


def save_gif(fig, draw, frames, name, fps=15):
    anim = animation.FuncAnimation(fig, draw, frames=frames)
    anim.save(FIGURES / name, writer=animation.PillowWriter(fps=fps), dpi=80,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)


# >> 1. the circle: area grows by one ring at a time

def circle_area_gif():
    R, N = 1.0, 10
    dr = R / N
    r0 = 0.6
    which, total = phases(intro=12, rings=N * 6, triangle=22, centered=60, hold=24)

    fig = new_figure(11, 6.2)
    left = fig.add_axes([0.03, 0.2, 0.42, 0.68])
    right = fig.add_axes([0.55, 0.2, 0.4, 0.66])

    def draw(f):
        phase, t = which(f)
        left.clear()
        right.clear()
        board(left, equal=True)
        left.axis("off")
        left.set_xlim(-1.2, 1.2)
        left.set_ylim(-1.2, 1.2)
        board(right)
        right.set_xlim(0, 1.08)
        right.set_ylim(0, 6.9)
        right.set_xlabel("radius", fontsize=12, color=MUTED)
        right.set_xticks([0, r0, 1])
        right.set_xticklabels(["0", "r", "R"])
        right.set_yticks([])
        for txt in fig.texts[:]:
            txt.remove()

        fig.text(0.5, 0.94, "Why is a circle's area $\\pi R^2$?   Cut it into rings.",
                 ha="center", fontsize=17, fontweight="bold")
        left.add_patch(plt.Circle((0, 0), R, fill=False, color=BLUE, linewidth=2))

        if phase == "intro":
            fig.text(0.5, 0.07, "each ring is a thin strip: unroll it and it is almost a rectangle",
                     ha="center", fontsize=13, color=MUTED)
            return

        # how many rings have been unrolled so far
        if phase == "rings":
            k_now, grow = int(t * N * 0.999), ease((t * N) % 1 * 1.6)
        else:
            k_now, grow = N, 1.0
        faded = phase == "centered" or phase == "hold"

        for k in range(min(k_now + 1, N)):
            r = k * dr
            current = phase == "rings" and k == k_now
            color = YELLOW if current else BLUE
            alpha = 0.18 if faded else (1.0 if current else 0.55)
            left.add_patch(Wedge((0, 0), r + dr, 0, 360, width=dr * 0.92, color=color, alpha=alpha))
            height = 2 * np.pi * r * (grow if current else 1.0)
            right.add_patch(Rectangle((r, 0), dr * 0.92, height, color=color, alpha=alpha))
            if current:
                right.text(r + dr / 2, height + 0.2, "$2\\pi r$", ha="center", fontsize=13, color=YELLOW)
                left.text(0, -1.17, f"ring at r = {r:.1f},  thickness dr = {dr}", ha="center",
                          fontsize=11, color=YELLOW)

        if phase == "rings":
            fig.text(0.5, 0.07, "a ring at radius r has length $2\\pi r$ and thickness $dr$:   area $\\approx 2\\pi r\\,dr$",
                     ha="center", fontsize=14)

        if phase == "triangle":
            s = ease(t * 1.5)
            right.plot([0, R * s], [0, 2 * np.pi * R * s], color=FG, linewidth=2)
            right.plot([R, R], [0, 2 * np.pi * R * s], color=FG, linewidth=1, linestyle=":")
            right.text(1.02, np.pi, "$2\\pi R$", fontsize=13, va="center")
            fig.text(0.5, 0.07, "the strips fill a triangle:   area $= \\frac{1}{2} \\cdot R \\cdot 2\\pi R = \\pi R^2$",
                     ha="center", fontsize=15)

        if phase in ("centered", "hold"):
            h = shrink(t, 0.28, 0.02) if phase == "centered" else 0.02
            rr = np.linspace(0, 1.05, 50)
            right.plot(rr, 2 * np.pi * rr, color=BLUE, linewidth=2.5)
            right.text(0.08, 5.9, "$\\dfrac{dA}{dr} = 2\\pi r$", fontsize=16, color=BLUE)
            # the band between r0 - h and r0 + h, in both pictures
            left.add_patch(Wedge((0, 0), r0 + h, 0, 360, width=2 * h, color=YELLOW, alpha=0.85))
            left.add_patch(plt.Circle((0, 0), r0, fill=False, color=FG, linewidth=1.2, linestyle="--"))
            band = np.linspace(r0 - h, r0 + h, 20)
            right.fill_between(band, 0, 2 * np.pi * band, color=YELLOW, alpha=0.85)
            right.axvline(r0, color=FG, linewidth=1, linestyle="--")

            area_band = np.pi * ((r0 + h) ** 2 - (r0 - h) ** 2)
            forward = np.pi * ((r0 + h) ** 2 - r0 ** 2) / h
            left.text(0, -1.17, f"h = {h:.3f}", ha="center", fontsize=12, color=YELLOW, family="monospace")
            fig.text(0.5, 0.115, "$\\dfrac{A(r+h) - A(r-h)}{2h}$" + f"  =  {area_band / (2 * h):.4f}",
                     ha="center", fontsize=17, color=YELLOW)
            fig.text(0.5, 0.035,
                     f"centered: exactly $2\\pi r$ = {2 * np.pi * r0:.4f} for every h        "
                     f"one-sided $\\frac{{A(r+h) - A(r)}}{{h}}$ = {forward:.4f}  (off by $\\pi h$)",
                     ha="center", fontsize=12, color=MUTED)

    save_gif(fig, draw, total, "circle_area.gif")


# >> 2. the area under x^2: a sliver of width dx has area about x^2 dx

def parabola_area_gif():
    x0 = 1.5
    A = lambda x: x ** 3 / 3  # area under t^2 from 0 to x
    which, total = phases(sweep=26, sliver=18, forward=46, centered=46, hold=26)

    fig = new_figure(11, 6.2)
    ax = fig.add_axes([0.07, 0.22, 0.56, 0.66])
    side = fig.add_axes([0.66, 0.22, 0.32, 0.66])
    xs = np.linspace(0, 2.15, 200)

    def draw(f):
        phase, t = which(f)
        ax.clear()
        board(ax)
        ax.set_xlim(0, 2.2)
        ax.set_ylim(0, 4.9)
        ax.set_xticks([x0])
        ax.set_xticklabels(["x"], fontsize=13)
        ax.set_yticks([])
        for txt in fig.texts[:]:
            txt.remove()
        fig.text(0.5, 0.94, "The area under $x^2$ grows at a rate of $x^2$", ha="center",
                 fontsize=17, fontweight="bold")

        ax.plot(xs, xs ** 2, color=BLUE, linewidth=2.5)
        ax.text(1.35, 4.3, "$y = x^2$", color=BLUE, fontsize=16)

        sweep_to = x0 * ease(t) if phase == "sweep" else x0
        fill = np.linspace(0, sweep_to, 100)
        ax.fill_between(fill, 0, fill ** 2, color=BLUE, alpha=0.25)
        ax.text(sweep_to * 0.62, 0.35, "$A(x)$", fontsize=16, color=FG, ha="center")

        side.clear()
        side.axis("off")

        def readout(lines):
            for k, (text, color, size) in enumerate(lines):
                side.text(0.0, 0.92 - 0.14 * k, text, fontsize=size, color=color, transform=side.transAxes)

        if phase == "sweep":
            readout([("what is $\\dfrac{dA}{dx}$ ?", FG, 18)])
        elif phase == "sliver":
            dx = 0.4
            ax.add_patch(Rectangle((x0, 0), dx, x0 ** 2, color=GREEN, alpha=0.75 * ease(t * 2)))
            top = np.linspace(x0, x0 + dx, 30)
            ax.fill_between(top, x0 ** 2, top ** 2, color=RED, alpha=0.8 * ease(t * 2 - 0.5))
            ax.annotate("missed bit $\\approx x\\,\\delta x^2$\nshrinks faster than $\\delta x$",
                        (x0 + dx * 0.7, x0 ** 2 + 0.35), (x0 - 0.15, 3.6), ha="right", fontsize=11, color=RED,
                        alpha=ease(t * 2 - 0.5), arrowprops=dict(arrowstyle="->", color=RED, alpha=ease(t * 2 - 0.5)))
            ax.annotate("", (x0 + dx + 0.05, x0 ** 2), (x0 + dx + 0.05, 0),
                        arrowprops=dict(arrowstyle="<->", color=FG))
            ax.text(x0 + dx + 0.08, x0 ** 2 / 2, "$x^2$", fontsize=14)
            readout([("push x a little: $\\delta x$", GREEN, 15),
                     ("new sliver $\\approx$ a rectangle", FG, 13),
                     ("$\\delta A \\approx x^2 \\cdot \\delta x$", GREEN, 18),
                     ("so  $\\dfrac{\\delta A}{\\delta x} \\approx x^2$", FG, 18)])
        elif phase == "forward":
            dx = shrink(t, 0.4, 0.02)
            band = np.linspace(x0, x0 + dx, 30)
            ax.fill_between(band, 0, band ** 2, color=GREEN, alpha=0.85)
            rate = (A(x0 + dx) - A(x0)) / dx
            readout([("one-sided:  $\\dfrac{A(x+h) - A(x)}{h}$", GREEN, 16),
                     (f"h = {dx:.3f}", FG, 13),
                     (f"= {rate:.4f}", GREEN, 18),
                     (f"true $x^2$ = {x0 ** 2:.4f}", FG, 13),
                     (f"error = {rate - x0 ** 2:.4f}   ($\\approx x h$)", RED, 13)])
        else:
            h = shrink(t, 0.4, 0.02) if phase == "centered" else 0.02
            band = np.linspace(x0 - h, x0 + h, 30)
            ax.fill_between(band, 0, band ** 2, color=YELLOW, alpha=0.85)
            ax.axvline(x0, color=FG, linewidth=1, linestyle="--")
            rate = (A(x0 + h) - A(x0 - h)) / (2 * h)
            readout([("centered:  $\\dfrac{A(x+h) - A(x-h)}{2h}$", YELLOW, 16),
                     (f"h = {h:.3f}", FG, 13),
                     (f"= {rate:.4f}", YELLOW, 18),
                     (f"true $x^2$ = {x0 ** 2:.4f}", FG, 13),
                     (f"error = {rate - x0 ** 2:.5f}   ($= h^2/3$)", GREEN, 13)])
            if phase == "hold":
                fig.text(0.5, 0.06, "the sliver on the left and the sliver on the right lean opposite ways,"
                         " so their errors cancel", ha="center", fontsize=13, color=MUTED)
        if phase in ("forward", "centered"):
            fig.text(0.5, 0.06, "shrink h: the rate of change of the area is the height of the curve",
                     ha="center", fontsize=13, color=MUTED)

    save_gif(fig, draw, total, "area_under_x2.gif")


# >> 3. the square: why the centered nudge leaves no h^2 behind

def square_gif():
    x0 = 1.5
    which, total = phases(forward=56, pause=10, centered=56, hold=26)

    fig = new_figure(11, 6.2)
    left = fig.add_axes([0.03, 0.2, 0.42, 0.68])
    right = fig.add_axes([0.55, 0.2, 0.41, 0.66])
    xs = np.linspace(0, 2.5, 200)

    def draw(f):
        phase, t = which(f)
        left.clear()
        right.clear()
        board(left, equal=True)
        left.axis("off")
        left.set_xlim(-0.25, 2.25)
        left.set_ylim(-0.3, 2.2)
        board(right)
        right.set_xlim(0, 2.5)
        right.set_ylim(0, 6.3)
        right.set_xticks([x0])
        right.set_xticklabels(["x"], fontsize=13)
        right.set_yticks([])
        for txt in fig.texts[:]:
            txt.remove()
        fig.text(0.5, 0.94, "$f(x) = x^2$ is the area of a square", ha="center", fontsize=17, fontweight="bold")

        right.plot(xs, xs ** 2, color=BLUE, linewidth=2.5)
        tangent = x0 ** 2 + 2 * x0 * (xs - x0)
        right.plot(xs, tangent, color=YELLOW, linewidth=1.6, alpha=0.9)
        right.text(2.05, x0 ** 2 + 2 * x0 * 0.55 - 0.6, "slope $2x$", color=YELLOW, fontsize=12)

        centered = phase in ("centered", "hold")
        if phase == "forward":
            h = shrink(t, 0.55, 0.04)
        elif phase == "pause":
            h = 0.04
        else:
            h = shrink(t, 0.55, 0.04) if phase == "centered" else 0.04

        if not centered:
            left.add_patch(Rectangle((0, 0), x0, x0, color=BLUE, alpha=0.7))
            left.add_patch(Rectangle((x0, 0), h, x0, color=GREEN, alpha=0.85))
            left.add_patch(Rectangle((0, x0), x0, h, color=GREEN, alpha=0.85))
            left.add_patch(Rectangle((x0, x0), h, h, color=RED, alpha=0.95))
            left.text(x0 / 2, x0 / 2, "$x^2$", ha="center", va="center", fontsize=18)
            left.text(x0 + h + 0.06, x0 + h + 0.06, "$h^2$", color=RED, fontsize=15)
            left.text(x0 / 2, -0.2, "x", ha="center", fontsize=13)
            left.text(x0 + h / 2, -0.2, "h", ha="center", fontsize=12, color=GREEN)

            a, b = x0, x0 + h
            slope = (b ** 2 - a ** 2) / (b - a)
            right.plot(xs, a ** 2 + slope * (xs - a), color=RED, linewidth=1.6, linestyle="--")
            right.plot([a, b], [a ** 2, b ** 2], "o", color=RED, markersize=6)
            fig.text(0.5, 0.115, "$(x+h)^2 - x^2 = 2xh + h^2$", ha="center", fontsize=17, color=GREEN)
            fig.text(0.5, 0.04, f"one-sided slope = 2x + h = {slope:.3f}      true 2x = {2 * x0:.3f}"
                     "      the red corner $h^2$ is the error", ha="center", fontsize=12, color=MUTED)
        else:
            inner = x0 - h
            left.add_patch(Rectangle((0, 0), x0, x0, fill=False, edgecolor=FG, linewidth=1.2, linestyle="--"))
            left.add_patch(Rectangle((0, 0), inner, inner, color=BLUE, alpha=0.7))
            left.add_patch(Rectangle((inner, 0), 2 * h, x0 + h, color=YELLOW, alpha=0.85))
            left.add_patch(Rectangle((0, inner), inner, 2 * h, color=YELLOW, alpha=0.85))
            left.text(inner / 2, inner / 2, "$(x-h)^2$", ha="center", va="center", fontsize=15)
            left.text(x0 + h + 0.07, (x0 + h) / 2, "length $x+h$", rotation=90, va="center", fontsize=11, color=YELLOW)
            left.text(inner / 2, x0 + h + 0.07, "length $x-h$", ha="center", fontsize=11, color=YELLOW)
            left.text(x0, -0.2, "2h", ha="center", fontsize=12, color=YELLOW)

            a, b = x0 - h, x0 + h
            slope = (b ** 2 - a ** 2) / (b - a)
            right.plot(xs, a ** 2 + slope * (xs - a), color=GREEN, linewidth=1.8, linestyle="--")
            right.plot([a, b], [a ** 2, b ** 2], "o", color=GREEN, markersize=6)
            fig.text(0.5, 0.115, "$(x+h)^2 - (x-h)^2 = 2h\\,(x+h) + 2h\\,(x-h) = 4xh$",
                     ha="center", fontsize=17, color=YELLOW)
            fig.text(0.5, 0.04, f"centered slope = 4xh / 2h = {slope:.3f}, exactly 2x for any h:"
                     "  one strip is h too long, the other h too short", ha="center", fontsize=12, color=MUTED)
        right.text(0.1, 5.8, f"h = {h:.3f}", fontsize=13, family="monospace", color=FG)

    save_gif(fig, draw, total, "square_x2.gif")


# >> 4. the gradient: nudge one coordinate at a time

def gradient_gif():
    f = lambda x, y: 0.5 * x ** 2 + 1.5 * y ** 2
    p = np.array([1.2, 0.6])
    true_grad = np.array([p[0], 3 * p[1]])  # (df/dx, df/dy)
    h_show = 0.4  # big enough to see; the printed numbers use h = 1e-5

    def num_partial(i, h=1e-5):
        e = np.zeros(2)
        e[i] = h
        return (f(*(p + e)) - f(*(p - e))) / (2 * h)

    num_grad = np.array([num_partial(0), num_partial(1)])
    rel_error = np.max(np.abs(num_grad - true_grad) / np.maximum(np.abs(num_grad), np.abs(true_grad)))
    which, total = phases(intro=16, dx=38, dy=38, combine=30, hold=26)

    fig = new_figure(11, 6.2)
    ax = fig.add_axes([0.03, 0.12, 0.56, 0.76])
    gx, gy = np.meshgrid(np.linspace(-2.4, 2.4, 200), np.linspace(-1.7, 1.7, 200))
    side = fig.add_axes([0.62, 0.12, 0.36, 0.76])
    scale = 0.45  # arrow length per unit of gradient

    def draw(frame):
        phase, t = which(frame)
        ax.clear()
        board(ax, equal=True)
        ax.axis("off")
        ax.set_xlim(-2.4, 2.4)
        ax.set_ylim(-1.7, 1.7)
        for txt in fig.texts[:]:
            txt.remove()
        fig.text(0.5, 0.94, "Gradient = one derivative per input, found one nudge at a time",
                 ha="center", fontsize=17, fontweight="bold")
        ax.contour(gx, gy, f(gx, gy), levels=[0.15, 0.4, 0.8, 1.26, 1.9, 2.7, 3.6], colors=[BLUE],
                   linewidths=1.2, alpha=0.6)
        ax.contour(gx, gy, f(gx, gy), levels=[f(*p)], colors=[BLUE], linewidths=2.4)
        ax.plot(*p, "o", color=FG, markersize=9, zorder=5)
        ax.text(p[0] - 0.12, p[1] + 0.14, "(x, y)", fontsize=12, ha="right")
        ax.text(-2.3, -1.6, "$f(x, y) = \\frac{1}{2}x^2 + \\frac{3}{2}y^2$", fontsize=14, color=BLUE)

        side.clear()
        side.axis("off")
        code = ["for i in range(len(x)):", "    x[i] += h   -> f(x+h)", "    x[i] -= 2h  -> f(x-h)",
                "    grad[i] = (f(x+h) - f(x-h)) / 2h"]
        active_i = {"dx": 0, "dy": 1}.get(phase)
        side.text(0, 0.97, "numerical_gradient", fontsize=13, fontweight="bold", transform=side.transAxes)
        for k, line in enumerate(code):
            side.text(0, 0.88 - 0.065 * k, line, fontsize=10, family="monospace", transform=side.transAxes,
                      color=YELLOW if active_i is not None and k > 0 else MUTED)
        if active_i is not None:
            side.text(0, 0.6, f"i = {active_i}  ({'x' if active_i == 0 else 'y'})", fontsize=13,
                      family="monospace", color=RED if active_i == 0 else GREEN, transform=side.transAxes)

        def nudge(i, s, color, name):
            e = np.zeros(2)
            e[i] = h_show * ease(s * 1.6)
            for sign in (1, -1):
                q = p + sign * e
                ax.plot(*q, "o", color=color, markersize=7, zorder=5)
                ax.text(q[0] + (0.06 if i == 1 else 0), q[1] + (0.12 if i == 0 else 0) * sign
                        + (0 if i == 0 else 0.04), f"{f(*q):.2f}", fontsize=10, color=color,
                        ha="center" if i == 0 else "left")
            ax.plot([p[0] - e[0], p[0] + e[0]], [p[1] - e[1], p[1] + e[1]], color=color, linewidth=1.5)

        def component(i, grow, color):
            d = np.zeros(2)
            d[i] = true_grad[i] * scale * grow
            ax.add_patch(FancyArrowPatch(tuple(p), tuple(p + d), arrowstyle="-|>", mutation_scale=18,
                                         color=color, linewidth=2.4, zorder=4))

        lines = []
        if phase == "intro":
            lines = [("at this point, which way is uphill,", FG), ("and how steep?", FG)]
        if phase == "dx":
            nudge(0, t, RED, "x")
            component(0, ease((t - 0.4) * 2), RED)
            lines = [("hold y still, nudge only x:", RED),
                     (f"$\\dfrac{{\\partial f}}{{\\partial x}} \\approx \\dfrac{{f(x+h,y) - f(x-h,y)}}{{2h}}$ = {num_grad[0]:.2f}", RED)]
        if phase in ("dy", "combine", "hold"):
            component(0, 1, RED)
        if phase == "dy":
            nudge(1, t, GREEN, "y")
            component(1, ease((t - 0.4) * 2), GREEN)
            lines = [("hold x still, nudge only y:", GREEN),
                     (f"$\\dfrac{{\\partial f}}{{\\partial y}} \\approx \\dfrac{{f(x,y+h) - f(x,y-h)}}{{2h}}$ = {num_grad[1]:.2f}", GREEN)]
        if phase in ("combine", "hold"):
            component(1, 1, GREEN)
            s = ease(t * 1.4) if phase == "combine" else 1
            ax.add_patch(FancyArrowPatch(tuple(p), tuple(p + true_grad * scale * s), arrowstyle="-|>",
                                         mutation_scale=22, color=YELLOW, linewidth=3, zorder=6))
            ax.plot([p[0] + true_grad[0] * scale, p[0] + true_grad[0] * scale],
                    [p[1], p[1] + true_grad[1] * scale], color=GREEN, linewidth=1, linestyle=":")
            ax.text(*(p + true_grad * scale + np.array([0.05, 0.05])), "$\\nabla f$", fontsize=17, color=YELLOW)
            lines = [(f"numerical   ({num_grad[0]:.6f}, {num_grad[1]:.6f})", YELLOW),
                     (f"analytical  (x, 3y) = ({true_grad[0]:.6f}, {true_grad[1]:.6f})", FG),
                     (f"max relative error = {rel_error:.1e}", GREEN),
                     ("points straight across the contour, uphill", MUTED)]
        for k, (text, color) in enumerate(lines):
            side.text(0, 0.45 - 0.09 * k, text, fontsize=11.5 if "$" not in text else 14, color=color,
                      transform=side.transAxes)

    save_gif(fig, draw, total, "gradient_2d.gif")


# >> 5. how small should h be?

def error_vs_h_png():
    x, true = 1.0, np.cos(1.0)  # f = sin, so f'(1) = cos(1)
    hs = np.logspace(-13, -0.3, 300)
    forward = np.abs((np.sin(x + hs) - np.sin(x)) / hs - true)
    centered = np.abs((np.sin(x + hs) - np.sin(x - hs)) / (2 * hs) - true)

    fig = new_figure(10, 5.8)
    ax = fig.add_axes([0.1, 0.13, 0.86, 0.74])
    board(ax)
    ax.loglog(hs, forward, color=RED, linewidth=1.8, label="one-sided  $(f(x+h) - f(x))/h$")
    ax.loglog(hs, centered, color=YELLOW, linewidth=1.8, label="centered  $(f(x+h) - f(x-h))/2h$")
    ax.axvline(1e-5, color=GREEN, linewidth=1.2, linestyle="--")
    ax.text(1.4e-5, 3e-1, "h = 1e-5\nnumerical_gradient default", color=GREEN, fontsize=11)
    ax.annotate("sweet spot", (1e-5, 1e-11), (2e-8, 1e-12), color=GREEN, fontsize=11,
                arrowprops=dict(arrowstyle="->", color=GREEN))
    ax.text(3e-3, 1e-2, "error $\\propto h$", color=RED, fontsize=13)
    ax.text(3e-3, 3e-7, "error $\\propto h^2$", color=YELLOW, fontsize=13)
    ax.text(2e-13, 2e-2, "too small: floating-point\nround-off takes over ($\\propto \\epsilon / h$)",
            color=MUTED, fontsize=11)
    ax.set_xlabel("step size h", fontsize=12)
    ax.set_ylabel("|estimate - true derivative|", fontsize=12)
    ax.grid(True, which="major", color="#2a2f3a", linewidth=0.6)
    legend = ax.legend(loc="lower right", facecolor=BG, edgecolor=MUTED, fontsize=11)
    for text in legend.get_texts():
        text.set_color(FG)
    fig.suptitle("Error of the derivative of sin at x = 1, for every h", fontsize=15, fontweight="bold", y=0.96)
    fig.savefig(FIGURES / "error_vs_h.png", dpi=150, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    for make in (circle_area_gif, parabola_area_gif, square_gif, gradient_gif, error_vs_h_png):
        make()
        print(f"done: {make.__name__}")
