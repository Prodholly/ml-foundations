"""
calculus intuition figures for day 6: what a derivative is, and why the centered difference works.

everything is drawn like a page of my notebook: graph paper, pen, highlighter and sticky notes.
run from 6_grad_check:  python -m examples.calculus_intuition
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to files, no window

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle, Wedge

warnings.filterwarnings("ignore", message="Glyph")  # the handwriting font lacks a few symbols; mathtext covers them

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# >> the notebook

PAPER = "#fbf6e9"
GRID = "#dbe5ef"
INK = "#23314d"
PENCIL = "#8a8f9c"
BLUE = "#2f6db5"
GREEN = "#4caf50"
HIGHLIGHT = "#ffd43b"
PINK = "#d6336c"
RED = "#e8590c"
PURPLE = "#7048e8"
STICKY = "#fff0a3"

HAND = ["Noteworthy", "Bradley Hand", "Comic Sans MS", "DejaVu Sans"]
TITLE = ["Avenir Next", "Avenir", "DejaVu Sans"]

plt.rcParams.update({
    "font.family": TITLE,
    "mathtext.fontset": "cm",
    "text.color": INK,
    "hatch.linewidth": 1.6,
})


def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)  # smoothstep: slow start, slow finish


def shrink(t, start, end):
    # move from start to end on a log scale, so small h gets as much screen time as large h
    return float(np.exp(np.log(start) + (np.log(end) - np.log(start)) * ease(t)))


def lerp(a, b, t):
    return a + (b - a) * t


def pen(artist, wobble=1.0):
    # a slightly shaky line, like it was drawn by hand
    artist.set_sketch_params(0.7 * wobble, 90, 1.4)
    return artist


def notebook(width=11, height=6.2, step=0.24):
    # cream graph paper; the grid is drawn once and every frame is drawn on top of it
    fig = plt.figure(figsize=(width, height))
    fig.patch.set_facecolor(PAPER)
    bg = fig.add_axes([0, 0, 1, 1], zorder=-5)
    bg.set_xlim(0, width)
    bg.set_ylim(0, height)
    bg.axis("off")
    for x in np.arange(0, width, step):
        bg.axvline(x, color=GRID, linewidth=0.7)
    for y in np.arange(0, height, step):
        bg.axhline(y, color=GRID, linewidth=0.7)
    overlay = fig.add_axes([0, 0, 1, 1], zorder=10)
    return fig, overlay


def blank(ax, xlim=None, ylim=None, equal=False):
    ax.clear()
    ax.set_facecolor("none")
    ax.axis("off")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    if equal:
        ax.set_aspect("equal")


def heading(ov, title, subtitle=None):
    blank(ov, (0, 1), (0, 1))
    ov.text(0.5, 0.935, title, ha="center", va="center", fontsize=19, fontweight="bold", color=INK)
    if subtitle:
        ov.text(0.5, 0.875, subtitle, ha="center", va="center", fontsize=13, color=PENCIL, family=HAND, fontweight="bold")


def write(ax, x, y, text, color=INK, size=14, **kw):
    kw.setdefault("fontweight", "bold")
    return ax.text(x, y, text, color=color, fontsize=size, family=HAND, **kw)


def sticky(ax, x, y, w, h, lines, angle=1.5):
    # a sticky note with a soft shadow; lines are (text, color, size)
    ax.add_patch(Rectangle((x + w * 0.025, y - h * 0.035), w, h, angle=angle, rotation_point="center",
                           facecolor="#000000", alpha=0.08, zorder=8))
    ax.add_patch(Rectangle((x, y), w, h, angle=angle, rotation_point="center", facecolor=STICKY,
                           edgecolor="#f1d36b", linewidth=0.8, zorder=9))
    gap = h / (len(lines) + 0.6)
    for k, (text, color, size) in enumerate(lines):
        write(ax, x + w * 0.07, y + h - gap * (k + 0.85), text, color=color, size=size, va="center",
              rotation=angle, rotation_mode="anchor", zorder=10)


def arrow(ax, start, end, color=INK, width=1.8, rad=0.0, head=14, zorder=7, style="-|>"):
    return ax.add_patch(pen(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=head, color=color,
                                            linewidth=width, connectionstyle=f"arc3,rad={rad}", zorder=zorder)))


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


def save_gif(fig, draw, frames, name, fps=16):
    anim = animation.FuncAnimation(fig, draw, frames=frames)
    anim.save(FIGURES / name, writer=animation.PillowWriter(fps=fps), dpi=72,
              savefig_kwargs={"facecolor": PAPER})
    plt.close(fig)


# >> 1. a circle is a stack of rings: unroll them

def circle_area_gif():
    R, N = 1.1, 10
    dr = R / N
    C = np.array([-2.2, 3.3])  # circle center
    K = 1.7   # the circle is drawn K times larger than the strips, so the rings are visible
    X0, S = 1.2, 5.8 # where the unrolled strips start, and how much the r axis is stretched
    r0 = 0.66
    ring_colors = LinearSegmentedColormap.from_list("rings", ["#4c6ef5", "#15aabf", "#40c057"])
    theta = np.linspace(0, 2 * np.pi, 90)
    phi = -np.pi / 2 + theta	# cut each ring at the bottom and unroll from there

    start_at = lambda k: 10 + 5 * k
    unroll_len = 13
    which, total = phases(rings=start_at(N - 1) + unroll_len, triangle=26, centered=58, hold=26)

    fig, ov = notebook()
    ax = fig.add_axes([0.02, 0.03, 0.96, 0.8])

    def ring_shape(k, e):
        r_in, r_out = k * dr, (k + 1) * dr
        mid = (r_in + r_out) / 2
        shapes = []
        for rho in (r_out, r_in):
            on_circle = C[None, :] + K * rho * np.stack([np.cos(phi), np.sin(phi)], axis=1)
            on_strip = np.stack([np.full_like(theta, X0 + S * rho), mid * theta], axis=1)
            shapes.append(lerp(on_circle, on_strip, e))
        return np.vstack([shapes[0], shapes[1][::-1]])

    def draw(f):
        phase, t = which(f)
        blank(ax, (-4.3, 12.6), (-0.9, 7.4), equal=True)
        heading(ov, "Why is a circle's area $\\pi R^2$ ?", "cut it into thin rings ... then unroll every ring")

        # the r axis under the strips
        pen(ax.plot([X0 - 0.2, X0 + S * R + 0.5], [0, 0], color=INK, linewidth=1.6)[0])
        write(ax, X0 - 0.1, -0.55, "0", size=12)
        write(ax, X0 + S * R - 0.15, -0.55, "R", size=13)
        write(ax, X0 + S * R + 0.6, -0.12, "r", size=14, color=PENCIL)

        if phase == "rings":
            unrolled = [ease((f - start_at(k)) / unroll_len) for k in range(N)]
        else:
            unrolled = [1.0] * N
        faded = phase in ("centered", "hold")

        for k in range(N):
            e = unrolled[k]
            color = ring_colors(k / (N - 1))
            ax.add_patch(Polygon(ring_shape(k, e), closed=True, facecolor=color, edgecolor=PAPER,
                                 linewidth=0.6, alpha=0.28 if faded else 0.92, zorder=3))
            if faded:  # keep a ghost of the circle while the strips are faded
                ax.add_patch(Polygon(ring_shape(k, 0), closed=True, facecolor=color, edgecolor=PAPER,
                                     linewidth=0.6, alpha=0.22, zorder=2))
        circle = pen(plt.Circle(tuple(C), K * R, fill=False, color=INK, linewidth=1.6, linestyle="--", zorder=4))
        ax.add_patch(circle)

        if phase == "rings":
            k = 6
            if unrolled[k] > 0.97:
                top = (k + 0.5) * dr * 2 * np.pi
                x = X0 + S * (k + 1) * dr
                arrow(ax, (x + 1.6, top + 0.9), (x + 0.12, top - 0.2), color=PINK, rad=0.25)
                write(ax, x + 1.65, top + 0.8, "this strip is\n$2\\pi r$ tall, $dr$ wide", color=PINK, size=13)
            write(ax, C[0] - 1.9, 0.3, "ring area $\\approx 2\\pi r \\cdot dr$", size=15, color=INK)

        if phase == "triangle":
            s = ease(t * 1.6)
            top = (X0 + S * R, 2 * np.pi * R)
            pen(ax.plot([X0, lerp(X0, top[0], s)], [0, lerp(0, top[1], s)], color=PINK, linewidth=2.6,
                        zorder=6)[0], 0.8)
            pen(ax.plot([top[0], top[0]], [0, top[1] * s], color=PINK, linewidth=1.6, linestyle="--",
                        zorder=6)[0])
            if t > 0.45:
                write(ax, top[0] + 0.25, top[1] / 2, "$2\\pi R$", color=PINK, size=17)
                write(ax, X0 + S * R / 2 - 0.4, -0.95, "base $R$", color=PINK, size=14)
            if t > 0.6:
                sticky(ax, 8.6, 1.4, 3.8, 2.4, [
                    ("a triangle!", PINK, 16),
                    ("$\\frac{1}{2} \\cdot R \\cdot 2\\pi R$", INK, 19),
                    ("$= \\pi R^2$", PINK, 21)])

        if faded:
            h = shrink(t, 0.3, 0.015) if phase == "centered" else 0.015
            rr = np.linspace(0, R * 1.04, 60)
            pen(ax.plot(X0 + S * rr, 2 * np.pi * rr, color=BLUE, linewidth=2.6, zorder=6)[0])
            write(ax, X0 + 0.2, 6.1, "$\\dfrac{dA}{dr} = 2\\pi r$", color=BLUE, size=19)
            # the band from r - h to r + h, on the circle and under the line
            ax.add_patch(Wedge(tuple(C), K * (r0 + h), 0, 360, width=K * 2 * h, color=HIGHLIGHT, alpha=0.95, zorder=5))
            ax.add_patch(pen(plt.Circle(tuple(C), K * r0, fill=False, color=INK, linewidth=1, linestyle=":", zorder=6)))
            band = np.linspace(r0 - h, r0 + h, 20)
            ax.fill_between(X0 + S * band, 0, 2 * np.pi * band, color=HIGHLIGHT, alpha=0.95, zorder=5)
            pen(ax.plot([X0 + S * r0] * 2, [0, 2 * np.pi * r0], color=INK, linewidth=1, linestyle=":", zorder=6)[0])
            write(ax, X0 + S * r0 - 0.1, -0.55, "r", size=13)
            write(ax, C[0] - 0.75, C[1] - K * R - 0.6, f"h = {h:.3f}", size=14, color=INK)

            band_area = np.pi * ((r0 + h) ** 2 - (r0 - h) ** 2)
            one_sided = np.pi * ((r0 + h) ** 2 - r0 ** 2) / h
            sticky(ax, 8.3, 0.7, 4.1, 5.0, [
                ("nudge r both ways:", INK, 14),
                ("$\\frac{A(r+h) - A(r-h)}{2h}$", INK, 19),
                (f"= {band_area / (2 * h):.4f}", BLUE, 17),
                (f"$2\\pi r$ = {2 * np.pi * r0:.4f}   exact!", GREEN, 14),
                (f"one-sided: {one_sided:.4f}", RED, 13),
                ("(off by $\\pi h$)", RED, 12)], angle=-1.5)

    save_gif(fig, draw, total, "circle_area.gif")


# >> 2. my note: the area under x^2 grows at a rate of x^2

def parabola_area_gif():
    x0 = 1.5
    A = lambda x: x ** 3 / 3  # area under t^2 from 0 to x
    which, total = phases(sweep=26, sliver=24, forward=50, centered=54, hold=30)
    h_start, h_end = 0.45, 0.012
    trace_h = np.exp(np.linspace(np.log(h_start), np.log(h_end), 120))

    fig, ov = notebook()
    ax = fig.add_axes([0.03, 0.08, 0.58, 0.76])
    tr = fig.add_axes([0.68, 0.2, 0.29, 0.52])
    xs = np.linspace(-0.62, 2.2, 300)

    def axes_by_hand():
        arrow(ax, (-0.7, 0), (2.35, 0), width=1.6, head=12)
        arrow(ax, (0, -0.15), (0, 5.0), width=1.6, head=12)
        write(ax, 2.3, -0.4, "x", size=15)
        write(ax, -0.2, 4.85, "y", size=15)
        pen(ax.plot(xs, xs ** 2, color=BLUE, linewidth=2.6, zorder=5)[0])
        write(ax, 1.75, 4.5, "$y = x^2$", color=BLUE, size=17)

    def trace_panel(f_upto, c_upto):
        blank(tr, (h_start * 1.04, 0), (2.12, 3.08))
        tr.set_facecolor("#ffffffaa")
        tr.axis("on")
        for side in ("top", "right"):
            tr.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            tr.spines[side].set_color(INK)
        tr.tick_params(colors=PENCIL, labelsize=9)
        tr.axhline(x0 ** 2, color=INK, linestyle="--", linewidth=1)
        tr.text(h_start, x0 ** 2 - 0.06, "  true $x^2$ = 2.25", fontsize=11, color=INK)
        tr.set_xlabel("h  (shrinking  →)", fontsize=11, color=INK, family=HAND)
        tr.set_title("the estimate, as h shrinks", fontsize=13, color=INK, family=HAND)
        if f_upto:
            hh = trace_h[:f_upto]
            tr.plot(hh, (A(x0 + hh) - A(x0)) / hh, color=RED, linewidth=2.2)
            tr.plot(hh[-1], (A(x0 + hh[-1]) - A(x0)) / hh[-1], "o", color=RED, markersize=6)
            tr.text(0.2, 2.9, "one-sided", color=RED, fontsize=12, family=HAND)
        if c_upto:
            hh = trace_h[:c_upto]
            tr.plot(hh, (A(x0 + hh) - A(x0 - hh)) / (2 * hh), color="#e8a800", linewidth=2.6)
            tr.plot(hh[-1], (A(x0 + hh[-1]) - A(x0 - hh[-1])) / (2 * hh[-1]), "o", color="#e8a800", markersize=6)
            tr.text(0.32, 2.36, "centered", color="#e8a800", fontsize=12, family=HAND)

    def draw(f):
        phase, t = which(f)
        blank(ax, (-0.75, 2.45), (-0.55, 5.1))
        heading(ov, "The area under $x^2$ grows at a rate of $x^2$", "my note, animated")
        axes_by_hand()

        sweep_to = x0 * ease(t * 1.2) if phase == "sweep" else x0
        fill = np.linspace(0, sweep_to, 100)
        ax.fill_between(fill, 0, fill ** 2, facecolor="none", edgecolor=GREEN, hatch="///", linewidth=0, zorder=2)
        if phase == "sweep":
            arrow(ax, (0.45, 2.7), (0.95, 0.45), color=PINK, rad=-0.2)
            write(ax, 0.1, 2.8, "A(x)", color=PINK, size=18)
            write(ax, 0.6, 3.3, "?", color=PINK, size=22)
            trace_panel(0, 0)
            return

        if phase == "sliver":
            dx = 0.45
            s = ease(t * 1.5)
            ax.add_patch(Rectangle((x0, 0), dx, x0 ** 2 * s, color=GREEN, alpha=0.85, zorder=3))
            if t > 0.5:
                top = np.linspace(x0, x0 + dx, 30)
                ax.fill_between(top, x0 ** 2, top ** 2, color=RED, alpha=0.85, zorder=3)
                arrow(ax, (x0 + dx + 0.12, 0), (x0 + dx + 0.12, x0 ** 2), color=INK, width=1.3, head=10, style="<|-|>")
                write(ax, x0 + dx + 0.17, x0 ** 2 / 2, "$x^2$", size=16)
                write(ax, x0 + 0.1, -0.45, "$\\delta x$", color=GREEN, size=17)
                arrow(ax, (0.55, 3.9), (x0 + 0.2, 1.4), color=PINK, rad=-0.25)
                write(ax, 0.15, 3.95, "$\\delta A$", color=PINK, size=20)
                sticky(ax, -0.7, 2.35, 1.55, 1.25, [
                    ("$\\delta A \\approx x^2 \\cdot \\delta x$", INK, 16),
                    ("so  $\\dfrac{\\delta A}{\\delta x} \\approx x^2$", PINK, 16)], angle=-2)
            trace_panel(0, 0)
            return

        if phase == "forward":
            k = max(1, int(ease(t) * (len(trace_h) - 1)) + 1)
            h = trace_h[k - 1]
            ax.add_patch(Rectangle((x0, 0), h, x0 ** 2, color=GREEN, alpha=0.85, zorder=3))
            top = np.linspace(x0, x0 + h, 30)
            ax.fill_between(top, x0 ** 2, top ** 2, color=RED, alpha=0.9, zorder=3)
            rate = (A(x0 + h) - A(x0)) / h
            sticky(ax, -0.72, 2.15, 1.75, 1.75, [
                ("one-sided:  push right only", RED, 12),
                (f"h = {h:.3f}", INK, 14),
                (f"$\\delta A / \\delta x$ = {rate:.4f}", INK, 14),
                (f"error = {rate - x0 ** 2:.4f}  ($\\approx x h$)", RED, 12)], angle=-2)
            trace_panel(k, 0)
            return

        # centered: one sliver on each side of x
        k = max(1, int(ease(t) * (len(trace_h) - 1)) + 1) if phase == "centered" else len(trace_h)
        h = trace_h[k - 1]
        ax.add_patch(Rectangle((x0 - h, 0), 2 * h, x0 ** 2, color=HIGHLIGHT, alpha=0.9, zorder=3))
        right = np.linspace(x0, x0 + h, 30)
        left = np.linspace(x0 - h, x0, 30)
        ax.fill_between(right, x0 ** 2, right ** 2, color=RED, alpha=0.9, zorder=4)
        ax.fill_between(left, left ** 2, x0 ** 2, color=PURPLE, alpha=0.75, zorder=4)
        pen(ax.plot([x0, x0], [0, x0 ** 2], color=INK, linestyle=":", linewidth=1, zorder=5)[0])
        if h > 0.12:
            arrow(ax, (x0 + 0.55, 3.6), (x0 + h * 0.6, x0 ** 2 + h * 1.1), color=RED, rad=0.2, width=1.4)
            write(ax, x0 + 0.35, 3.7, "missed", color=RED, size=13)
            arrow(ax, (x0 - 0.75, 3.2), (x0 - h * 0.5, x0 ** 2 - h * 0.6), color=PURPLE, rad=-0.2, width=1.4)
            write(ax, x0 - 1.2, 3.3, "counted extra", color=PURPLE, size=13)
        rate = (A(x0 + h) - A(x0 - h)) / (2 * h)
        lines = [("centered:  push both ways", "#b8860b", 12),
                 (f"h = {h:.3f}", INK, 14),
                 (f"$\\delta A / 2h$ = {rate:.5f}", INK, 14),
                 (f"error = {rate - x0 ** 2:.5f}  ($= h^2/3$)", GREEN, 12)]
        if phase == "hold":
            lines = [("missed and extra", INK, 13), ("almost cancel:", INK, 13),
                     ("one-sided error $\\sim h$", RED, 13), ("centered error $\\sim h^2$", GREEN, 13)]
        sticky(ax, -0.72, 2.15, 1.75, 1.75, lines, angle=-2)
        trace_panel(len(trace_h), k)

    save_gif(fig, draw, total, "area_under_x2.gif")


# >> 3. x^2 is a square: why the centered nudge leaves no h^2 behind

def square_gif():
    x, h = 3.0, 0.6
    which, total = phases(intro=14, grow=42, reset=12, centered=30, fly=26, balance=30, hold=34)
    bench = np.array([4.8, 0.0])  # where the strips get laid out side by side

    fig, ov = notebook()
    ax = fig.add_axes([0.02, 0.05, 0.96, 0.78])

    def rect(x0, y0, w, hgt):
        return np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + hgt], [x0, y0 + hgt]])

    def tile(corners, color, alpha=0.9, z=3, outline=None):
        ax.add_patch(pen(Polygon(corners, closed=True, facecolor=color, alpha=alpha,
                                 edgecolor=outline or color, linewidth=1.4, zorder=z), 0.6))

    def side_labels(size):
        write(ax, size / 2 - 0.1, -0.42, "x", size=16)
        write(ax, -0.38, size / 2 - 0.1, "x", size=16)

    def draw(f):
        phase, t = which(f)
        blank(ax, (-0.8, 11.6), (-1.25, 4.6), equal=True)
        heading(ov, "$x^2$ is the area of a square", "so nudge the side and watch the area change")

        if phase in ("intro", "grow"):
            s = ease(t * 1.4) if phase == "intro" else 1
            tile(rect(0, 0, x * s, x * s), BLUE, alpha=0.55)
            write(ax, x * s / 2 - 0.25, x * s / 2 - 0.15, "$x^2$", size=22)
            side_labels(x * s)
        if phase == "grow":
            a = ease(t * 3)  # right strip slides in
            b = ease(t * 3 - 1)    # top strip slides down
            c = ease(t * 3 - 2) # the corner pops in
            tile(rect(x + 3 * (1 - a), 0, h, x), GREEN)
            tile(rect(0, x + 2 * (1 - b), x, h), GREEN)
            if c > 0:
                q = h * c
                tile(rect(x + (h - q) / 2, x + (h - q) / 2, q, q), PINK, alpha=0.95, z=4)
            write(ax, x + h / 2 - 0.12, -0.42, "h", size=15, color=GREEN)
            write(ax, x + 0.05, x / 2, "$xh$", size=16, color="white")
            write(ax, x / 2 - 0.2, x + 0.1, "$xh$", size=16, color="white")
            if c > 0.9:
                ax.add_patch(pen(Ellipse((x + h / 2, x + h / 2), 1.0, 1.0, fill=False, color=PINK, linewidth=2), 2))
                arrow(ax, (x + 1.9, x + 1.5), (x + h + 0.15, x + h + 0.1), color=PINK, rad=0.3)
                write(ax, x + 1.95, x + 1.4, "$h^2$: the error!", color=PINK, size=17)
            write(ax, 4.6, 2.1, "$(x+h)^2 - x^2 = 2xh + h^2$", size=19, color=INK)
            if c > 0.5:
                write(ax, 4.6, 1.2, "$\\div\\, h$:   slope $\\approx 2x + h$", size=19, color=PINK)
                write(ax, 4.6, 0.5, "one-sided: always off by h", size=14, color=PENCIL)

        if phase == "reset":
            tile(rect(0, 0, x, x), BLUE, alpha=0.55 * (1 - ease(t)) + 0.25)
            write(ax, 4.6, 2.1, "now push both ways: $x + h$ and $x - h$", size=18, color=INK, alpha=ease(t * 2))

        if phase in ("centered", "fly", "balance", "hold"):
            ax.add_patch(pen(Rectangle((0, 0), x, x, fill=False, edgecolor=INK, linestyle="--", linewidth=1.2, zorder=6)))
            inner = x - h
            tile(rect(0, 0, inner, inner), BLUE, alpha=0.55)
            write(ax, inner / 2 - 0.55, inner / 2 - 0.1, "$(x-h)^2$", size=18)
            side_labels(x)

            # the two strips that make up (x+h)^2 - (x-h)^2
            long_home = rect(inner, 0, 2 * h, x + h)  # vertical: 2h wide, x+h tall
            short_home = rect(0, inner, inner, 2 * h)	# horizontal: x-h long, 2h tall
            long_bench = np.array([[bench[0], 2.5], [bench[0], 2.5 + 2 * h],
                                   [bench[0] + x + h, 2.5 + 2 * h], [bench[0] + x + h, 2.5]])
            short_bench = rect(bench[0], 0.7, inner, 2 * h)

            if phase == "centered":
                g = ease(t * 1.3)
                tile(lerp(rect(inner, 0, 2 * h, 0), long_home, g), HIGHLIGHT, alpha=0.95)
                tile(lerp(rect(0, inner, 0, 2 * h), short_home, g), HIGHLIGHT, alpha=0.95)
                if t > 0.6:
                    write(ax, x + h + 0.1, (x + h) / 2, "$x+h$", size=15, color="#b8860b")
                    write(ax, inner / 2 - 0.3, x + h + 0.1, "$x-h$", size=15, color="#b8860b")
                    write(ax, 4.6, 2.1, "$(x+h)^2 - (x-h)^2$ = these two strips", size=18)
                return

            m = ease(t * 1.2) if phase == "fly" else 1
            long_now = lerp(long_home, long_bench, m)
            short_now = lerp(short_home, short_bench, m)
            if phase == "fly":
                tile(long_now, HIGHLIGHT, alpha=0.95)
                tile(short_now, HIGHLIGHT, alpha=0.95)
                if t > 0.8:
                    write(ax, bench[0] + x + h + 0.15, 2.75, "$x+h$: h too long", size=14, color="#b8860b")
                    write(ax, bench[0] + inner + 0.15, 0.95, "$x-h$: h too short", size=14, color="#b8860b")
                return

            # balance: cut h off the long strip and move it onto the short one
            mv = ease(t * 1.6) if phase == "balance" else 1
            tile(rect(bench[0], 2.5, x, 2 * h), HIGHLIGHT, alpha=0.95)
            tile(short_bench, HIGHLIGHT, alpha=0.95)
            piece = rect(lerp(bench[0] + x, bench[0] + inner, mv), lerp(2.5, 0.7, mv), h, 2 * h)
            tile(piece, PINK, alpha=0.95, z=5)
            pen(ax.plot([bench[0] + x] * 2, [0.4, 3.95], color=INK, linestyle="--", linewidth=1.2)[0])
            write(ax, bench[0] + x - 0.1, 4.05, "x", size=16)
            if mv < 0.3:
                write(ax, bench[0] + x + h + 0.2, 2.85, "cut here, move down", size=13, color=PINK)
            if phase == "hold" or mv > 0.95:
                write(ax, bench[0] + x + 0.3, 1.9, "both strips:\n$2h \\times x$", size=16, color=INK)
                write(ax, bench[0] - 0.1, -0.45, "area $= 4xh$,   $\\div\\, 2h = 2x$", size=20, color=PINK)
                write(ax, bench[0] - 0.1, -1.1, "exactly the slope. no $h^2$ left over.", size=15, color=GREEN)

    save_gif(fig, draw, total, "square_x2.gif")


# >> 4. the gradient: a hiker feeling for uphill, one direction at a time

def gradient_gif():
    f = lambda x, y: 4 - 0.5 * x ** 2 - 1.5 * y ** 2
    grad_f = lambda q: np.array([-q[0], -3 * q[1]])
    p = np.array([2.0, 0.85])
    h_show = 0.38  # big enough to see; the printed numbers use h = 1e-5

    def num_partial(q, i, h=1e-5):
        e = np.zeros(2)
        e[i] = h
        return (f(*(q + e)) - f(*(q - e))) / (2 * h)

    num_grad = np.array([num_partial(p, 0), num_partial(p, 1)])
    true_grad = grad_f(p)
    rel_error = np.max(np.abs(num_grad - true_grad) / np.maximum(np.abs(num_grad), np.abs(true_grad)))

    # a few uphill steps, each one along the gradient
    path = [p.copy()]
    for _ in range(9):
        path.append(path[-1] + 0.22 * grad_f(path[-1]))
    path = np.array(path)

    which, total = phases(intro=16, east=36, north=36, arrow=22, climb=40, hold=30)
    fig, ov = notebook()
    ax = fig.add_axes([0.02, 0.04, 0.6, 0.8])
    side = fig.add_axes([0.63, 0.06, 0.35, 0.78])
    gx, gy = np.meshgrid(np.linspace(-3.2, 3.2, 220), np.linspace(-2.1, 2.1, 220))
    terrain = LinearSegmentedColormap.from_list("terrain", ["#e3eecb", "#f1e6be", "#ebcf9f", "#dca77d", "#c98663"])
    levels = np.linspace(-2, 4, 13)

    def footprint(q, color):
        ax.add_patch(Ellipse(tuple(q), 0.16, 0.26, angle=-15, color=color, zorder=6))

    def tag(q, text, color, dx=0.0, dy=0.0):
        ax.text(q[0] + dx, q[1] + dy, text, fontsize=11, family=HAND, color=color, ha="center", zorder=8,
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor=color, linewidth=1))

    def draw(frame):
        phase, t = which(frame)
        blank(ax, (-3.2, 3.2), (-2.1, 2.1), equal=True)
        heading(ov, "Gradient: which way is uphill, and how steep?", "feel for it one direction at a time")
        ax.contourf(gx, gy, f(gx, gy), levels=levels, cmap=terrain, alpha=0.9, zorder=1)
        lines = ax.contour(gx, gy, f(gx, gy), levels=levels, colors="#9c6b45", linewidths=0.8, alpha=0.7, zorder=2)
        ax.clabel(lines, levels[::2], fontsize=8, fmt="%.0f", colors="#7a5234")
        ax.add_patch(pen(Rectangle((-3.2, -2.1), 6.4, 4.2, fill=False, edgecolor=INK, linewidth=1.4, zorder=3)))

        # compass and summit flag
        arrow(ax, (-2.75, -1.75), (-2.75, -1.2), width=1.4, head=10)
        arrow(ax, (-2.75, -1.75), (-2.2, -1.75), width=1.4, head=10)
        write(ax, -2.83, -1.12, "N", size=11)
        write(ax, -2.12, -1.83, "E", size=11)
        pen(ax.plot([0, 0], [0, 0.42], color=INK, linewidth=1.6, zorder=6)[0])
        ax.add_patch(Polygon([[0, 0.42], [0.28, 0.33], [0, 0.24]], color=PINK, zorder=6))
        write(ax, 0.08, -0.25, "summit", size=11, color="#7a5234")

        # the hiker
        here = path[0] if phase not in ("climb", "hold") else path[min(int(ease(t) * (len(path) - 1)) if phase == "climb" else len(path) - 1, len(path) - 1)]
        ax.plot(*here, "o", color=INK, markersize=11, markeredgecolor="white", markeredgewidth=2, zorder=9)
        if phase == "intro":
            arrow(ax, (p[0] + 0.7, p[1] + 0.85), (p[0] + 0.1, p[1] + 0.1), color=PINK, rad=0.3)
            write(ax, p[0] - 0.1, p[1] + 0.9, "you are here", color=PINK, size=14)

        def probe(i, s, color):
            e = np.zeros(2)
            e[i] = h_show * ease(s * 1.8)
            for sign in (1, -1):
                q = p + sign * e
                footprint(q, color)
                if s > 0.45:
                    tag(q, f"{f(*q):.2f}", color, dx=0.0 if i else 0.0, dy=0.25 if sign > 0 else -0.42)
            pen(ax.plot([p[0] - e[0], p[0] + e[0]], [p[1] - e[1], p[1] + e[1]], color=color, linewidth=1.4,
                        linestyle="--", zorder=5)[0])

        def leg(i, grow, color):
            d = np.zeros(2)
            d[i] = true_grad[i] * 0.32 * grow
            if grow > 0:
                arrow(ax, tuple(p), tuple(p + d), color=color, width=2.6, head=16, zorder=8)

        if phase == "east":
            probe(0, t, RED)
            leg(0, ease((t - 0.55) * 2.5), RED)
        if phase == "north":
            leg(0, 1, RED)
            probe(1, t, BLUE)
            leg(1, ease((t - 0.55) * 2.5), BLUE)
        if phase == "arrow":
            leg(0, 1, RED)
            leg(1, 1, BLUE)
            g = ease(t * 1.4)
            arrow(ax, tuple(p), tuple(p + true_grad * 0.32 * g), color=PINK, width=3.4, head=22, zorder=9)
            if t > 0.6:
                write(ax, p[0] - 1.25, p[1] - 0.45, "$\\nabla f$  steepest uphill", color=PINK, size=15)
        if phase in ("climb", "hold"):
            n = int(ease(t) * (len(path) - 1)) + 1 if phase == "climb" else len(path)
            pen(ax.plot(path[:n, 0], path[:n, 1], color=PINK, linewidth=2, linestyle="--", zorder=7)[0])
            ax.plot(path[:n, 0], path[:n, 1], "o", color=PINK, markersize=4, zorder=7)

        # the code, with the line that is running highlighted
        blank(side, (0, 1), (0, 1))
        code = ["def numerical_gradient(f, x, h=1e-5):",
                "    for i in range(len(x)):",
                "        x[i] += h     # step forward",
                "        x[i] -= 2h    # step back",
                "        grad[i] = (f(+) - f(-)) / 2h",
                "    return grad"]
        active = {"east": [1, 2, 3, 4], "north": [1, 2, 3, 4], "arrow": [5]}.get(phase, [])
        for k, line in enumerate(code):
            y = 0.95 - 0.055 * k
            if k in active:
                side.add_patch(Rectangle((-0.01, y - 0.026), 1.0, 0.05, color=HIGHLIGHT, alpha=0.75, zorder=1))
            side.text(0, y, line, fontsize=9.5, family="monospace", color=INK if k in active else PENCIL,
                      va="center", zorder=2)
        notes = {
            "intro": [("Standing on a hill. Which way", INK, 14), ("is up, and how steep?", INK, 14)],
            "east": [("i = 0: step east and west", RED, 15), ("only x changes",  PENCIL, 12),
                     (f"$\\partial f / \\partial x \\approx$ {num_grad[0]:.3f}", RED, 17),
                     ("(downhill to the east)", PENCIL, 12)],
            "north": [("i = 1: step north and south", BLUE, 15), ("only y changes", PENCIL, 12),
                      (f"$\\partial f / \\partial y \\approx$ {num_grad[1]:.3f}", BLUE, 17),
                      ("(steeper this way)", PENCIL, 12)],
            "arrow": [("put the two together:", INK, 14),
                      (f"$\\nabla f$ = ({num_grad[0]:.3f}, {num_grad[1]:.3f})", PINK, 17),
                      (f"analytical $(-x, -3y)$ agrees", GREEN, 13),
                      (f"relative error {rel_error:.0e}", GREEN, 13)],
            "climb": [("follow it uphill, step by step", PINK, 14),
                      ("each step = a fresh gradient", PENCIL, 12)],
            "hold": [("flip the sign and walk downhill:", INK, 14),
                     ("that's gradient descent,", PINK, 15), ("that's training a model", PINK, 15)],
        }[phase]
        sticky(side, 0.02, 0.06, 0.94, 0.5, notes, angle=-1.5)

    save_gif(fig, draw, total, "gradient_2d.gif")


# >> 5. how small should h be?

def error_vs_h_png():
    x, true = 1.0, np.cos(1.0)  # f = sin, so f'(1) = cos(1)
    hs = np.logspace(-13, -0.3, 300)
    forward = np.abs((np.sin(x + hs) - np.sin(x)) / hs - true)
    centered = np.abs((np.sin(x + hs) - np.sin(x - hs)) / (2 * hs) - true)
    best = hs[np.argmin(centered[hs > 1e-7]) + np.sum(hs <= 1e-7)]

    fig, ov = notebook(10.5, 6.2)
    blank(ov, (0, 1), (0, 1))
    ov.text(0.5, 0.94, "How small should h be?  Not too small.", ha="center", fontsize=19, fontweight="bold")
    ov.text(0.5, 0.885, "error of the estimated derivative of sin at x = 1", ha="center", fontsize=13,
            color=PENCIL, family=HAND)
    ax = fig.add_axes([0.1, 0.12, 0.85, 0.71])
    ax.set_facecolor("#ffffffb0")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
    ax.tick_params(colors=INK)
    ax.axvspan(1e-13, 1e-8, color=PURPLE, alpha=0.07)
    ax.axvspan(1e-3, 10 ** -0.3, color=RED, alpha=0.07)
    ax.loglog(hs, forward, color=RED, linewidth=2, label="one-sided  $(f(x+h) - f(x))\\,/\\,h$")
    ax.loglog(hs, centered, color="#e8a800", linewidth=2.4, label="centered  $(f(x+h) - f(x-h))\\,/\\,2h$")
    ax.axvline(1e-5, color=GREEN, linewidth=1.4, linestyle="--")
    ax.plot(best, centered[np.searchsorted(hs, best)], "*", color=GREEN, markersize=18, zorder=5)
    write(ax, 1.4e-5, 2e-2, "h = 1e-5\nthe default in\nnumerical_gradient", color=GREEN, size=13)
    write(ax, 2e-13, 3e-2, "too small:\nround-off noise\nwins  ($\\propto \\epsilon/h$)", color=PURPLE, size=14)
    write(ax, 1.3e-3, 3e-12, "too big:\nthe curve bends\naway from the\nsecant", color=RED, size=14)
    write(ax, 4e-3, 6e-3, "slope 1", color=RED, size=13)
    write(ax, 4e-3, 6e-7, "slope 2", color="#b8860b", size=13)
    ax.set_xlabel("step size h", fontsize=13, family=HAND)
    ax.set_ylabel("|estimate - true derivative|", fontsize=13, family=HAND)
    ax.grid(True, which="major", color=GRID, linewidth=0.7)
    ax.legend(loc="upper left", bbox_to_anchor=(0.25, 0.99), facecolor="white", edgecolor=GRID, fontsize=11)
    fig.savefig(FIGURES / "error_vs_h.png", dpi=150, facecolor=PAPER)
    plt.close(fig)


if __name__ == "__main__":
    FIGURES.mkdir(exist_ok=True)
    for make in (circle_area_gif, parabola_area_gif, square_gif, gradient_gif, error_vs_h_png):
        make()
        print(f"done: {make.__name__}")
