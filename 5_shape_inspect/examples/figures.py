"""Draws the Day 5 figures: the reshape diagram, the broadcast diagram and the reshape GIF."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save to files, no window

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# >> style

BG = "#fbfaf7"
INK = "#1f2933"
MUTED = "#8a94a3"
GOLD = "#e8b007"
ROW_HUES = ["#3b6fb6", "#d9692b", "#2f9e72"]  # one hue per row of the (3, 4) grid
A_HUE, B_HUE, C_HUE, OUT_HUE = "#3b6fb6", "#d9692b", "#8a5cc2", "#2f9e72"
MONO = "DejaVu Sans Mono"

plt.rcParams.update({"font.family": "DejaVu Sans", "text.color": INK})


def lighten(color, t):
    # mix a color with white: t=0 keeps it, t=1 gives white
    r, g, b = to_rgb(color)
    return (r + (1 - r) * t, g + (1 - g) * t, b + (1 - b) * t)


def element_color(i):
    # each row has its own hue, getting lighter along the columns
    row, col = divmod(i, 4)
    return lighten(ROW_HUES[row], 0.08 * col)


def cell(ax, x, y, text, face, sub=None, ghost=False, ring=None, size=1.0, fontsize=14):
    pad = 0.05
    ax.add_patch(FancyBboxPatch(
        (x + pad, y + pad), size - 2 * pad, size - 2 * pad,
        boxstyle="round,pad=0,rounding_size=0.12",
        facecolor=lighten(face, 0.78) if ghost else face,
        edgecolor=face if ghost else "white",
        linewidth=1.4, linestyle=(0, (3, 2)) if ghost else "-",
        zorder=2,
    ))
    if ring:
        ax.add_patch(FancyBboxPatch(
            (x + pad / 2, y + pad / 2), size - pad, size - pad,
            boxstyle="round,pad=0,rounding_size=0.14",
            facecolor="none", edgecolor=ring, linewidth=3.2, zorder=3,
        ))
    color = face if ghost else "white"
    ty = y + size * (0.56 if sub else 0.5)
    ax.text(x + size / 2, ty, text, ha="center", va="center", fontsize=fontsize,
            fontweight="bold", color=color, zorder=4)
    if sub:
        ax.text(x + size / 2, y + size * 0.2, sub, ha="center", va="center", fontsize=8,
                family=MONO, color=color, alpha=0.85, zorder=4)


def blank_slot(ax, x, y):
    ax.add_patch(FancyBboxPatch(
        (x + 0.05, y + 0.05), 0.9, 0.9, boxstyle="round,pad=0,rounding_size=0.12",
        facecolor="none", edgecolor="#d5d9df", linewidth=1.2, linestyle=(0, (3, 2)), zorder=1,
    ))


def bracket(ax, x0, x1, y, color, label):
    # a square bracket under a run of cells, with a label below it
    ax.plot([x0 + 0.1, x0 + 0.1, x1 - 0.1, x1 - 0.1], [y + 0.15, y, y, y + 0.15],
            color=color, linewidth=2, solid_capstyle="round")
    ax.text((x0 + x1) / 2, y - 0.3, label, ha="center", va="center", fontsize=10,
            family=MONO, color=color, fontweight="bold")


def new_figure(width, height, xlim, ylim):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def save(fig, name):
    fig.savefig(FIGURES / name, dpi=160, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# >> shared pieces of the reshape picture

STRIP_Y = 6.0  # the (12,) vector sits on this row


def grid_xy(i, x0, top):
    # where element i lands in a (3, 4) grid whose top-left corner is (x0, top)
    row, col = divmod(i, 4)  # row-major: i = row * 4 + col
    return x0 + col, top - 1 - row


def draw_strip(ax, highlight=None, ghosts=()):
    for i in range(12):
        ax.text(i + 0.5, STRIP_Y + 1.25, str(i), ha="center", fontsize=9, family=MONO, color=MUTED)
        if i in ghosts:
            blank_slot(ax, i, STRIP_Y)
        else:
            cell(ax, i, STRIP_Y, str(i), element_color(i), ring=GOLD if i == highlight else None)
    ax.text(-0.35, STRIP_Y + 1.25, "i", ha="right", fontsize=9, family=MONO, color=MUTED)
    ax.text(-0.35, STRIP_Y + 0.5, "x", ha="right", va="center", fontsize=15, fontweight="bold")
    ax.text(12.2, STRIP_Y + 0.5, "(12,)", ha="left", va="center", fontsize=13, family=MONO)


def draw_grid_frame(ax, x0, top):
    # row and column indices, axis names and the shape label around a (3, 4) grid
    for r in range(3):
        ax.text(x0 - 0.3, top - 0.5 - r, str(r), ha="center", va="center", fontsize=10,
                family=MONO, color=ROW_HUES[r], fontweight="bold")
    for c in range(4):
        ax.text(x0 + c + 0.5, top - 3.3, str(c), ha="center", va="center", fontsize=10,
                family=MONO, color=MUTED)
    ax.text(x0 - 0.85, top - 1.5, "axis 0  (rows)", rotation=90, ha="center", va="center",
            fontsize=10, color=MUTED)
    ax.text(x0 + 2, top - 3.75, "axis 1  (columns)", ha="center", va="center", fontsize=10, color=MUTED)
    ax.text(x0 + 4.2, top - 1.5, "(3, 4)", ha="left", va="center", fontsize=13, family=MONO)


# >> figure 1: reshape diagram

def save_reshape_diagram() -> None:
    fig, ax = new_figure(13, 7.4, (-1.6, 13.4), (-1.3, 7.8))
    x0, top = 0.6, 3.6

    draw_strip(ax, highlight=11)
    for r in range(3):
        bracket(ax, 4 * r, 4 * r + 4, STRIP_Y - 0.25, ROW_HUES[r], f"x[{4 * r}:{4 * r + 4}]")

    for i in range(12):
        gx, gy = grid_xy(i, x0, top)
        cell(ax, gx, gy, str(i), element_color(i), sub=f"[{i // 4},{i % 4}]", ring=GOLD if i == 11 else None)
    draw_grid_frame(ax, x0, top)

    # one arrow for the reshape: the row colors already show where each run of 4 lands
    ax.add_patch(FancyArrowPatch((x0 + 2, 5.0), (x0 + 2, 3.85), arrowstyle="-|>", mutation_scale=22,
                                 linewidth=2, color=INK))
    ax.text(x0 + 2.35, 4.42, ".reshape(3, 4)", va="center", fontsize=12, family=MONO)

    # the rule, written out next to the grid
    bx, by = 6.9, 0.25
    ax.add_patch(FancyBboxPatch((bx, by), 5.9, 3.35, boxstyle="round,pad=0.1,rounding_size=0.25",
                                facecolor="white", edgecolor="#e3e6ea", linewidth=1.2))
    ax.text(bx + 0.3, by + 2.85, "Row-major (C order)", fontsize=12, fontweight="bold")
    ax.text(bx + 0.3, by + 2.3, "the last axis changes fastest", fontsize=10, color=MUTED)
    ax.text(bx + 0.3, by + 1.55, "x[i]  →  X[i // 4, i % 4]", fontsize=12, family=MONO)
    ax.text(bx + 0.3, by + 0.75, "x[11] →  X[2, 3]", fontsize=12, family=MONO, color="#a67c00",
            fontweight="bold")
    ax.text(bx + 0.3, by + 0.3, "same element, same memory", fontsize=9.5, color=MUTED)

    fig.suptitle("x.reshape(3, 4): the data stays put, only the indexing changes",
                 fontsize=16, fontweight="bold", y=0.95)
    save(fig, "reshape_diagram.png")


# >> figure 2: broadcast diagram

def shape_table(ax, x, y, rows):
    # rows: (name, dims, color, stretched); dims are right-aligned, None means "missing"
    ax.text(x + 1.6, y + 0.65, "axis 0", ha="center", fontsize=8.5, color=MUTED)
    ax.text(x + 2.6, y + 0.65, "axis 1", ha="center", fontsize=8.5, color=MUTED)
    for k, (name, dims, color, stretched) in enumerate(rows):
        yy = y - 0.6 * k
        if name == "out":
            ax.plot([x, x + 3.1], [yy + 0.32, yy + 0.32], color=INK, linewidth=1)
        ax.text(x, yy, name, fontsize=12, family=MONO, fontweight="bold", color=color, va="center")
        for d, value in enumerate(dims):
            hit = d in stretched
            label = "–" if value is None else str(value)
            ax.text(x + 1.6 + d, yy, label, ha="center", va="center", fontsize=12, family=MONO,
                    color=B_HUE if hit else INK, fontweight="bold" if hit else "normal")


def broadcast_row(ax, y, left, left_ghost_rows, left_ghost_cols, right, right_ghost_rows,
                  right_ghost_cols, out, hues, names):
    # draws  left + right = out  as three (3, 4) grids, ghost cells are the stretched copies
    positions = [0, 5.4, 10.8]
    for (grid, ghost_rows, ghost_cols, hue, name, gx) in [
        (left, left_ghost_rows, left_ghost_cols, hues[0], names[0], positions[0]),
        (right, right_ghost_rows, right_ghost_cols, hues[1], names[1], positions[1]),
        (out, (), (), hues[2], names[2], positions[2]),
    ]:
        for r in range(3):
            for c in range(4):
                ghost = r in ghost_rows or c in ghost_cols
                cell(ax, gx + c, y + 2 - r, str(grid[r, c]), hue, ghost=ghost, fontsize=13)
        ax.text(gx + 2, y + 3.35, name, ha="center", fontsize=12, family=MONO, color=hue, fontweight="bold")
    ax.text(4.7, y + 1.5, "+", ha="center", va="center", fontsize=26, color=INK)
    ax.text(10.1, y + 1.5, "=", ha="center", va="center", fontsize=26, color=INK)


def save_broadcast_diagram() -> None:
    fig, ax = new_figure(15, 9.6, (-6.4, 15.2), (-7.7, 6.0))

    # top: bias addition, (3, 4) + (4,)
    A = np.arange(12).reshape(3, 4)   # shape: (3, 4)
    b = np.array([10, 20, 30, 40])    # shape: (4,)
    out1 = A + b                      # shape: (3, 4)
    broadcast_row(ax, 2.0, A, (), (), np.broadcast_to(b, (3, 4)), (1, 2), (), out1,
                  (A_HUE, B_HUE, OUT_HUE), ("A  (3, 4)", "b  (4,)", "A + b  (3, 4)"))
    shape_table(ax, -6.0, 4.3, [
        ("A", (3, 4), A_HUE, ()),
        ("b", (None, 4), B_HUE, (0,)),
        ("out", (3, 4), OUT_HUE, ()),
    ])
    ax.text(-6.0, 1.75, "– b has no axis 0, so it\ncounts as 1 and is\ncopied down 3 rows",
            fontsize=10, color=MUTED, va="top", linespacing=1.4)

    # bottom: outer add, (1, 4) + (3, 1), both stretch
    r = np.arange(4).reshape(1, 4)        # shape: (1, 4)
    c = np.array([[0], [10], [20]])       # shape: (3, 1)
    out2 = r + c                          # shape: (3, 4)
    broadcast_row(ax, -5.0, np.broadcast_to(r, (3, 4)), (1, 2), (), np.broadcast_to(c, (3, 4)), (),
                  (1, 2, 3), out2, (A_HUE, C_HUE, OUT_HUE), ("r  (1, 4)", "c  (3, 1)", "r + c  (3, 4)"))
    shape_table(ax, -6.0, -2.7, [
        ("r", (1, 4), A_HUE, (0,)),
        ("c", (3, 1), C_HUE, (1,)),
        ("out", (3, 4), OUT_HUE, ()),
    ])
    ax.text(-6.0, -5.25, "each size-1 axis\nstretches to match\nthe other array",
            fontsize=10, color=MUTED, va="top", linespacing=1.4)

    ax.plot([-6.0, 15.0], [-0.45, -0.45], color="#e3e6ea", linewidth=1.2)
    fig.suptitle("Broadcasting: align shapes from the right, then stretch the 1s",
                 fontsize=17, fontweight="bold", y=0.93)
    ax.text(4.4, -7.0,
            "Faded cells are never stored. NumPy reuses the same values (stride 0), so the copy costs no memory.\n"
            "Two axes are compatible when they are equal or one of them is 1. Otherwise: ValueError.",
            ha="center", va="center", fontsize=10.5, color=MUTED, linespacing=1.6)
    save(fig, "broadcast_diagram.png")


# >> figure 3: reshape animation

def ease(t):
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)  # smoothstep: slow start, slow finish


def save_reshape_animation() -> None:
    fig, ax = new_figure(10, 7, (-1.6, 13.4), (-1.3, 8.6))
    x0, top = 4.0, 3.6
    intro, move, hold, outro = 8, 16, 6, 24
    frames = intro + 3 * (move + hold) + outro

    def draw_frame(f):
        ax.clear()
        ax.set_xlim(-1.6, 13.4)
        ax.set_ylim(-1.3, 8.6)
        ax.set_aspect("equal")
        ax.axis("off")

        # how far each row has travelled: 0 = still in x, 1 = landed in X
        progress = []
        for r in range(3):
            start = intro + r * (move + hold)
            progress.append((f - start) / move)
        done = f >= intro + 3 * (move + hold)

        moving = {i for i in range(12) if progress[i // 4] > 0}
        draw_strip(ax, ghosts=moving, highlight=11 if done else None)
        for i in range(12):
            blank_slot(ax, *grid_xy(i, x0, top))
        draw_grid_frame(ax, x0, top)

        active = None
        for i in sorted(moving):
            r, c = divmod(i, 4)
            t = ease(progress[r] * 1.25 - 0.08 * c)  # columns leave one after another
            if 0 < progress[r] < 1:
                active = r
            gx, gy = grid_xy(i, x0, top)
            x = i + (gx - i) * t
            y = STRIP_Y + (gy - STRIP_Y) * t + 0.6 * np.sin(np.pi * t)  # a small lift on the way
            cell(ax, x, y, str(i), element_color(i), sub=f"[{r},{c}]" if t == 1 else None,
                 ring=GOLD if done and i == 11 else None)

        if done:
            caption, color = "x[11] == X[2, 3]  →  same element, row-major order", "#a67c00"
        elif active is not None:
            caption, color = f"row {active}  ←  x[{4 * active}:{4 * active + 4}]", ROW_HUES[active]
        else:
            caption, color = "x = np.arange(12)    shape (12,)", INK
        ax.text(6, -0.95, caption, ha="center", va="center", fontsize=13, family=MONO,
                fontweight="bold", color=color)
        ax.set_title("x.reshape(3, 4)", fontsize=16, fontweight="bold", family=MONO, pad=8)

    anim = animation.FuncAnimation(fig, draw_frame, frames=frames)
    anim.save(FIGURES / "reshape_animation.gif", writer=animation.PillowWriter(fps=16), dpi=90,
              savefig_kwargs={"facecolor": BG})
    plt.close(fig)


def save_all() -> list[str]:
    FIGURES.mkdir(exist_ok=True)
    save_reshape_diagram()
    save_broadcast_diagram()
    save_reshape_animation()
    return ["reshape_diagram.png", "broadcast_diagram.png", "reshape_animation.gif"]
