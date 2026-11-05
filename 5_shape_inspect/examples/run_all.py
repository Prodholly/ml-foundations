"""Runs every Day 5 task and saves the figures to figures/."""

from examples.figures import save_all
from numpy_ops import axis_demo, benchmark_demo, broadcast_demo, reshape_demo, shape_demo

if __name__ == "__main__":
    for title, demo in [
        ("shape inspection", shape_demo),
        ("Reshape & Indexing", reshape_demo),
        ("axis Operations", axis_demo),
        ("Broadcasting", broadcast_demo),
        ("loop vs vectorized", benchmark_demo),
    ]:
        print(f"\n=== {title} ===")
        demo()

    print("\n=== figures ===")
    for name in save_all():
        print(f"saved figures/{name}")
