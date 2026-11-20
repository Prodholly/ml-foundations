from pathlib import Path

import torch

from .model import RawMLP, cross_entropy

FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


def walk_graph(loss: torch.Tensor, params: dict[str, torch.Tensor]) -> tuple[dict, list]:
    names = {id(t): n for n, t in params.items()}
    nodes: dict[int, tuple[str, int]] = {}
    edges: list[tuple[int, int]] = []

    frontier = [(loss.grad_fn, 0)]
    while frontier:
        fn, depth = frontier.pop(0)
        key = id(fn)
        if key in nodes:
            nodes[key] = (nodes[key][0], max(nodes[key][1], depth))
            continue

        label = type(fn).__name__
        if hasattr(fn, "variable"):
            v = fn.variable
            label = f"{names.get(id(v), 'leaf')}\n{tuple(v.shape)}"
        nodes[key] = (label, depth)

        for nxt, _ in fn.next_functions:
            if nxt is not None:
                edges.append((id(nxt), key))
                frontier.append((nxt, depth + 1))

    changed = True
    while changed:
        changed = False
        for child, parent in edges:
            if nodes[child][1] < nodes[parent][1] + 1:
                nodes[child] = (nodes[child][0], nodes[parent][1] + 1)
                changed = True
    return nodes, edges


def draw_graph(nodes: dict, edges: list, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    levels: dict[int, list[int]] = {}
    for key, (_, depth) in nodes.items():
        levels.setdefault(depth, []).append(key)
    max_depth = max(levels)
    pos = {}
    for depth, keys in levels.items():
        for i, key in enumerate(sorted(keys, key=lambda k: nodes[k][0])):
            pos[key] = ((i + 1) / (len(keys) + 1), 1 - depth / max(max_depth, 1))

    fig, ax = plt.subplots(figsize=(9, 1.1 * (max_depth + 2)))

    for child, parent in edges:
        (x0, y0), (x1, y1) = pos[child], pos[parent]
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="#52514e", lw=1, shrinkA=14, shrinkB=14))

    for key, (label, depth) in nodes.items():
        is_leaf = "\n" in label
        color = "#cfe3fa" if is_leaf else ("#cdeedf" if depth == 0 else "#ecebe7")
        x, y = pos[key]
        ax.text(x, y, label, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc=color, ec="#52514e", lw=0.8))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.12, 1.12)
    ax.axis("off")
    ax.set_title("autograd graph of cross_entropy(RawMLP([784, 32, 10])(x), y)", fontsize=11)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def render_compute_graph() -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    model = RawMLP([784, 32, 10], seed=0)

    x = torch.randn(16, 784)
    y = torch.randint(0, 10, (16,))

    loss = cross_entropy(model(x), y)

    params = {"W1": model.layers[0].W, "b1": model.layers[0].b, "W2": model.layers[1].W, "b2": model.layers[1].b}

    try:
        from torchviz import make_dot

        dot = make_dot(loss, params=params)
        dot.format = "png"
        dot.render(str(FIG_DIR / "compute_graph"), cleanup=True)
        path = FIG_DIR / "compute_graph.png"
    except Exception as e:
        print(f"torchviz unavailable ({type(e).__name__}), drawing the graph from grad_fn myself")
        nodes, edges = walk_graph(loss, params)
        path = draw_graph(nodes, edges, FIG_DIR / "compute_graph.png")

    print("saved", path)
    return path


if __name__ == "__main__":
    render_compute_graph()
