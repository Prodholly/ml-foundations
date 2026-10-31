# ml-foundations

I am doing this from-scratch implementation of core ML engineering primitives in pure Python to refresh and keep sharp my understanding of ML/AI.

Things are moving really fast in AI, and I thought to slow down and recheck if I am losing out and accummulating cognitive atrophy.

I built this to understand what frameworks like PyTorch do under the hood, one component at a time.

Note: everything I do in this repo is taken as though I was a complete beginner; heck, it might help an actual beginner who is interested.

## The days

This is a 115-day ML engineering curriculum. Each day has its own folder with the code, tests, a runnable example, a README, and a research log explaining why I built it the way I did.

| Day | Build | What it covers | Notes |
|---|---|---|---|
| 1 | [DataLoader](1_dataloader/) | Batching, shuffling and collation with a Python generator | [Research log](1_dataloader/research-log/2026-10-21.md) |
| 2 | [Linear Algebra for ML](2_linalg/) | Matmul, einsum, softmax, cross-entropy and a speed benchmark | [Research log](2_linalg/research-log/2026-10-25.md) |
| 3 | [Scientific Data Explorer](3_scidata/) | A CLI tool that profiles any CSV or TSV file | [Research log](3_scidata/research-log/2026-10-31.md) |

Upcoming builds include a training loop, autograd engine, attention mechanism, and transformer from scratch.

## Getting started

Requires Python 3.10 or higher.

```bash
git clone https://github.com/Prodholly/ml-foundations.git
cd ml-foundations
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

That sets up every day at once and runs all the tests. To run one day's code, follow the README in its folder.

## Author

Emmanuel Adeloju — [github.com/Prodholly](https://github.com/Prodholly)

## License

MIT
