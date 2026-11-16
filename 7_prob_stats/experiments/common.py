"""
shared setup for the benchmark and the experiments: paths, fixed constants, seeds, logging, data, plotting
"""



from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from prob_stats import sigmoid

# --- paths -----------
HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

# --- global experiment constants ------
# one master seed; every per-run seed is derived from it deterministically
# so the whole suite is reproducible from this single integer
MASTER_SEED = 20261114

# learning rate and iteration budget held FIXED across every run, so
# that nothing in any result can be attributed to optimizer settings
# convergence is verified per run via ``grad_norm`` and logged
FIXED_LR = 0.5
FIXED_MAX_ITER = 20_000
FIXED_GRAD_TOL = 1e-9

# calibration binning, held fixed; ECE is only comparable across models
# evaluated with the same binning (see expected_calibration_error)
N_CALIBRATION_BINS = 10


def ensure_dirs() -> None:
    """create the results directories if they do not exist"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)


# --- provenance ------------------------------------------------------
def environment_record() -> dict:
    """
    python, numpy, platform, git commit and the fixed constants, written as the first line of each log
    """
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=HERE,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"

    return {
        "record_type": "environment",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": commit,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "master_seed": MASTER_SEED,
        "fixed_lr": FIXED_LR,
        "fixed_max_iter": FIXED_MAX_ITER,
        "fixed_grad_tol": FIXED_GRAD_TOL,
        "n_calibration_bins": N_CALIBRATION_BINS,
    }


# --- structured logging ------
class RunLogger:
    """
    append-only JSONL log, one row per run
    JSONL because different experiments log different fields, which a CSV handles badly
    """

    def __init__(self, path: Path, header: dict | None = None) -> None:
        ensure_dirs()
        self.path = path
        self.rows: list[dict] = []
        # truncate on open: a log that mixes two runs of the same
        # experiment is worse than no log
        self.path.write_text("")
        if header is not None:
            self._write(header)

    def _write(self, record: dict) -> None:
        with self.path.open("a") as fh:
            fh.write(json.dumps(record, default=float) + "\n")

    def log(self, **fields) -> None:
        """append one run record and keep it in memory for plotting"""
        self.rows.append(fields)
        self._write({"record_type": "run", **fields})

    def column(self, name: str) -> np.ndarray:
        """all values of one field across logged runs; shape: (n_runs,)"""
        return np.array([r[name] for r in self.rows], dtype=np.float64)

    def where(self, **conditions) -> list[dict]:
        """rows matching every given field value"""
        return [
            r
            for r in self.rows
            if all(r.get(k) == v for k, v in conditions.items())
        ]


# --- synthetic data --------------------------------------------------
@dataclass(frozen=True)
class DataConfig:
    """
    recipe for a synthetic classification dataset: n rows, d features, d_signal of them real
    synthetic because the hypotheses need the true weights and the noise level, which real data never gives me
    """

    n: int
    d: int
    d_signal: int
    signal_strength: float = 1.0
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.d_signal <= self.d:
            raise ValueError("require 0 < d_signal <= d")


def make_logistic_data(cfg: DataConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    X ~ N(0, 1) (n, d), w_true with d_signal non-zero entries, y ~ Bernoulli(sigmoid(X w_true))
    labels are sampled, not thresholded, so there's irreducible noise, without it the U-curve couldn't exist
    """
    rng = np.random.default_rng(cfg.seed)

    X = rng.normal(size=(cfg.n, cfg.d))  # shape: (n, d)

    w_true = np.zeros(cfg.d)  # shape: (d,)
    w_true[: cfg.d_signal] = rng.normal(
        scale=cfg.signal_strength, size=cfg.d_signal
    )

    logits = X @ w_true  # shape: (n,)
    p = sigmoid(logits)  # shape: (n,)
    y = (rng.random(cfg.n) < p).astype(np.float64)  # shape: (n,)

    return X, y, w_true


def derive_seed(*parts: int) -> int:
    """
    a run's seed, computed from MASTER_SEED and the run's coordinates
    any run can be rebuilt from the log alone, regardless of execution order
    """
    h = MASTER_SEED
    for part in parts:
        h = (h * 1_000_003 + int(part)) % (2**31 - 1)
    return int(h)


# --- plotting --------------------------------------------------------
def get_pyplot():
    """import matplotlib lazily with the Agg backend, so the tests never pay for it"""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "matplotlib is required for the experiment plots. "
            "Install it with: pip install matplotlib"
        ) from exc


def save_figure(fig, name: str) -> Path:
    """write a figure to the figures directory and report the path"""
    ensure_dirs()
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  wrote {path.relative_to(HERE.parent)}")
    return path
