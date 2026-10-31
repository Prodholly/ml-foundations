"""this functions to generate a 50-row, 12-column table, saved as both CSV and TSV.

The columns are built to exercise every field of a data profile: dtype (number, text, date),
null %, unique count, and mean/std/min/max. Each column covers a different case, including
edge cases a profiler has to handle: a constant column, an all-unique ID, an entirely empty
column, and a column whose missing values are written as "NA" instead of left blank.
"""

import csv
from pathlib import Path

import numpy as np

N_ROWS = 50
N_COLS = 12

# save next to this script, wherever it is run from
DATA_DIR = Path(__file__).parent

# seeded so every run produces the exact same files
rng = np.random.default_rng(42)


def with_nulls(values, n_nulls, marker=""):
    """Blank out exactly n_nulls randomly chosen cells, so each column's null % is exact."""
    values = list(values)
    for row in rng.choice(N_ROWS, size=n_nulls, replace=False):
        values[row] = marker
    return values


columns = {
    # number, ID: 1..50, every value distinct
    "col_1": list(range(1, N_ROWS + 1)),
    # number, integers 18..90, 5 blanks (10% null)
    "col_2": with_nulls(rng.integers(18, 91, size=N_ROWS), 5),
    # number, decimals around 50 with spread 10, 3 blanks (6% null)
    "col_3": with_nulls([f"{v:.2f}" for v in rng.normal(50, 10, size=N_ROWS)], 3),
    # number, integers -100..100 including negatives, no nulls
    "col_4": rng.integers(-100, 101, size=N_ROWS).tolist(),
    # text, 4 categories, 4 blanks (8% null)
    "col_5": with_nulls(rng.choice(["red", "green", "blue", "yellow"], size=N_ROWS), 4),
    # date, YYYY-MM-DD during 2025, 2 blanks (4% null)
    "col_6": with_nulls((np.datetime64("2025-01-01") + rng.integers(0, 365, size=N_ROWS)).astype(str), 2),
    # number, large decimals 1,000..100,000, no nulls
    "col_7": [f"{v:.2f}" for v in rng.uniform(1_000, 100_000, size=N_ROWS)],
    # text, product-style codes drawn from 30 possibilities, so some repeat; no nulls
    "col_8": [f"P{v:03d}" for v in rng.integers(1, 31, size=N_ROWS)],
    # number, constant: every value is 7, so std is 0 and unique count is 1
    "col_9": [7] * N_ROWS,
    # number, 0/1 flag with 10 missing cells written as "NA" (20% null if "NA" is treated as missing)
    "col_10": with_nulls(rng.integers(0, 2, size=N_ROWS), 10, marker="NA"),
    # text, ordered levels, 15 blanks (30% null)
    "col_11": with_nulls(rng.choice(["low", "medium", "high"], size=N_ROWS), 15),
    # empty: every cell blank (100% null), so no dtype or stats can be inferred
    "col_12": [""] * N_ROWS,
}

assert len(columns) == N_COLS

header = list(columns)
rows = list(zip(*columns.values()))

# same data, two delimiters
for name, delimiter in [("data.csv", ","), ("data.tsv", "\t")]:
    with open(DATA_DIR / name, "w", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter)
        writer.writerow(header)
        writer.writerows(rows)

print(f"wrote {N_ROWS} rows x {N_COLS} columns to {DATA_DIR / 'data.csv'} and {DATA_DIR / 'data.tsv'}")
