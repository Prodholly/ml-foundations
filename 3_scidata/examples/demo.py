"""Demo: this shows how to use profile() in your own code, without the CLI."""

from pathlib import Path

import pandas as pd
from data_explorer.profiler import profile

DATA_PATH = Path(__file__).parent.parent / "data" / "data.csv"

df = pd.read_csv(DATA_PATH)
rows = profile(df)

print(">>> One dict per column <<<")
print(f"{df.shape[1]} columns in the data, {len(rows)} dicts from profile()")
print(rows[1])

print(">>> Group the columns by type <<<")
for dtype in ["number", "text", "date", "empty"]:
    print(f"{dtype:>6}: {[row['column'] for row in rows if row['dtype'] == dtype]}")

print(">>> Find the columns with the most missing values <<<")
for row in sorted(rows, key=lambda row: row["null_pct"], reverse=True)[:3]:
    print(f"{row['column']}: {row['null_pct']:.1f}% missing")

print(">>> Find columns that carry no information <<<")
# one unique value (or none) can't help a model tell samples apart
print([row["column"] for row in rows if row["unique"] <= 1])
