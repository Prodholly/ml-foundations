# profiler.py: turns a DataFrame into one summary dict per column.
# This file only computes. It does not read files, print, or format anything.
# That split means you can test profile() with a small DataFrame made by hand.

import pandas as pd
from pandas.api.types import is_numeric_dtype


# Takes a pandas DataFrame. Returns a list of dicts, one per column.
# Each dict has the keys: column, dtype, null_pct, unique, mean, std, min, max.
def profile(df):
    results = []

    for column_name in df.columns:
        current_column = df[column_name]

        # percent of missing values, from 0 to 100
        num_missing_values = current_column.isna().sum()
        pct_miss_val = float((num_missing_values / len(current_column)) * 100)

        # distinct values, leaving out missing values
        unique_count = current_column.nunique()

        # text, date and empty columns keep these None values
        mean = std = minim = maxi = None

        new_column_dropped = current_column.dropna()

        # an all-blank column loads as float64, so pandas calls it numeric, but its mean
        # is nan, not None; this check must come first
        if new_column_dropped.empty:
            dtype = "empty"
        elif is_numeric_dtype(current_column):
            dtype = "number"
            # pandas skips missing values here; float() stops NumPy types leaking out
            mean = float(current_column.mean())
            std = float(current_column.std())
            minim = float(current_column.min())
            maxi = float(current_column.max())
        else:
            # errors="coerce" turns anything that isn't a date into a missing value, so if
            # none went missing, every value was a date
            parsed = pd.to_datetime(new_column_dropped, errors="coerce", format="%Y-%m-%d")
            dtype = "date" if parsed.notna().all() else "text"

        results.append({
            "column": column_name,
            "dtype": dtype,
            "null_pct": pct_miss_val,
            "unique": unique_count,
            "mean": mean,
            "std": std,
            "min": minim,
            "max": maxi,
        })

    return results
