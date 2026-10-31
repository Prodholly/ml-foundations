# data_explorer.py: the command-line entry point.
# Handles everything around profile(): arguments, loading the file, errors, and the table.
#
# Run from inside 3_scidata:
#   python data_explorer.py data/data.csv
#   python data_explorer.py data/data.tsv --sep "\t"

import argparse
import sys

import pandas as pd
from rich.console import Console
from rich.table import Table

from data_explorer.profiler import profile


def parse_args():
    parser = argparse.ArgumentParser(
        prog="data-explorer",
        description="CLI tool that profiles any CSV/TSV dataset — column types, missing values, and summary statistics",
    )
    parser.add_argument("filepath", help="path to the CSV or TSV file")
    parser.add_argument("--sep", default=",", help='column separator; use "\\t" for TSV files (default: ",")')
    return parser.parse_args()


def exit_with_error(message):
    print(f"error: {message}", file=sys.stderr)
    sys.exit(1)


def load_file(filepath, sep):
    # typing --sep "\t" in the terminal passes a backslash and a t, not a tab
    if sep == "\\t":
        sep = "\t"

    try:
        try:
            df = pd.read_csv(filepath, sep=sep, encoding="utf-8")
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(filepath, sep=sep, encoding="latin-1")
            except UnicodeDecodeError:
                exit_with_error(f"could not decode {filepath} as utf-8 or latin-1")
    except FileNotFoundError:
        exit_with_error(f"file not found: {filepath}")
    except pd.errors.EmptyDataError:
        exit_with_error(f"file is empty: {filepath}")

    # a file with a header but no rows raises no exception
    if df.empty:
        exit_with_error(f"file has no data rows: {filepath}")

    return df


def format_value(value):
    # "is None", not "if not value": a std of 0.0 is falsy but is a real result.
    # std is nan when a column has only one real value, since a sample std needs two
    if value is None or pd.isna(value):
        return "—"

    if isinstance(value, float):
        return f"{value:.2f}"

    return str(value)


def print_table(rows):
    table = Table(title="Data Profile", caption="this shows the output of your data",
                  header_style="bold white on cyan")

    for name in ["Column", "Type", "Null%", "Unique", "Mean", "Std", "Min", "Max"]:
        table.add_column(name, style="cyan")

    keys = ["column", "dtype", "null_pct", "unique", "mean", "std", "min", "max"]
    for row in rows:
        table.add_row(*[format_value(row[key]) for key in keys])

    # column names come from the user's file, so turn off rich's markup and emoji codes;
    # otherwise a header like "[/x]" crashes it and ":smile:" becomes an emoji
    Console(markup=False, emoji=False).print(table)


def main():
    args = parse_args()
    df = load_file(args.filepath, args.sep)
    rows = profile(df)
    print_table(rows)


if __name__ == "__main__":
    main()
