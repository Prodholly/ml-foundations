# Day 3: Scientific Data Explorer

A command-line tool that profiles any CSV or TSV file. Give it a file and it prints one table row per
column: the column's type, how much of it is missing, how many distinct values it has, and summary
statistics for numeric columns.

Note: like Days 1 and 2, everything here is written as though I was a complete beginner.

## Why this exists

The first thing to do with a new dataset is find out what is actually in it. A model will happily train on
a column that is 90% missing, a constant column, or numbers that loaded as text, and the only sign is bad
results later.

`df.describe()` gets part of the way there, but it skips text columns by default and doesn't report missing
values as a percentage or spot dates. This tool answers the same four questions for every column, in one table.

I also used this day to practise decomposing a project: I planned each file as numbered pseudocode before
writing any code, then built it up one piece at a time.

## Project Structure

```
3_scidata
├── data
│   ├── data.csv             # sample data, 50 rows x 12 columns
│   ├── data.tsv             # the same data, tab-separated
│   └── generate_data.py     # seeded script that creates both files
├── data_explorer
│   ├── __init__.py
│   └── profiler.py          # profile(df): the calculations
├── data_explorer.py         # the CLI: arguments, loading, errors, table
├── examples
│   └── demo.py              # using profile() from your own code
├── research-log
│   └── 2026-10-31.md        # daily build notes
├── tests
│   └── test_profiler.py     # pytest test suite
└── README.md
```

The full reasoning behind this build is in the [research log](research-log/2026-10-31.md).

## Installation

Run this from the repo root. It installs pandas and rich alongside NumPy, plus pytest.

```bash
git clone https://github.com/Prodholly/ml-foundations.git
cd ml-foundations
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quick Start

Run from inside `3_scidata`:

```bash
cd 3_scidata
python data_explorer.py data/data.csv
```

```
                                  Data Profile
┏━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━┳━━━━━━━━━━┓
┃ Column ┃ Type   ┃ Null%  ┃ Unique ┃ Mean     ┃ Std      ┃ Min     ┃ Max      ┃
┡━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━╇━━━━━━━━━━┩
│ col_1  │ number │ 0.00   │ 50     │ 25.50    │ 14.58    │ 1.00    │ 50.00    │
│ col_2  │ number │ 10.00  │ 33     │ 58.04    │ 19.58    │ 22.00   │ 89.00    │
│ col_3  │ number │ 6.00   │ 47     │ 50.51    │ 7.81     │ 33.17   │ 71.42    │
│ col_4  │ number │ 0.00   │ 44     │ -3.00    │ 56.43    │ -96.00  │ 94.00    │
│ col_5  │ text   │ 8.00   │ 4      │ —        │ —        │ —       │ —        │
│ col_6  │ date   │ 4.00   │ 45     │ —        │ —        │ —       │ —        │
│ col_7  │ number │ 0.00   │ 50     │ 51368.20 │ 30190.78 │ 2379.69 │ 99245.18 │
│ col_8  │ text   │ 0.00   │ 23     │ —        │ —        │ —       │ —        │
│ col_9  │ number │ 0.00   │ 1      │ 7.00     │ 0.00     │ 7.00    │ 7.00     │
│ col_10 │ number │ 20.00  │ 2      │ 0.50     │ 0.51     │ 0.00    │ 1.00     │
│ col_11 │ text   │ 30.00  │ 3      │ —        │ —        │ —       │ —        │
│ col_12 │ empty  │ 100.00 │ 0      │ —        │ —        │ —       │ —        │
└────────┴────────┴────────┴────────┴──────────┴──────────┴─────────┴──────────┘
```

For any other separator, pass `--sep`:

```bash
python data_explorer.py data/data.tsv --sep "\t"
python data_explorer.py my_data.csv --sep ";"
```

To use the profiler from your own code, see the demo, also from inside `3_scidata`:

```bash
python -m examples.demo
```

## What each column of the table means

| Field | Meaning |
|---|---|
| Type | `number`, `text`, `date`, or `empty` if the column has no values at all |
| Null% | percentage of values that are missing, from 0 to 100 |
| Unique | number of distinct values, not counting missing ones |
| Mean, Std, Min, Max | statistics for number columns, ignoring missing values; "—" for everything else |

Std is the sample standard deviation, which is what pandas' `.std()` gives. It needs at least two values,
so a column with only one real value shows "—".

## Key behaviors

**Two parts: calculating and displaying**

`profile()` in `profiler.py` only calculates. It takes a DataFrame and returns a list of dicts, with no
files or printing involved, so it can be tested with a small DataFrame made by hand. `data_explorer.py`
handles everything around it, and only the copy going to the screen is rounded.

**Errors are one line, not a traceback**

| Situation | Result |
|---|---|
| File not found | `error: file not found: ...` |
| 0-byte file | `error: file is empty: ...` |
| Header but no rows | `error: file has no data rows: ...` |
| Not valid UTF-8 | retries as latin-1 |

Errors go to stderr and the program exits with code 1, so they stay out of redirected output.

**The sample data is built to break things**

Each column in `data/data.csv` covers a different case: an ID where every value is unique, decimals,
negatives, blanks, dates, a constant column (`col_9`), missing values written as `NA` instead of left
blank (`col_10`), and a completely empty column (`col_12`). The data is seeded, so
`python data/generate_data.py` always produces the same files.

## Design decisions

The main decisions are below. The [research log](research-log/2026-10-31.md) has the full reasoning.

| Decision | Why |
|---|---|
| Keep `profile()` separate from the CLI | It can be tested with a hand-built DataFrame, and each file has one job, so the kind of bug tells me where to look: a wrong calculation or type label is in `profiler.py`, and a crash on a bad file, an argument problem or a display issue is in `data_explorer.py` |
| Return a list of dicts | One dict is one table row, and each value is labelled by its key |
| Use `number`, `text`, `date` and `empty` | They describe what a column means, not how pandas stores it (dates load as `str`, integers with blanks as `float64`) |
| Check for empty columns first | An all-blank column loads as `float64`, so it would pass the numeric check and give `nan` |
| Drop blanks before the date check | A blank looks exactly like a failed date, which would turn `col_6` into text |
| Convert stats with `float()` | Callers get plain Python numbers rather than NumPy types |
| Use the sample std | It's the pandas default, and a dataset is usually a sample. NumPy's default gives a different answer |
| Make `filepath` positional | The tool can't run without it, and the command is shorter than `--dataset` |
| Try UTF-8, then latin-1 | UTF-8 is the modern default, and latin-1 can read older files. latin-1 accepts every byte, so the "both failed" branch should almost never run |
| Send errors to stderr and exit with 1 | Errors stay out of redirected output, and other programs can tell the run failed |
| Round only for display | `profile()` keeps full precision, and only the on-screen copy becomes text |
| Test `is None`, not `if not value` | `col_9` has a real std of `0.0`, which is falsy |
| Turn off `rich` markup and emoji codes | Column names come from the user's file, so they are data, not instructions |
| Build test data by hand | I know the right answers, and changing the sample files can't break the tests |
| Use exact null counts in the sample data | Each column's null % is known in advance, rather than left to chance |

## Challenges and fixes

| Challenge | Fix |
|---|---|
| All-integer sample data gave the profiler nothing to find | Rebuilt it twice, ending with one column per case, worked out from the four profile fields |
| `Path(/data.csv)` was a `SyntaxError`, and the `/` pointed at the top of the disk | Built the path from `Path(__file__)`, so it works from any folder |
| `len(df.shape[1])` raised a `TypeError` | Looped over `df.columns`, which also gives the column name |
| Null % was upside down: 500% instead of 20% | Divided missing values by total rows |
| `.unique()` returned the values, not a count | Switched to `.nunique()`, which also leaves out missing values |
| `read_csv` at the top of `profiler.py` ran on every import | Moved loading into `data_explorer.py` |
| The empty `col_12` returned `nan` | Added the "empty" check before the numeric check |
| `--sep "\t"` arrives as a backslash and a `t`, not a tab | `load_file` converts it to a real tab |
| A column named `[/x]` crashed `rich` with a `MarkupError` | Turned off markup with `Console(markup=False, emoji=False)` |
| Std printed `nan` for a column with one real value | `format_value` shows "—" for `nan` as well as `None` |
| `make_sample_df["score"]` raised a `TypeError` | Called the function with `()`, and picked the column from the result |
| `python3 tests/test_profiler.py` raised `ModuleNotFoundError` | Used `pytest`, which reads `pythonpath` from `pyproject.toml` |
| Sample dates had no blanks, so nothing tested the `dropna()` date behaviour | Added `test_date_column_with_missing_values` |

## Running the Tests

From the repo root:

```bash
pytest 3_scidata/tests
```

Expected output:

```
8 passed in 0.xxs
```

Plain `pytest` from the repo root runs the tests for every day.

## Known limitations

- A column with mixed values, like mostly numbers plus a stray `abc`, is labelled `text` and gets no stats.
  pandas loads the whole column as text if a single value can't be converted.
- `NA`, `n/a` and blanks count as missing because pandas treats them that way by default. Other markers,
  like `unknown` or `missing`, don't, and they turn a number column into text.
- Only `YYYY-MM-DD` dates are detected. Other date formats are labelled `text`.
- A file whose rows have more fields than the header, or a folder passed as the path, still ends in a
  traceback rather than a one-line error.
- The wrong separator isn't an error. Pandas reads the file as one long column, so the table shows a single row.

## Concepts Covered

| Concept | Location |
|---|---|
| Profiling a dataset by column | `profiler.py` |
| Missing values with `isna`, `dropna` and `nunique` | `profiler.py` |
| Type detection with `is_numeric_dtype` and `pd.to_datetime` | `profiler.py` |
| Why a CSV has no types, and how pandas infers them | `profiler.py`, `data/generate_data.py` |
| Command-line arguments with `argparse` | `data_explorer.py` |
| Nested `try`/`except` and encoding fallback | `load_file` in `data_explorer.py` |
| stderr, stdout and exit codes | `exit_with_error` in `data_explorer.py` |
| Tables with `rich`, and treating file contents as plain data | `print_table` in `data_explorer.py` |
| Argument unpacking with `*` | `print_table` in `data_explorer.py` |
| Separating calculation from display | `profiler.py` vs `data_explorer.py` |
| Testing edge cases with hand-built data | `tests/test_profiler.py` |

## Roadmap

This is Day 3 of a 115-day ML engineering curriculum. Upcoming builds include a training loop, autograd
engine, attention mechanism, and transformer from scratch.

## Author

Emmanuel Adeloju — [github.com/Prodholly](https://github.com/Prodholly)

## License

MIT
