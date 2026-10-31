# test_profiler.py: tests for profile().
# Each test builds a tiny DataFrame by hand, so you can work out the right answer yourself.
# They don't read data/data.csv, so changing that file can't break them.

import pandas as pd
import pytest
from data_explorer.profiler import profile

STAT_KEYS = ["mean", "std", "min", "max"]


def make_sample_df():
    data = {
        "score": [1, 2, 3, 4, None],
        "name": ["fisayo", "zoe", "justina", "deola", "dara"],
        "when": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"]
    }
    return pd.DataFrame(data)


def profile_by_column(df):
    return {row["column"]: row for row in profile(df)}


def test_returns_one_row_per_column():
    df = make_sample_df()
    result = profile(df)

    assert len(result) == len(df.columns)
    assert [row["column"] for row in result] == list(df.columns)


def test_null_pct_with_known_missing_values():
    by_column = profile_by_column(make_sample_df())

    assert by_column["score"]["null_pct"] == pytest.approx(20.0)
    assert by_column["name"]["null_pct"] == pytest.approx(0.0)


def test_numeric_column_has_stats():
    score = profile_by_column(make_sample_df())["score"]

    assert score["dtype"] == "number"
    assert score["unique"] == 4
    assert score["mean"] == pytest.approx(2.5)
    assert score["std"] == pytest.approx(1.291, abs=0.001)
    assert score["min"] == pytest.approx(1.0)
    assert score["max"] == pytest.approx(4.0)


def test_text_column_has_no_stats():
    name = profile_by_column(make_sample_df())["name"]

    assert name["dtype"] == "text"
    assert name["unique"] == 5
    for key in STAT_KEYS:
        assert name[key] is None


def test_date_column_is_detected():
    when = profile_by_column(make_sample_df())["when"]

    assert when["dtype"] == "date"
    for key in STAT_KEYS:
        assert when[key] is None


def test_date_column_with_missing_values():
    df = pd.DataFrame({"when": ["2025-01-01", None, "2025-01-03"]})
    when = profile_by_column(df)["when"]

    assert when["dtype"] == "date"
    assert when["null_pct"] == pytest.approx(100 / 3)


def test_all_empty_column():
    df = pd.DataFrame({"blank": [None, None, None]})
    blank = profile_by_column(df)["blank"]

    assert blank["dtype"] == "empty"
    assert blank["null_pct"] == pytest.approx(100.0)
    assert blank["unique"] == 0
    for key in STAT_KEYS:
        assert blank[key] is None


def test_constant_column_has_zero_std():
    df = pd.DataFrame({"same": [7, 7, 7]})
    same = profile_by_column(df)["same"]

    assert same["std"] == 0.0
    assert same["unique"] == 1
