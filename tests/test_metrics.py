"""Metric tests. Every test runs on pandas and on Spark and expects the same result."""

import datetime as dt
import json
from decimal import Decimal

import pandas as pd
import pytest

from data_quality import (
    CountBelowColumn,
    CountBelowValue,
    CountCB,
    CountDuplicates,
    CountLag,
    CountNull,
    CountRatioBelow,
    CountTotal,
    CountValue,
    CountZeros,
    metrics,
)

NAN = float("nan")


def share(count, total):
    return {"total": total, "count": count, "delta": count / total}


def test_count_total(make_df):
    assert CountTotal()(make_df([(1,), (2,), (3,)], "x int")) == {"total": 3}
    assert CountTotal()(make_df([], "x int")) == {"total": 0}


def test_count_zeros(make_df):
    df = make_df([(0.0,), (1.5,), (0.0,), (None,)], "x double")
    assert CountZeros("x")(df) == share(2, 4)


def test_empty_table_has_zero_delta(make_df):
    assert CountZeros("x")(make_df([], "x double")) == {"total": 0, "count": 0, "delta": 0.0}


def test_column_names_are_literal(make_df):
    # Spark would read `a.b` as field b of a struct column a
    df = make_df([(0, 1.0), (1, NAN)], "`a.b` int, `c d` double")
    assert CountZeros("a.b")(df) == share(1, 2)
    assert CountNull(["a.b", "c d"])(df) == share(1, 2)
    assert CountDuplicates(["a.b"])(df) == share(0, 2)
    assert CountCB("a.b", conf=0.5)(df) == pytest.approx({"lcb": 0.25, "ucb": 0.75})


@pytest.mark.parametrize(("aggregation", "count"), [("any", 3), ("all", 1)])
def test_count_null(make_df, aggregation, count):
    df = make_df(
        [(1.0, "a"), (NAN, "b"), (None, None), (4.0, None), (5.0, "e")],
        "x double, s string",
    )
    assert CountNull(["x", "s"], aggregation)(df) == share(count, 5)


def test_count_null_on_date_column(make_df):
    # Spark's isnan() fails on dates (and on strings in ANSI mode)
    df = make_df([(dt.date(2022, 10, 24),), (None,)], "day date")
    assert CountNull(["day"])(df) == share(1, 2)


def test_count_null_accepts_a_column_name():
    assert CountNull("x").columns == ["x"]


def test_count_null_rejects_unknown_aggregation():
    with pytest.raises(ValueError, match="aggregation"):
        CountNull(["x"], aggregation="some")


def test_count_duplicates(make_df):
    df = make_df(
        [("a", 1), ("a", 1), ("a", 1), ("a", 2), (None, 3), (None, 3)],
        "k string, v int",
    )
    assert CountDuplicates(["k", "v"])(df) == share(3, 6)
    assert CountDuplicates(["k"])(df) == share(4, 6)


def test_count_duplicates_missing_values_are_equal(make_df):
    df = make_df([(None,), (NAN,), (1.0,)], "x double")
    assert CountDuplicates(["x"])(df) == share(1, 3)


def test_count_duplicates_none_and_nan_in_pandas_object_column():
    df = pd.DataFrame({"s": pd.Series(["a", None, NAN], dtype=object)})
    assert CountDuplicates(["s"])(df) == share(1, 3)


def test_count_duplicates_empty(make_df):
    df = make_df([], "k string")
    assert CountDuplicates(["k"])(df) == {"total": 0, "count": 0, "delta": 0.0}


def test_count_value(make_df):
    df = make_df([("a",), ("b",), ("a",), (None,)], "s string")
    assert CountValue("s", "a")(df) == share(2, 4)


@pytest.mark.parametrize(("strict", "count"), [(False, 2), (True, 1)])
def test_count_below_value(make_df, strict, count):
    df = make_df([(1.0,), (2.0,), (3.0,), (NAN,), (None,)], "x double")
    assert CountBelowValue("x", 2.0, strict)(df) == share(count, 5)


@pytest.mark.parametrize("value", [None, NAN])
def test_missing_value_is_rejected(value):
    # NaN = NaN is true in Spark and false in pandas
    with pytest.raises(ValueError, match="CountNull"):
        CountValue("x", value)
    with pytest.raises(ValueError, match="CountNull"):
        CountBelowValue("x", value)


@pytest.mark.parametrize(("strict", "count"), [(False, 2), (True, 1)])
def test_count_below_column(make_df, strict, count):
    # Spark orders NaN above all numbers, so `1 <= NaN` is true there: missing values are skipped
    df = make_df(
        [(1.0, 2.0), (2.0, 2.0), (3.0, 2.0), (1.0, NAN), (NAN, 1.0), (None, 1.0)],
        "x double, y double",
    )
    assert CountBelowColumn("x", "y", strict)(df) == share(count, 6)


@pytest.mark.parametrize(("strict", "count"), [(False, 2), (True, 1)])
def test_count_ratio_below(make_df, strict, count):
    # a zero divisor is skipped instead of failing with DIVIDE_BY_ZERO in Spark's ANSI mode
    df = make_df(
        [(500.0, 100.0, 6), (600.0, 100.0, 6), (700.0, 100.0, 6), (100.0, 0.0, 1), (None, 1.0, 1)],
        "revenue double, price double, qty int",
    )
    assert CountRatioBelow("revenue", "price", "qty", strict)(df) == share(count, 5)


def test_count_ratio_below_on_empty_string_columns(make_df):
    # Spark infers string columns from a CSV file with only a header
    df = make_df([], "x string, y string, z string")
    assert CountRatioBelow("x", "y", "z")(df) == {"total": 0, "count": 0, "delta": 0.0}


def test_count_cb_matches_pandas_quantile(make_df):
    values = [0.0, 330.0, 400.0, 500.0, 720.0, 850.0, 1600.0, None]
    expected = pd.Series(values, dtype=float).quantile([0.025, 0.975]).tolist()
    result = CountCB("x")(make_df([(v,) for v in values], "x double"))
    assert result == pytest.approx({"lcb": expected[0], "ucb": expected[1]})


def test_count_cb_interpolates(make_df):
    df = make_df([(1,), (2,), (3,), (4,)], "x int")
    assert CountCB("x", conf=0.5)(df) == pytest.approx({"lcb": 1.75, "ucb": 3.25})


def test_count_cb_on_decimals(make_df):
    df = make_df([(Decimal("1.50"),), (Decimal("2.50"),), (None,)], "x decimal(10,2)")
    assert CountCB("x", conf=0.5)(df) == pytest.approx({"lcb": 1.75, "ucb": 2.25})


def test_count_cb_empty(make_df):
    assert CountCB("x")(make_df([], "x double")) == {"lcb": None, "ucb": None}


@pytest.mark.parametrize("conf", [0, 1, 95])
def test_count_cb_rejects_invalid_conf(conf):
    with pytest.raises(ValueError, match="conf"):
        CountCB("x", conf=conf)


@pytest.fixture
def today(monkeypatch):
    monkeypatch.setattr(metrics, "_today", lambda: dt.date(2022, 10, 27))


@pytest.mark.usefixtures("today")
@pytest.mark.parametrize(
    ("rows", "schema"),
    [
        ([(dt.date(2022, 10, 24),), (dt.date(2022, 10, 20),), (None,)], "day date"),
        ([(dt.datetime(2022, 10, 24, 12, 30),), (dt.datetime(2022, 10, 1),)], "day timestamp"),
        ([("2022-10-24 12:30:00",), ("2022-10-20",), ("",), (None,)], "day string"),
    ],
    ids=["date", "timestamp", "string"],
)
def test_count_lag(make_df, rows, schema):
    result = CountLag("day")(make_df(rows, schema))
    assert result == {"today": "2022-10-27", "last_day": "2022-10-24", "lag": 3}


@pytest.mark.usefixtures("today")
def test_count_lag_format(make_df):
    df = make_df([(dt.date(2022, 10, 24),)], "day date")
    result = CountLag("day", fmt="%d.%m.%Y")(df)
    assert result == {"today": "27.10.2022", "last_day": "24.10.2022", "lag": 3}


@pytest.mark.usefixtures("today")
def test_count_lag_empty(make_df):
    result = CountLag("day")(make_df([], "day date"))
    assert result == {"today": "2022-10-27", "last_day": None, "lag": None}


def test_count_lag_rejects_numbers(make_df):
    df = make_df([(20221024,)], "day int")
    with pytest.raises(TypeError, match="CountLag needs a date"):
        CountLag("day")(df)


@pytest.mark.usefixtures("today")
def test_count_lag_uses_spark_session_time_zone(spark):
    # The timestamp is 2022-10-25 09:30 in UTC, but the date is taken in the session time zone
    previous = spark.conf.get("spark.sql.session.timeZone")
    spark.conf.set("spark.sql.session.timeZone", "Pacific/Honolulu")
    try:
        df = spark.sql("SELECT TIMESTAMP '2022-10-24 23:30:00' AS day")
        result = CountLag("day")(df)
    finally:
        spark.conf.set("spark.sql.session.timeZone", previous)
    assert result["last_day"] == "2022-10-24"


def test_results_are_plain_python_values(make_df):
    df = make_df([(1.0, 0, dt.date(2022, 10, 24)), (2.0, 3, None)], "x double, y int, day date")
    checks = [
        CountTotal(),
        CountZeros("y"),
        CountNull(["x"]),
        CountDuplicates(["x"]),
        CountValue("y", 3),
        CountBelowValue("x", 1.5),
        CountBelowColumn("x", "y"),
        CountRatioBelow("x", "y", "x"),
        CountCB("x"),
        CountLag("day"),
    ]
    for metric in checks:
        result = metric(df)
        # no numpy scalars: they print as np.int64(1) and are not JSON serialisable
        assert all(type(v) in (int, float, str, type(None)) for v in result.values()), result
        json.dumps(result)


def test_unsupported_dataframe_type():
    with pytest.raises(TypeError, match="Unsupported DataFrame type"):
        CountTotal()([[1, 2], [3, 4]])
