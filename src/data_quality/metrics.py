"""Data quality metrics for pandas and PySpark DataFrames.

Every metric is a dataclass: construct it with its parameters, then call it on
a DataFrame to get a dict of results. The same metric works on pandas and
PySpark DataFrames and returns the same plain Python values (``int``,
``float``, ``str`` or ``None``) for the same data.

Missing values are treated the same way by both engines: ``None``, ``NaN``,
``NaT`` and ``pd.NA`` in pandas, ``null`` and ``NaN`` in Spark. Column names
are taken literally: a dot in a name doesn't refer to a nested Spark field.
"""

from __future__ import annotations

import datetime as dt
import functools
import operator
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd

try:
    from pyspark.sql import functions as F
    from pyspark.sql import types as T
except ImportError:  # PySpark is optional: pip install "data-quality-pipeline[spark]"
    F = T = None

if TYPE_CHECKING:
    from pyspark.sql import Column
    from pyspark.sql import DataFrame as SparkDataFrame

    DataFrame = pd.DataFrame | SparkDataFrame

__all__ = [
    "CountBelowColumn",
    "CountBelowValue",
    "CountCB",
    "CountDuplicates",
    "CountLag",
    "CountNull",
    "CountRatioBelow",
    "CountTotal",
    "CountValue",
    "CountZeros",
    "Metric",
]


@dataclass
class Metric:
    """Base class for metrics.

    Subclasses implement ``_call_pandas`` and ``_call_pyspark``; calling a
    metric dispatches to the right one based on the type of the DataFrame.
    """

    def __call__(self, df: DataFrame) -> dict[str, Any]:
        if isinstance(df, pd.DataFrame):
            return self._call_pandas(df)
        if _is_spark_dataframe(df):
            return self._call_pyspark(df)
        raise TypeError(
            f"Unsupported DataFrame type: {type(df).__module__}.{type(df).__qualname__}. "
            "Expected pandas.DataFrame or pyspark.sql.DataFrame."
        )

    def _call_pandas(self, df: pd.DataFrame) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not support pandas")

    def _call_pyspark(self, df: SparkDataFrame) -> dict[str, Any]:
        raise NotImplementedError(f"{type(self).__name__} does not support PySpark")


@dataclass
class CountTotal(Metric):
    """Total number of rows."""

    def _call_pandas(self, df: pd.DataFrame) -> dict[str, Any]:
        return {"total": len(df)}

    def _call_pyspark(self, df: SparkDataFrame) -> dict[str, Any]:
        return {"total": df.count()}


@dataclass
class CountZeros(Metric):
    """Number of rows where ``column`` is zero."""

    column: str

    def _call_pandas(self, df: pd.DataFrame) -> dict[str, Any]:
        return _share((df[self.column] == 0).sum(), len(df))

    def _call_pyspark(self, df: SparkDataFrame) -> dict[str, Any]:
        return _spark_count_where(df, _col(self.column) == 0)


@dataclass
class CountNull(Metric):
    """Number of rows with missing values in ``columns``.

    With ``aggregation="any"`` a row counts if at least one of the columns is
    missing, with ``aggregation="all"`` only if all of them are.
    """

    columns: list[str]
    aggregation: Literal["any", "all"] = "any"

    def __post_init__(self) -> None:
        self.columns = _as_column_list(self.columns)
        if self.aggregation not in ("any", "all"):
            raise ValueError(f"aggregation must be 'any' or 'all', got {self.aggregation!r}")

    def _call_pandas(self, df: pd.DataFrame) -> dict[str, Any]:
        missing = df[self.columns].isna()
        rows = missing.any(axis=1) if self.aggregation == "any" else missing.all(axis=1)
        return _share(rows.sum(), len(df))

    def _call_pyspark(self, df: SparkDataFrame) -> dict[str, Any]:
        combine = operator.or_ if self.aggregation == "any" else operator.and_
        condition = functools.reduce(combine, (_spark_is_missing(df, c) for c in self.columns))
        return _spark_count_where(df, condition)


@dataclass
class CountDuplicates(Metric):
    """Number of rows that repeat an earlier row in ``columns``.

    A value that occurs three times counts as two duplicates. All missing
    values are equal to each other.
    """

    columns: list[str]

    def __post_init__(self) -> None:
        self.columns = _as_column_list(self.columns)

    def _call_pandas(self, df: pd.DataFrame) -> dict[str, Any]:
        keys = df[self.columns]
        # pandas tells None and NaN apart in object columns; Spark has only null
        keys = keys.astype(object).where(keys.notna(), None)
        return _share(keys.duplicated().sum(), len(df))

    def _call_pyspark(self, df: SparkDataFrame) -> dict[str, Any]:
        keys = []
        for i, column in enumerate(self.columns):
            key = _col(column)
            if _is_spark_float(df, column):
                key = F.nanvl(key, F.lit(None))  # NaN and null are the same missing value
            keys.append(key.alias(f"__key{i}"))
        row = (
            df.groupBy(*keys)
            .agg(F.count(F.lit(1)).alias("__rows"))
            .agg(F.sum("__rows").alias("total"), F.sum(F.col("__rows") - 1).alias("count"))
            .first()
        )
        return _share(row["count"] or 0, row["total"] or 0)


@dataclass
class CountValue(Metric):
    """Number of rows where ``column`` equals ``value``."""

    column: str
    value: str | int | float

    def __post_init__(self) -> None:
        _check_value(self.value)

    def _call_pandas(self, df: pd.DataFrame) -> dict[str, Any]:
        return _share((df[self.column] == self.value).sum(), len(df))

    def _call_pyspark(self, df: SparkDataFrame) -> dict[str, Any]:
        return _spark_count_where(df, _col(self.column) == self.value)


@dataclass
class CountBelowValue(Metric):
    """Number of rows where ``column <= value`` (``<`` if ``strict``)."""

    column: str
    value: float
    strict: bool = False

    def __post_init__(self) -> None:
        _check_value(self.value)

    def _call_pandas(self, df: pd.DataFrame) -> dict[str, Any]:
        series = df[self.column]
        below = series < self.value if self.strict else series <= self.value
        return _share(below.sum(), len(df))

    def _call_pyspark(self, df: SparkDataFrame) -> dict[str, Any]:
        column = _col(self.column)
        below = column < self.value if self.strict else column <= self.value
        return _spark_count_where(df, below & ~_spark_is_missing(df, self.column))


@dataclass
class CountBelowColumn(Metric):
    """Number of rows where ``column_x <= column_y`` (``<`` if ``strict``)."""

    column_x: str
    column_y: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> dict[str, Any]:
        x, y = df[self.column_x], df[self.column_y]
        below = x < y if self.strict else x <= y
        return _share(below.sum(), len(df))

    def _call_pyspark(self, df: SparkDataFrame) -> dict[str, Any]:
        x, y = _col(self.column_x), _col(self.column_y)
        below = x < y if self.strict else x <= y
        present = ~_spark_is_missing(df, self.column_x) & ~_spark_is_missing(df, self.column_y)
        return _spark_count_where(df, below & present)


@dataclass
class CountRatioBelow(Metric):
    """Number of rows where ``column_x / column_y <= column_z`` (``<`` if ``strict``).

    Rows where ``column_y`` is zero are not counted.
    """

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> dict[str, Any]:
        y = df[self.column_y]
        ratio = df[self.column_x] / y.where(y != 0)
        z = df[self.column_z]
        below = ratio < z if self.strict else ratio <= z
        return _share(below.sum(), len(df))

    def _call_pyspark(self, df: SparkDataFrame) -> dict[str, Any]:
        # try_divide returns null for a zero divisor instead of failing in ANSI mode
        x, y = _col(self.column_x).cast("double"), _col(self.column_y).cast("double")
        ratio = F.try_divide(x, y)
        z = _col(self.column_z)
        below = ratio < z if self.strict else ratio <= z
        present = functools.reduce(
            operator.and_,
            (~_spark_is_missing(df, c) for c in (self.column_x, self.column_y, self.column_z)),
        )
        return _spark_count_where(df, below & present)


@dataclass
class CountCB(Metric):
    """Bounds of the central ``conf`` share of values in ``column``.

    ``lcb`` and ``ucb`` are the ``(1 - conf) / 2`` and ``(1 + conf) / 2``
    quantiles with linear interpolation (pandas' default method). They are
    ``None`` if the column has no values.
    """

    column: str
    conf: float = 0.95

    def __post_init__(self) -> None:
        if not 0 < self.conf < 1:
            raise ValueError(f"conf must be between 0 and 1, got {self.conf!r}")

    @property
    def _probabilities(self) -> list[float]:
        alpha = (1 - self.conf) / 2
        return [alpha, 1 - alpha]

    def _call_pandas(self, df: pd.DataFrame) -> dict[str, Any]:
        values = df[self.column]
        if values.dtype == object:  # e.g. Decimal values read from Parquet
            values = pd.to_numeric(values)
        lcb, ucb = values.quantile(self._probabilities).tolist()
        return {"lcb": _float_or_none(lcb), "ucb": _float_or_none(ucb)}

    def _call_pyspark(self, df: SparkDataFrame) -> dict[str, Any]:
        # Exact percentiles with linear interpolation, same as pandas.quantile().
        # approxQuantile() returns actual column values instead, so its bounds differ.
        values = df.where(~_spark_is_missing(df, self.column))
        bounds = values.agg(F.percentile(_col(self.column), self._probabilities)).first()[0]
        lcb, ucb = bounds or (None, None)
        return {"lcb": _float_or_none(lcb), "ucb": _float_or_none(ucb)}


@dataclass
class CountLag(Metric):
    """Number of days between today and the latest date in ``column``.

    ``column`` holds dates, timestamps or ISO 8601 strings like ``2022-10-24``.
    Timestamps are converted to dates in their own time zone in pandas and in
    the session time zone (``spark.sql.session.timeZone``) in Spark.

    ``fmt`` is the ``strftime`` format of ``today`` and ``last_day`` in the
    result. ``last_day`` and ``lag`` are ``None`` if the column has no values.
    """

    column: str
    fmt: str = "%Y-%m-%d"

    def _call_pandas(self, df: pd.DataFrame) -> dict[str, Any]:
        series = df[self.column]
        if pd.api.types.is_numeric_dtype(series):  # includes bool
            raise TypeError(self._not_a_date(series.dtype))
        last_day = pd.to_datetime(series, format="ISO8601").max()
        return self._result(None if pd.isna(last_day) else last_day.date())

    def _call_pyspark(self, df: SparkDataFrame) -> dict[str, Any]:
        column = _col(self.column)
        dtype = _spark_type(df, self.column)
        if isinstance(dtype, T.NumericType | T.BooleanType):
            raise TypeError(self._not_a_date(dtype.simpleString()))
        if isinstance(dtype, T.StringType):
            column = F.when(F.trim(column) != "", column)  # blank strings are missing
        last_day = df.agg(F.max(F.to_date(column))).first()[0]
        return self._result(last_day)

    def _not_a_date(self, dtype: Any) -> str:
        return (
            f"CountLag needs a date, timestamp or ISO 8601 string column, "
            f"but {self.column!r} is {dtype}"
        )

    def _result(self, last_day: dt.date | None) -> dict[str, Any]:
        today = _today()
        return {
            "today": today.strftime(self.fmt),
            "last_day": None if last_day is None else last_day.strftime(self.fmt),
            "lag": None if last_day is None else (today - last_day).days,
        }


def _today() -> dt.date:
    return dt.date.today()


def _share(count: Any, total: Any) -> dict[str, Any]:
    """Result of the counting metrics; ``delta`` is 0.0 for an empty table."""
    count, total = int(count), int(total)
    return {"total": total, "count": count, "delta": count / total if total else 0.0}


def _float_or_none(value: Any) -> float | None:
    return None if value is None or pd.isna(value) else float(value)


def _as_column_list(columns: str | list[str]) -> list[str]:
    columns = [columns] if isinstance(columns, str) else list(columns)
    if not columns:
        raise ValueError("columns must not be empty")
    return columns


def _check_value(value: Any) -> None:
    # comparisons with NaN differ: never true in pandas, NaN = NaN in Spark
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        raise ValueError(f"value must not be missing, got {value!r}; use CountNull instead")


def _is_spark_dataframe(obj: object) -> bool:
    # Before PySpark 4.0 a Spark Connect DataFrame is not a pyspark.sql.DataFrame.
    # The classes are looked up in sys.modules because Spark Connect needs extra
    # dependencies, and there can't be DataFrames of a module that isn't loaded.
    for module_name in ("pyspark.sql", "pyspark.sql.connect.dataframe"):
        module = sys.modules.get(module_name)
        if module is not None and isinstance(obj, module.DataFrame):
            return True
    return False


def _col(name: str) -> Column:
    """Spark column by its literal name, so that ``a.b`` isn't a nested field."""
    return F.col("`" + name.replace("`", "``") + "`")


def _spark_type(df: SparkDataFrame, column: str) -> T.DataType:
    return df.select(_col(column)).schema.fields[0].dataType


def _is_spark_float(df: SparkDataFrame, column: str) -> bool:
    return isinstance(_spark_type(df, column), T.FloatType | T.DoubleType)


def _spark_is_missing(df: SparkDataFrame, column: str) -> Column:
    """null, or NaN for floating point columns: what pandas treats as missing."""
    col = _col(column)
    return col.isNull() | F.isnan(col) if _is_spark_float(df, column) else col.isNull()


def _spark_count_where(df: SparkDataFrame, condition: Column) -> dict[str, Any]:
    """Count all rows and the rows matching ``condition`` in a single pass."""
    row = df.agg(
        F.count(F.lit(1)).alias("total"),
        F.count(F.when(condition, True)).alias("count"),
    ).first()
    return _share(row["count"], row["total"])
