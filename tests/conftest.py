"""Shared fixtures: a local SparkSession and DataFrame builders for both engines."""

from __future__ import annotations

import os
import re
import shutil
import sys

import pandas as pd
import pytest

# pandas dtypes for DDL types, used to give empty pandas DataFrames a proper schema
_PANDAS_DTYPES = {
    "int": "int64",
    "double": "float64",
    "string": "object",
    "date": "datetime64[ns]",
    "timestamp": "datetime64[ns]",
}


def _skip_spark_tests(reason):
    # CI sets REQUIRE_SPARK to make sure the Spark tests actually run
    if os.environ.get("REQUIRE_SPARK"):
        pytest.fail(reason)
    pytest.skip(reason)


@pytest.fixture(scope="session")
def spark():
    """Local SparkSession. Spark tests are skipped if PySpark or Java is missing."""
    try:
        from pyspark.sql import SparkSession
    except ImportError:
        _skip_spark_tests("PySpark is not installed")
    if not (os.environ.get("JAVA_HOME") or shutil.which("java")):
        _skip_spark_tests("Java is required to run Spark tests")

    # Python workers must use the same interpreter as the driver
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    session = (
        SparkSession.builder.master("local[1]")
        .appName("data-quality-tests")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )
    session.sparkContext.setLogLevel("ERROR")
    yield session
    session.stop()


@pytest.fixture(params=["pandas", "spark"])
def engine(request):
    return request.param


@pytest.fixture
def make_df(request, engine):
    """Build a pandas or Spark DataFrame from rows and a Spark DDL schema.

    ``make_df([(1, "a")], "x int, s string")``
    """
    spark = request.getfixturevalue("spark") if engine == "spark" else None

    def build(rows, schema):
        if spark is not None:
            return spark.createDataFrame(rows, schema)
        # "x int, `a b` decimal(10,2)" -> {"x": "int", "a b": "decimal(10,2)"}
        fields = {}
        for field in re.split(r",(?![^(]*\))", schema):
            name, dtype = re.fullmatch(r"\s*(`[^`]*`|\S+)\s+(.+?)\s*", field).groups()
            fields[name.strip("`")] = dtype
        df = pd.DataFrame(rows, columns=list(fields))
        if not rows:
            df = df.astype({name: _PANDAS_DTYPES[dtype] for name, dtype in fields.items()})
        return df

    return build
