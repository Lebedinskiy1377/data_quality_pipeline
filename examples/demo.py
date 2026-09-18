"""Run the example checklist against the sample data in examples/data.

    python examples/demo.py                  # pandas
    python examples/demo.py --engine spark   # PySpark: pip install -e ".[spark]", needs Java
    python examples/demo.py --engine both

The sample data contains a few planted problems, so some checks fail.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_quality import (
    Check,
    CountBelowColumn,
    CountBelowValue,
    CountCB,
    CountDuplicates,
    CountLag,
    CountNull,
    CountRatioBelow,
    CountTotal,
    CountZeros,
    Report,
)

DATA_DIR = Path(__file__).parent / "data"
TABLES = {
    "sales": DATA_DIR / "ke_daily_sales.csv",
    "relevance": DATA_DIR / "ke_visits.csv",
}

# (table name, metric, limits): every limit is an inclusive (low, high) range
# for a key of the metric result. Empty limits only record the value.
CHECKLIST: list[Check] = [
    # Sales: day, item_id, qty, price, revenue
    ("sales", CountTotal(), {"total": (1, 1e6)}),
    ("sales", CountLag("day"), {"lag": (0, 3)}),
    ("sales", CountDuplicates(["day", "item_id"]), {"count": (0, 0)}),
    ("sales", CountNull(["qty"]), {"count": (0, 0)}),
    # revenue should be price * qty: rows where revenue / price < qty
    ("sales", CountRatioBelow("revenue", "price", "qty", strict=True), {"delta": (0, 0.05)}),
    ("sales", CountCB("revenue"), {}),
    ("sales", CountZeros("qty"), {"delta": (0, 0.3)}),
    ("sales", CountBelowValue("price", 100.0), {"delta": (0, 0.3)}),
    # Clickstream: day, item_id, views, clicks, payments
    ("relevance", CountTotal(), {"total": (1, 1e6)}),
    ("relevance", CountLag("day"), {"lag": (0, 3)}),
    ("relevance", CountZeros("views"), {"delta": (0, 0.2)}),
    ("relevance", CountZeros("clicks"), {"delta": (0, 0.5)}),
    ("relevance", CountNull(["views", "clicks", "payments"]), {"delta": (0, 0.1)}),
    ("relevance", CountBelowValue("views", 10), {"delta": (0, 0.5)}),
    # funnel: there can't be more clicks than views or more payments than clicks
    ("relevance", CountBelowColumn("views", "clicks", strict=True), {"count": (0, 0)}),
    ("relevance", CountBelowColumn("clicks", "payments", strict=True), {"count": (0, 0)}),
]


def load_pandas() -> dict[str, pd.DataFrame]:
    return {name: pd.read_csv(path) for name, path in TABLES.items()}


def load_spark(spark):
    return {
        name: spark.read.csv(str(path), header=True, inferSchema=True)
        for name, path in TABLES.items()
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the example data quality checklist.")
    parser.add_argument("--engine", choices=["pandas", "spark", "both"], default="pandas")
    args = parser.parse_args(argv)

    if args.engine in ("pandas", "both"):
        report = Report(CHECKLIST)
        report.fit(load_pandas())
        print(f"Engine: pandas\n{report.to_str()}\n")

    if args.engine in ("spark", "both"):
        from pyspark.sql import SparkSession

        spark = SparkSession.builder.master("local[*]").appName("dq-demo").getOrCreate()
        spark.sparkContext.setLogLevel("ERROR")
        try:
            report = Report(CHECKLIST)
            report.fit(load_spark(spark))
            print(f"Engine: PySpark\n{report.to_str()}\n")
        finally:
            spark.stop()


if __name__ == "__main__":
    main()
