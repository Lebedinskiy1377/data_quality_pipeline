"""The example checklist catches the problems planted in the sample data."""

import datetime as dt
import importlib.util
from pathlib import Path

import pytest

from data_quality import Report, metrics

DEMO = Path(__file__).parents[1] / "examples" / "demo.py"

EXPECTED_FAILURES = [
    ("sales", "CountDuplicates(columns=['day', 'item_id'])"),
    ("sales", "CountRatioBelow(column_x='revenue', column_y='price', column_z='qty', strict=True)"),
    ("relevance", "CountLag(column='day', fmt='%Y-%m-%d')"),
    ("relevance", "CountBelowColumn(column_x='clicks', column_y='payments', strict=True)"),
]


@pytest.fixture(scope="module")
def demo():
    spec = importlib.util.spec_from_file_location("demo", DEMO)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def tables(request, engine, demo):
    if engine == "pandas":
        return demo.load_pandas()
    return demo.load_spark(request.getfixturevalue("spark"))


def test_demo_checklist(demo, tables, monkeypatch):
    # sales were last updated a day before, relevance a month before
    monkeypatch.setattr(metrics, "_today", lambda: dt.date(2022, 10, 25))
    result = Report(demo.CHECKLIST).fit(tables)["result"]

    assert result["error"].tolist() == [""] * len(result)
    failed = result.loc[result["status"] == "F", ["table_name", "metric"]]
    assert list(failed.itertuples(index=False, name=None)) == EXPECTED_FAILURES


def test_engines_agree(demo, spark):
    pandas_result = Report(demo.CHECKLIST).fit(demo.load_pandas())["result"]
    spark_result = Report(demo.CHECKLIST).fit(demo.load_spark(spark))["result"]

    assert spark_result["status"].tolist() == pandas_result["status"].tolist()
    for spark_values, pandas_values in zip(
        spark_result["values"], pandas_result["values"], strict=True
    ):
        assert spark_values == pytest.approx(pandas_values)


def test_main(demo, capsys):
    demo.main([])
    output = capsys.readouterr().out
    assert output.startswith("Engine: pandas\nDQ Report for tables ['relevance', 'sales']")
    assert "Total: 16" in output
