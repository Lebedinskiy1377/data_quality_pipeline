"""Report tests."""

from dataclasses import dataclass

import pytest

from data_quality import CountCB, CountTotal, CountZeros, Metric, Report


@dataclass
class Constant(Metric):
    """Returns ``result`` for any DataFrame."""

    result: object

    def _call_pandas(self, df):
        return self.result

    def _call_pyspark(self, df):
        return self.result


def test_check_statuses(make_df):
    tables = {"t": make_df([(0,), (1,), (2,)], "x int")}
    checklist = [
        ("t", CountTotal(), {"total": (1, 10)}),  # passed
        ("t", CountZeros("x"), {"count": (0, 0)}),  # failed
        ("t", CountZeros("y"), {}),  # error: unknown column
        ("other", CountTotal(), {}),  # error: unknown table
        ("t", CountTotal(), {"rows": (0, 1)}),  # error: unknown result key
    ]
    report = Report(checklist).fit(tables)
    result = report["result"]

    assert result["status"].tolist() == [".", "F", "E", "E", "E"]
    assert result["error"][0] == ""
    assert "'y'" in result["error"][2] or "`y`" in result["error"][2]
    assert result["error"][3] == "LookupError: table 'other' not found"
    assert result["error"][4] == (
        "LookupError: limits ['rows'] are not in the metric result ['total']"
    )
    assert result["values"][4] == {"total": 3}

    assert (report["passed"], report["failed"], report["errors"], report["total"]) == (1, 1, 3, 5)
    assert (report["passed_pct"], report["failed_pct"], report["errors_pct"]) == (20.0, 20.0, 60.0)
    assert report["title"] == "DQ Report for tables ['t']"


def test_limits_are_inclusive(make_df):
    tables = {"t": make_df([(1,), (2,)], "x int")}
    report = Report([("t", CountTotal(), {"total": (2, 2)})]).fit(tables)
    assert report["passed"] == 1


def test_missing_value_fails(make_df):
    tables = {"t": make_df([], "x double")}
    report = Report([("t", CountCB("x"), {"lcb": (0, 10)})]).fit(tables)
    assert report["result"]["status"][0] == "F"


def test_unknown_limit_is_an_error_even_if_another_limit_fails(make_df):
    tables = {"t": make_df([(0,)], "x int")}
    report = Report([("t", CountZeros("x"), {"count": (0, 0), "detla": (0, 1)})]).fit(tables)
    assert report["result"]["status"][0] == "E"
    assert "['detla']" in report["result"]["error"][0]


def test_refit_uses_new_data(make_df):
    # Reports used to be cached by str(tables). For Spark that is just the schema,
    # so fitting an empty table returned the report of the previous one.
    report = Report([("t", CountTotal(), {"total": (1, 10)})])
    first = report.fit({"t": make_df([(1,), (2,)], "x int")})
    second = report.fit({"t": make_df([], "x int")})

    assert first["result"]["values"][0] == {"total": 2}
    assert second["result"]["values"][0] == {"total": 0}
    assert second["result"]["status"][0] == "F"
    assert report.report_ is second


def test_metric_must_return_a_dict(make_df):
    report = Report([("t", Constant(None), {})]).fit({"t": make_df([(1,)], "x int")})
    assert report["result"]["status"][0] == "E"
    assert report["result"]["error"][0] == "TypeError: Constant returned None instead of a dict"


def test_to_str(make_df):
    report = Report(
        [
            ("t", CountZeros("x"), {"delta": (0, 0.5)}),
            ("t", Constant({"small": 0.0000312, "big": 1487.49999}), {}),
        ]
    )
    report.fit({"t": make_df([(0,), (1,), (2,)], "x int")})
    text = report.to_str()

    assert text.startswith("DQ Report for tables ['t']\n")
    assert "CountZeros(column='x')" in text
    assert "{'total': 3, 'count': 1, 'delta': 0.3333}" in text
    assert "{'small': 3.12e-05, 'big': 1487.5}" in text
    assert "Passed: 2 (100.0%)" in text
    assert text.endswith("Total: 2")


def test_to_str_before_fit():
    with pytest.raises(RuntimeError, match="not fitted"):
        Report([]).to_str()


def test_empty_checklist():
    report = Report([]).fit({})
    assert report["total"] == 0
    assert report["passed_pct"] == 0.0


def test_fit_needs_a_dict_of_tables():
    with pytest.raises(TypeError, match="tables"):
        Report([]).fit(None)


def test_fitted_reports_can_be_compared():
    first, second = Report([]), Report([])
    first.fit({})
    assert first == second


@pytest.mark.parametrize(
    "check",
    [
        ("t", CountTotal()),  # no limits
        ("t", CountTotal, {}),  # metric class instead of an instance
        ("t", CountTotal(), {"total": 5}),  # limit is not a (low, high) pair
        ("t", CountTotal(), {"total": (None, 5)}),  # missing bound
        ("t", CountTotal(), {"total": (10, 1)}),  # low > high
    ],
)
def test_invalid_check(check):
    with pytest.raises((TypeError, ValueError)):
        Report([check])
