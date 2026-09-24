"""Data quality report: run a checklist of metrics against a set of tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from data_quality.metrics import Metric

if TYPE_CHECKING:
    from data_quality.metrics import DataFrame

__all__ = ["ERROR", "FAILED", "PASSED", "Check", "Limits", "Report"]

Limits = dict[str, tuple[float, float]]
Check = tuple[str, Metric, Limits]

# Status of a check in the report
PASSED = "."  # all values are within their limits
FAILED = "F"  # a value is outside its limits or missing
ERROR = "E"  # the check could not be run, e.g. an unknown table or column

_RESULT_COLUMNS = ["table_name", "metric", "limits", "values", "status", "error"]


@dataclass
class Report:
    """Data quality report.

    ``checklist`` is a list of ``(table_name, metric, limits)`` checks.
    ``limits`` maps keys of the metric result to inclusive ``(low, high)``
    ranges, e.g. ``("sales", CountNull(["qty"]), {"count": (0, 0)})``. Use
    empty limits (``{}``) to only record a value.
    """

    checklist: list[Check]
    report_: dict[str, Any] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.checklist = list(self.checklist)
        for check in self.checklist:
            _validate_check(check)

    def fit(self, tables: Mapping[str, DataFrame]) -> dict[str, Any]:
        """Run the checklist against ``tables`` and build the report.

        ``tables`` maps table names used in the checklist to pandas or PySpark
        DataFrames. A check that raises an exception gets the ``E`` status
        and does not stop the other checks.

        Returns the report (also saved as ``report_``): a dict with the
        ``title``, the ``result`` DataFrame with a row per check, and the
        ``passed``, ``failed``, ``errors`` and ``total`` counts with their
        ``*_pct`` percentages.
        """
        if not isinstance(tables, Mapping):
            raise TypeError(f"tables must map table names to DataFrames, got {tables!r}")
        rows = [_run_check(tables, *check) for check in self.checklist]
        result = pd.DataFrame(rows, columns=_RESULT_COLUMNS)
        total = len(result)

        report: dict[str, Any] = {
            "title": f"DQ Report for tables {sorted(map(str, tables))}",
            "result": result,
        }
        for key, status in (("passed", PASSED), ("failed", FAILED), ("errors", ERROR)):
            count = int((result["status"] == status).sum())
            report[key] = count
            report[f"{key}_pct"] = round(100 * count / total, 2) if total else 0.0
        report["total"] = total

        self.report_ = report
        return report

    def to_str(self) -> str:
        """Format the report as text."""
        if self.report_ is None:
            raise RuntimeError(
                "This Report instance is not fitted yet. Call 'fit' before using this method."
            )
        report = self.report_

        result = report["result"].copy()
        result["values"] = result["values"].map(_format_values)
        return (
            f"{report['title']}\n\n"
            f"{_left_aligned(result)}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )


def _validate_check(check: Any) -> None:
    try:
        _, metric, limits = check
    except (TypeError, ValueError):
        raise ValueError(
            f"A check must be a (table_name, metric, limits) tuple, got {check!r}"
        ) from None
    if not isinstance(metric, Metric):
        raise TypeError(f"Expected a Metric instance, got {metric!r} in check {check!r}")
    if not isinstance(limits, Mapping):
        raise TypeError(f"limits must be a dict, got {limits!r} in check {check!r}")
    for key, bounds in limits.items():
        if not _is_range(bounds):
            raise ValueError(
                f"Limit {key!r} must be a (low, high) pair with low <= high, "
                f"got {bounds!r} in check {check!r}"
            )


def _is_range(bounds: Any) -> bool:
    if not isinstance(bounds, tuple | list) or len(bounds) != 2:
        return False
    low, high = bounds
    try:
        return low is not None and high is not None and low <= high
    except TypeError:  # bounds that can't be compared
        return False


def _run_check(
    tables: Mapping[str, DataFrame], table_name: str, metric: Metric, limits: Limits
) -> dict[str, Any]:
    row = {
        "table_name": table_name,
        "metric": repr(metric),
        "limits": str(limits),
        "values": {},
        "status": PASSED,
        "error": "",
    }
    try:
        if table_name not in tables:
            raise LookupError(f"table {table_name!r} not found")
        values = metric(tables[table_name])
        if not isinstance(values, Mapping):
            raise TypeError(f"{type(metric).__name__} returned {values!r} instead of a dict")
        row["values"] = values
        unknown = [key for key in limits if key not in values]
        if unknown:
            raise LookupError(f"limits {unknown} are not in the metric result {list(values)}")
        if not all(_within(values[key], low, high) for key, (low, high) in limits.items()):
            row["status"] = FAILED
    except Exception as exc:  # a broken check is reported and must not stop the others
        row["status"] = ERROR
        row["error"] = _describe(exc)
    return row


def _within(value: Any, low: float, high: float) -> bool:
    """Whether ``low <= value <= high``; a missing value (None, NaN) never is."""
    if value is None or (pd.api.types.is_scalar(value) and pd.isna(value)):
        return False
    return low <= value <= high


def _describe(exc: Exception) -> str:
    """Exception name and the first line of its message."""
    lines = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {lines[0]}" if lines else type(exc).__name__


def _format_values(values: dict[str, Any]) -> str:
    """Metric values with floats rounded for display."""
    return str({key: _round(value) for key, value in values.items()})


def _round(value: Any) -> Any:
    """Round a float to 4 decimal places, or to 4 significant digits if it is smaller."""
    if not isinstance(value, float):
        return value
    return round(value, 4) if abs(value) >= 1e-4 else float(f"{value:.4g}")


def _left_aligned(df: pd.DataFrame) -> str:
    """Render a DataFrame as a text table with left-aligned, untruncated columns."""
    lines = [["", *map(str, df.columns)]]
    for index, row in zip(df.index, df.itertuples(index=False), strict=True):
        lines.append([str(index), *map(str, row)])
    widths = [max(map(len, column)) for column in zip(*lines, strict=True)]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(line, widths, strict=True)).rstrip()
        for line in lines
    )
