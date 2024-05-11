"""DQ Report."""

from typing import Tuple, Dict, List, Union
from dataclasses import dataclass
from data_quality.metrics import Metric

import pandas as pd
import pyspark.sql as ps

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


@dataclass
class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"

    def fit(self, tables: Dict[str, Union[pd.DataFrame, ps.DataFrame]], check: List[CheckType]) -> Dict:
        """Calculate DQ metrics and build report."""

        if self.engine == "pandas":
            return self._fit_pandas(tables, check)

        if self.engine == "pyspark":
            return self._fit_pyspark(tables, check)

        raise NotImplementedError("Only pandas and pyspark APIs currently supported!")

    def _fit_pandas(self, tables: Dict[str, pd.DataFrame], check: List[CheckType]) -> Dict:
        """Calculate DQ metrics and build report."""
        self.checklist = check

        if not hasattr(self, "_reports"):
            self._reports = {}
        hash_key = hash(str(tables))
        if hash_key in self._reports:
            return self._reports[hash_key]

        self.report_ = {}
        report = self.report_

        result = []
        for table, metric, limits in self.checklist:
            row = {"table_name": table, "metric": repr(metric), "limits": str(limits)}
            try:
                df = tables[table]
                values = metric(df)
                row["values"] = values
                row["status"] = "."
                row["error"] = ""
                for key, (a, b) in limits.items():
                    value = values[key]
                    if not (a <= value <= b):
                        row["status"] = "F"

            except Exception as e:
                row["status"] = "E"
                row["values"] = {}
                row["error"] = type(e).__name__

            result.append(row)

        tables = sorted(list(set(tables.keys())))
        result = pd.DataFrame(result)

        report["title"] = f"DQ Report for tables {tables}"
        report["result"] = result

        total = len(result)
        passed = sum(result["status"] == ".")
        failed = sum(result["status"] == "F")
        errors = sum(result["status"] == "E")

        report["passed"] = passed
        report["passed_pct"] = round(100 * passed / total, 2)
        report["failed"] = failed
        report["failed_pct"] = round(100 * failed / total, 2)
        report["errors"] = errors
        report["errors_pct"] = round(100 * errors / total, 2)
        report["total"] = total

        self._reports[hash_key] = report

        return report

    def _fit_pyspark(self, tables: Dict[str, ps.DataFrame], check: List[CheckType]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: PySpark"""
        self.checklist = check

        if not hasattr(self, "_reports"):
            self._reports = {}
        hash_key = hash(str(tables))
        if hash_key in self._reports:
            return self._reports[hash_key]

        self.report_ = {}
        report = self.report_

        result = []
        for table, metric, limits in self.checklist:
            row = {"table_name": table, "metric": repr(metric), "limits": str(limits)}

            try:
                df = tables[table]
                values = metric(df)
                row["values"] = values
                row["status"] = "."
                row["error"] = ""

                for key, (a, b) in limits.items():
                    value = values[key]
                    if not (a <= value <= b):
                        row["status"] = "F"

            except Exception as e:
                row["status"] = "E"
                row["values"] = {}
                row["error"] = type(e).__name__

            result.append(row)

        tables = sorted(list(set(tables.keys())))
        result = pd.DataFrame(result)

        report["title"] = f"DQ Report for tables {tables}"
        report["result"] = result

        total = len(result)
        passed = sum(result["status"] == ".")
        failed = sum(result["status"] == "F")
        errors = sum(result["status"] == "E")

        report["passed"] = passed
        report["passed_pct"] = round(100 * passed / total, 2)
        report["failed"] = failed
        report["failed_pct"] = round(100 * failed / total, 2)
        report["errors"] = errors
        report["errors_pct"] = round(100 * errors / total, 2)
        report["total"] = total

        self._reports[hash_key] = report

        return report

    def to_str(self) -> None:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before using this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )
