"""DQ Report."""

from typing import Tuple, Dict, List
from collections import Counter
from dataclasses import dataclass
from user_input.metrics import Metric
from functools import lru_cache

import pandas as pd

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


@dataclass
class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"

    def fit(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report."""
        self.report_ = {}
        report = self.report_

        # Check if engine supported
        if self.engine != "pandas":
            raise NotImplementedError("Only pandas API currently supported!")
        report["title"] = f"DQ Report for tables {sorted(list(tables.keys()))}"

        result = {"table_name": [],
                  "metric": [],
                  "limits": [],
                  "values": [],
                  "status": [],
                  "error": []}

        for name, table in tables.items():
            for table_name, metric, limits in self.checklist:
                if name == table_name:
                    result["table_name"].append(name)
                    try:
                        res = metric(table)
                        result["metric"].append(str(metric))
                        result["limits"].append(str(limits))
                        result["values"].append(res)

                        status = "."
                        for param, values in limits.items():
                            if not (values[0] <= res.get(param) <= values[1]):
                                status = "F"
                                break

                        result["status"].append(status)
                        result["error"].append("")

                    except Exception as e:
                        result["metric"].append(metric)
                        result["limits"].append(limits)
                        result["values"].append({})
                        result["status"].append("E")
                        result["error"].append(type(e).__name__)
                else:
                    continue

        rep = pd.DataFrame(result)
        stats = Counter(result["status"])
        passed = stats.get(".") if stats.get(".") is not None else 0
        passed_percent = round(100.0 * passed / len(rep), 2)
        failed = stats.get("F") if stats.get("F") is not None else 0
        failed_percent = round(100.0 * failed / len(rep), 2)
        errors = stats.get("E") if stats.get("E") is not None else 0
        errors_percent = round(100.0 * errors / len(rep), 2)

        report["result"] = rep
        report["passed"] = passed
        report["passed_pct"] = passed_percent
        report["failed"] = failed
        report["failed_pct"] = failed_percent
        report["errors"] = errors
        report["errors_pct"] = errors_percent
        report["total"] = len(rep)

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
