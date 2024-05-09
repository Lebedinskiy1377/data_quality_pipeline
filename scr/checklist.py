"""Report checklist."""

from metrics import (
    CountTotal,
    CountLag,
    CountDuplicates,
    CountNull,
    CountRatioBelow,
    CountCB,
    CountZeros,
    CountBelowValue,
    CountBelowColumn,
)

# Checklist contains checks consist of:
# - table_name
# - metric
# - limits

CHECKLIST = [
    # Table with sales ["day", "item_id", "qty", "revenue", "price"]
    ("sales", CountTotal(), {"total": (1, 1e6)}),
    ("sales", CountLag("day"), {"lag": (0, 3)}),
    ("sales", CountDuplicates(["day", "item_id"]), {"total": (0, 0)}),
    ("sales", CountNull(["qty"]), {"total": (0, 0)}),
    ("sales", CountRatioBelow("revenue", "price", "qty", False), {"delta": (0, 0.05)}),
    ("sales", CountCB("revenue"), {}),
    ("sales", CountZeros("qty"), {"delta": (0, 0.3)}),
    ("sales", CountBelowValue("price", 100.0), {"delta": (0, 0.3)}),
    # Table with clickstream ["dt", "item_id", "views", "clicks", "payments"]
    ("relevance", CountTotal(), {"total": (1, 1e6)}),
    ("relevance", CountLag("dt"), {"lag": (0, 3)}),
    ("relevance", CountZeros("views"), {"delta": (0, 0.2)}),
    ("relevance", CountZeros("clicks"), {"delta": (0, 0.5)}),
    ("relevance", CountNull(["views", "clicks", "payments"]), {"delta": (0, 0.1)}),
    ("relevance", CountBelowValue("views", 10), {"delta": (0, 0.5)}),
    ("relevance", CountBelowColumn("clicks", "views"), {"total": (0, 0)}),
    ("relevance", CountBelowColumn("payments", "clicks"), {"total": (0, 0)}),
]
