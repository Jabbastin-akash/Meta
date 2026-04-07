"""Compatibility re-export for grading utilities.

Some tooling/tests expect top-level imports like `from grader import grade`.
The implementation lives in `server.grader`.
"""

from server.grader import (  # noqa: F401
    GraderResult,
    compute_mrr,
    compute_ndcg,
    compute_precision_at_k,
    grade,
)

__all__ = [
    "GraderResult",
    "compute_ndcg",
    "compute_precision_at_k",
    "compute_mrr",
    "grade",
]
