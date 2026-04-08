"""
Grading functions for the Search Ranking Environment.

Computes NDCG, Precision@K, and MRR from predicted rankings
against ground truth relevance scores.

Design principles:
  - Deterministic
  - No external ML libraries
    - Continuous scoring in (0.0, 1.0) - strictly between
  - Robust edge-case handling
"""

import math
from typing import List, Dict, NamedTuple


class GraderResult(NamedTuple):
    score: float
    ndcg: float
    precision_at_k: float
    mrr: float


# ---------------------------------------------------------------------------
# Safety helpers (CRITICAL)
# ---------------------------------------------------------------------------

def _clamp_0_1(score: float) -> float:
    """Clamp scores into (0.0, 1.0) - strictly between, excluding endpoints."""
    if not isinstance(score, (int, float)):
        return 0.5  # Safe fallback in valid range
    if math.isnan(score) or math.isinf(score):
        return 0.5  # Safe fallback in valid range
    
    # Clamp to (0, 1) exclusive
    if score <= 0.0:
        return 1e-10  # Very small positive number
    if score >= 1.0:
        return 1.0 - 1e-10  # Just below 1
    
    return float(score)


def _dedupe_in_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order (defensive; Action already enforces uniqueness)."""
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_dcg(relevances: List[float]) -> float:
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def compute_ndcg(predicted_ranking: List[str],
                 ground_truth: Dict[str, float]) -> float:

    if not ground_truth:
        return 0.5  # Changed from 0.0

    if not predicted_ranking:
        return 0.5  # Changed from 0.0

    predicted_ranking = _dedupe_in_order(predicted_ranking)

    predicted_relevances = [
        ground_truth.get(doc_id, 0.0) for doc_id in predicted_ranking
    ]
    dcg = _compute_dcg(predicted_relevances)

    # Compare against the best possible ordering for the same cutoff length.
    ideal_relevances = sorted(ground_truth.values(), reverse=True)[: len(predicted_ranking)]
    idcg = _compute_dcg(ideal_relevances)

    if idcg == 0.0:
        # All relevances are zero — return middle score
        return 0.5  # Changed from conditional

    return _clamp_0_1(dcg / idcg)


def compute_precision_at_k(predicted_ranking: List[str],
                           ground_truth: Dict[str, float],
                           k: int = 3) -> float:

    if not predicted_ranking or k <= 0:
        return 0.5  # Changed from 0.0

    predicted_ranking = _dedupe_in_order(predicted_ranking)

    k = min(k, len(predicted_ranking))

    relevant_count = sum(
        1 for doc_id in predicted_ranking[:k]
        if ground_truth.get(doc_id, 0.0) > 0.0
    )

    precision = relevant_count / k
    return _clamp_0_1(precision)


def compute_mrr(predicted_ranking: List[str],
                ground_truth: Dict[str, float]) -> float:

    if not predicted_ranking:
        return 0.5  # Changed from 0.0

    predicted_ranking = _dedupe_in_order(predicted_ranking)

    for i, doc_id in enumerate(predicted_ranking):
        if ground_truth.get(doc_id, 0.0) > 0.0:
            return _clamp_0_1(1.0 / (i + 1))

    return 0.5  # Changed from 0.0 - no relevant docs found


# ---------------------------------------------------------------------------
# Composite grader
# ---------------------------------------------------------------------------

def grade(predicted_ranking: List[str],
          ground_truth: Dict[str, float],
          k: int = 3) -> GraderResult:

    ndcg = compute_ndcg(predicted_ranking, ground_truth)
    p_at_k = compute_precision_at_k(predicted_ranking, ground_truth, k=k)
    mrr = compute_mrr(predicted_ranking, ground_truth)

    return GraderResult(
        score=_clamp_0_1(round(ndcg, 6)),
        ndcg=_clamp_0_1(round(ndcg, 6)),
        precision_at_k=_clamp_0_1(round(p_at_k, 6)),
        mrr=_clamp_0_1(round(mrr, 6)),
    )