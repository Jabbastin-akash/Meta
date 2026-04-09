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
# Constants
# ---------------------------------------------------------------------------
MIN_SCORE = 0.1
MAX_SCORE = 0.95
SAFE_FALLBACK = 0.5  # Midpoint for edge cases


# ---------------------------------------------------------------------------
# Safety helpers (CRITICAL)
# ---------------------------------------------------------------------------

def _clamp_strict_0_1(score: float) -> float:
    """
    Clamp scores into (0.0, 1.0) - strictly between, excluding endpoints.
    
    This function ensures NO value can ever be exactly 0.0 or 1.0.
    """
    if not isinstance(score, (int, float)):
        return SAFE_FALLBACK
    
    if math.isnan(score) or math.isinf(score):
        return SAFE_FALLBACK
    
    # Clamp to strict bounds
    if score <= 0.0:
        return MIN_SCORE
    if score >= 1.0:
        return MAX_SCORE
    
    # Ensure floating-point values don't sit on boundaries
    clamped = max(MIN_SCORE, min(MAX_SCORE, float(score)))
    
    # Final safety check - should never trigger, but guarantees contract
    assert 0.0 < clamped < 1.0, f"Score {clamped} escaped bounds!"
    
    return clamped


def _dedupe_in_order(items: List[str]) -> List[str]:
    """Remove duplicates while preserving order."""
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
    """Compute Discounted Cumulative Gain."""
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += (2 ** rel - 1) / math.log2(i + 2)
    return dcg


# ---------------------------------------------------------------------------
# Public metric functions
# ---------------------------------------------------------------------------

def compute_ndcg(predicted_ranking: List[str],
                 ground_truth: Dict[str, float]) -> float:
    """
    Compute Normalized Discounted Cumulative Gain.
    
    Returns a score strictly in (0, 1).
    """
    # Edge case: no ground truth
    if not ground_truth:
        return SAFE_FALLBACK

    # Edge case: no predictions
    if not predicted_ranking:
        return SAFE_FALLBACK

    predicted_ranking = _dedupe_in_order(predicted_ranking)

    # Get relevance scores for predicted ranking
    predicted_relevances = [
        ground_truth.get(doc_id, 0.0) for doc_id in predicted_ranking
    ]
    dcg = _compute_dcg(predicted_relevances)

    # Get ideal ranking
    ideal_relevances = sorted(
        ground_truth.values(), 
        reverse=True
    )[: len(predicted_ranking)]
    idcg = _compute_dcg(ideal_relevances)

    # Edge case: all documents have zero relevance
    if idcg == 0.0:
        return SAFE_FALLBACK

    # Compute NDCG
    ndcg = dcg / idcg
    
    # Clamp to (0, 1)
    return _clamp_strict_0_1(ndcg)


def compute_precision_at_k(predicted_ranking: List[str],
                           ground_truth: Dict[str, float],
                           k: int = 3) -> float:
    """
    Compute Precision at K.
    
    Returns a score strictly in (0, 1).
    """
    # Edge cases
    if not predicted_ranking or k <= 0:
        return SAFE_FALLBACK

    predicted_ranking = _dedupe_in_order(predicted_ranking)
    k = min(k, len(predicted_ranking))

    # Count relevant documents in top-k
    relevant_count = sum(
        1 for doc_id in predicted_ranking[:k]
        if ground_truth.get(doc_id, 0.0) > 0.0
    )

    # Compute precision
    precision = relevant_count / k
    
    # Clamp to (0, 1)
    return _clamp_strict_0_1(precision)


def compute_mrr(predicted_ranking: List[str],
                ground_truth: Dict[str, float]) -> float:
    """
    Compute Mean Reciprocal Rank.
    
    Returns a score strictly in (0, 1).
    """
    # Edge case: no predictions
    if not predicted_ranking:
        return SAFE_FALLBACK

    predicted_ranking = _dedupe_in_order(predicted_ranking)

    # Find first relevant document
    for i, doc_id in enumerate(predicted_ranking):
        if ground_truth.get(doc_id, 0.0) > 0.0:
            mrr = 1.0 / (i + 1)
            return _clamp_strict_0_1(mrr)

    # No relevant document found
    return MIN_SCORE


# ---------------------------------------------------------------------------
# Composite grader
# ---------------------------------------------------------------------------

def grade(predicted_ranking: List[str],
          ground_truth: Dict[str, float],
          k: int = 3) -> GraderResult:
    """
    Grade a predicted ranking against ground truth.
    
    Returns all scores strictly in (0, 1).
    """
    # Compute all metrics
    ndcg = compute_ndcg(predicted_ranking, ground_truth)
    p_at_k = compute_precision_at_k(predicted_ranking, ground_truth, k=k)
    mrr = compute_mrr(predicted_ranking, ground_truth)

    # Double-check all scores are in valid range
    ndcg = _clamp_strict_0_1(ndcg)
    p_at_k = _clamp_strict_0_1(p_at_k)
    mrr = _clamp_strict_0_1(mrr)

    # Primary score is NDCG
    score = ndcg

    return GraderResult(
        score=score,
        ndcg=ndcg,
        precision_at_k=p_at_k,
        mrr=mrr,
    )