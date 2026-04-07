"""
Grading functions for the Search Ranking Environment.

Computes NDCG, Precision@K, and MRR from predicted rankings
against ground truth relevance scores.

Design principles:
  - Deterministic
  - No external ML libraries
  - Continuous scoring strictly in (0, 1)
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
# Safety helper (CRITICAL)
# ---------------------------------------------------------------------------

def _safe_score(score: float) -> float:
    MIN_SCORE = 0.1
    MAX_SCORE = 0.85
    if score <= MIN_SCORE:
        return MIN_SCORE
    if score >= MAX_SCORE:
        return MAX_SCORE
    return score


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

    if not predicted_ranking:
        return _safe_score(0.0)

    predicted_relevances = [
        ground_truth.get(doc_id, 0.0) for doc_id in predicted_ranking
    ]
    dcg = _compute_dcg(predicted_relevances)

    ideal_relevances = sorted(ground_truth.values(), reverse=True)
    idcg = _compute_dcg(ideal_relevances)

    if idcg == 0.0:
        raw = 1.0 if dcg == 0.0 else 0.0
        return _safe_score(raw)

    return _safe_score(dcg / idcg)


def compute_precision_at_k(predicted_ranking: List[str],
                           ground_truth: Dict[str, float],
                           k: int = 3) -> float:

    if not predicted_ranking or k <= 0:
        return _safe_score(0.0)

    k = min(k, len(predicted_ranking))

    relevant_count = sum(
        1 for doc_id in predicted_ranking[:k]
        if ground_truth.get(doc_id, 0.0) > 0.0
    )

    precision = relevant_count / k
    return _safe_score(precision)


def compute_mrr(predicted_ranking: List[str],
                ground_truth: Dict[str, float]) -> float:

    if not predicted_ranking:
        return _safe_score(0.0)

    for i, doc_id in enumerate(predicted_ranking):
        if ground_truth.get(doc_id, 0.0) > 0.0:
            return _safe_score(1.0 / (i + 1))

    return _safe_score(0.0)


# ---------------------------------------------------------------------------
# Composite grader
# ---------------------------------------------------------------------------

def grade(predicted_ranking: List[str],
          ground_truth: Dict[str, float],
          k: int = 3) -> GraderResult:

    ndcg = compute_ndcg(predicted_ranking, ground_truth)
    p_at_k = compute_precision_at_k(predicted_ranking, ground_truth, k=k)
    mrr = compute_mrr(predicted_ranking, ground_truth)

    # Round first
    ndcg = round(ndcg, 6)
    p_at_k = round(p_at_k, 6)
    mrr = round(mrr, 6)

    # Then clamp (CRITICAL ORDER)
    ndcg = _safe_score(ndcg)
    p_at_k = _safe_score(p_at_k)
    mrr = _safe_score(mrr)

    return GraderResult(
        score=ndcg,
        ndcg=ndcg,
        precision_at_k=p_at_k,
        mrr=mrr,
    )