"""
Grading functions for the Search Ranking Environment.

Computes NDCG, Precision@K, and MRR from predicted rankings
against ground truth relevance scores.

Design principles:
  - Deterministic: identical inputs always produce identical outputs
  - No external ML libraries (pure math + stdlib)
  - Continuous scoring in [0.0, 1.0] with meaningful partial credit
  - Robust edge-case handling (empty inputs, missing IDs, duplicates)
"""

import math
from typing import List, Dict, NamedTuple


class GraderResult(NamedTuple):
    """Structured output from the grader."""
    score: float          # Primary reward (NDCG)
    ndcg: float           # Normalized Discounted Cumulative Gain
    precision_at_k: float # Precision@K
    mrr: float            # Mean Reciprocal Rank


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_dcg(relevances: List[float]) -> float:
    """
    Compute Discounted Cumulative Gain for a list of relevance scores
    ordered by predicted rank position.

    DCG = Σ (2^rel_i - 1) / log2(i + 2)   for i = 0 … n-1

    The denominator uses (i + 2) so that rank-1 maps to log2(2) = 1.
    """
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

    Args:
        predicted_ranking: Ordered list of document IDs (agent's ranking).
        ground_truth: Mapping of document ID → relevance score.

    Returns:
        NDCG score between 0.0 and 1.0.

    Edge cases:
        - Empty predicted_ranking → 0.0
        - Unknown doc IDs → treated as relevance 0.0
        - All ground-truth relevances are 0 → 1.0 (any ordering is "ideal")
    """
    if not predicted_ranking:
        return 0.0

    # DCG from agent's ranking (unknown IDs default to 0.0 relevance)
    predicted_relevances = [
        ground_truth.get(doc_id, 0.0) for doc_id in predicted_ranking
    ]
    dcg = _compute_dcg(predicted_relevances)

    # Ideal DCG (documents sorted by relevance descending)
    ideal_relevances = sorted(ground_truth.values(), reverse=True)
    idcg = _compute_dcg(ideal_relevances)

    if idcg == 0.0:
        # All relevances are zero — every ordering is equally "perfect"
        return 1.0 if dcg == 0.0 else 0.0

    return dcg / idcg


def compute_precision_at_k(predicted_ranking: List[str],
                           ground_truth: Dict[str, float],
                           k: int = 3) -> float:
    """
    Compute Precision@K — the fraction of the top-K ranked documents
    that are relevant (relevance > 0).

    Args:
        predicted_ranking: Ordered list of document IDs.
        ground_truth: Mapping of document ID → relevance score.
        k: Number of top positions to evaluate.

    Returns:
        Precision@K as a float between 0.0 and 1.0.

    Edge cases:
        - Empty ranking → 0.0
        - k larger than ranking length → uses full ranking
        - Unknown doc IDs → treated as non-relevant
    """
    if not predicted_ranking or k <= 0:
        return 0.0

    k = min(k, len(predicted_ranking))
    relevant_count = sum(
        1 for doc_id in predicted_ranking[:k]
        if ground_truth.get(doc_id, 0.0) > 0.0
    )
    return relevant_count / k


def compute_mrr(predicted_ranking: List[str],
                ground_truth: Dict[str, float]) -> float:
    """
    Compute Mean Reciprocal Rank — the reciprocal of the rank position
    of the first relevant document.

    Args:
        predicted_ranking: Ordered list of document IDs.
        ground_truth: Mapping of document ID → relevance score.

    Returns:
        MRR as a float between 0.0 and 1.0.

    Edge cases:
        - Empty ranking → 0.0
        - No relevant documents → 0.0
        - Unknown doc IDs → treated as non-relevant
    """
    if not predicted_ranking:
        return 0.0

    for i, doc_id in enumerate(predicted_ranking):
        if ground_truth.get(doc_id, 0.0) > 0.0:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Composite grader
# ---------------------------------------------------------------------------

def grade(predicted_ranking: List[str],
          ground_truth: Dict[str, float],
          k: int = 3) -> GraderResult:
    """
    Run the full grading pipeline and return all metrics at once.

    This is the primary entry point used by env.step().

    Args:
        predicted_ranking: Ordered list of document IDs.
        ground_truth: Mapping of document ID → relevance score.
        k: K value for Precision@K.

    Returns:
        GraderResult with score (primary NDCG reward) and auxiliary metrics.
    """
    ndcg = compute_ndcg(predicted_ranking, ground_truth)
    p_at_k = compute_precision_at_k(predicted_ranking, ground_truth, k=k)
    mrr = compute_mrr(predicted_ranking, ground_truth)

    return GraderResult(
        score=round(ndcg, 6),
        ndcg=round(ndcg, 6),
        precision_at_k=round(p_at_k, 6),
        mrr=round(mrr, 6),
    )
