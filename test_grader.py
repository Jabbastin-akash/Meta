"""
Comprehensive tests for grader.py.
Covers: NDCG, Precision@K, MRR, composite grade(), edge cases, determinism.
"""

from grader import compute_ndcg, compute_precision_at_k, compute_mrr, grade


# --- Ground truth fixtures ---

GROUND_TRUTH = {
    "d1": 3.0,
    "d2": 2.0,
    "d3": 1.0,
    "d4": 0.0,
    "d5": 0.0,
}

IDEAL_RANKING = ["d1", "d2", "d3", "d4", "d5"]
WORST_RANKING = ["d4", "d5", "d3", "d2", "d1"]


# ===== NDCG Tests =====

def test_ndcg_perfect():
    score = compute_ndcg(IDEAL_RANKING, GROUND_TRUTH)
    assert abs(score - 1.0) < 1e-6, f"Expected 1.0, got {score}"
    print("[PASS] NDCG: perfect ranking → 1.0")


def test_ndcg_worst():
    score = compute_ndcg(WORST_RANKING, GROUND_TRUTH)
    assert 0.0 < score < 1.0, f"Expected intermediate, got {score}"
    print(f"[PASS] NDCG: worst ranking → {score:.6f} (partial credit)")


def test_ndcg_partial():
    partial = ["d2", "d1", "d3", "d4", "d5"]  # swapped top 2
    score = compute_ndcg(partial, GROUND_TRUTH)
    assert 0.8 < score < 1.0, f"Expected near-perfect, got {score}"
    print(f"[PASS] NDCG: near-perfect ranking → {score:.6f}")


def test_ndcg_empty_ranking():
    score = compute_ndcg([], GROUND_TRUTH)
    assert score == 0.0, f"Expected 0.0 for empty ranking, got {score}"
    print("[PASS] NDCG: empty ranking → 0.0")


def test_ndcg_missing_ids():
    ranking = ["d1", "unknown_1", "unknown_2", "d4", "d5"]
    score = compute_ndcg(ranking, GROUND_TRUTH)
    assert 0.0 < score < 1.0, f"Expected degraded score, got {score}"
    print(f"[PASS] NDCG: missing IDs treated as 0.0 relevance → {score:.6f}")


def test_ndcg_all_zero_relevance():
    gt_zero = {"d1": 0.0, "d2": 0.0, "d3": 0.0}
    score = compute_ndcg(["d1", "d2", "d3"], gt_zero)
    assert score == 1.0, f"Expected 1.0 when all relevance is 0, got {score}"
    print("[PASS] NDCG: all-zero relevance → 1.0")


def test_ndcg_deterministic():
    s1 = compute_ndcg(IDEAL_RANKING, GROUND_TRUTH)
    s2 = compute_ndcg(IDEAL_RANKING, GROUND_TRUTH)
    assert s1 == s2, "NDCG must be deterministic"
    print("[PASS] NDCG: deterministic across calls")


# ===== Precision@K Tests =====

def test_precision_at_k_all_relevant():
    score = compute_precision_at_k(["d1", "d2", "d3"], GROUND_TRUTH, k=3)
    assert score == 1.0, f"Expected 1.0, got {score}"
    print("[PASS] Precision@3: all top-3 relevant → 1.0")


def test_precision_at_k_none_relevant():
    score = compute_precision_at_k(["d4", "d5", "d3"], GROUND_TRUTH, k=2)
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("[PASS] Precision@2: no relevant in top-2 → 0.0")


def test_precision_at_k_partial():
    score = compute_precision_at_k(["d1", "d4", "d3"], GROUND_TRUTH, k=3)
    assert abs(score - 2.0 / 3.0) < 1e-6, f"Expected 0.667, got {score}"
    print(f"[PASS] Precision@3: 2/3 relevant → {score:.6f}")


def test_precision_at_k_empty():
    score = compute_precision_at_k([], GROUND_TRUTH, k=3)
    assert score == 0.0
    print("[PASS] Precision@K: empty ranking → 0.0")


def test_precision_at_k_k_larger_than_list():
    score = compute_precision_at_k(["d1", "d2"], GROUND_TRUTH, k=10)
    assert score == 1.0, f"Expected 1.0, got {score}"
    print("[PASS] Precision@K: k > len(ranking) → uses full list")


# ===== MRR Tests =====

def test_mrr_first_is_relevant():
    score = compute_mrr(["d1", "d4", "d5"], GROUND_TRUTH)
    assert score == 1.0, f"Expected 1.0, got {score}"
    print("[PASS] MRR: first doc relevant → 1.0")


def test_mrr_second_is_relevant():
    score = compute_mrr(["d4", "d1", "d5"], GROUND_TRUTH)
    assert score == 0.5, f"Expected 0.5, got {score}"
    print("[PASS] MRR: second doc relevant → 0.5")


def test_mrr_none_relevant():
    score = compute_mrr(["d4", "d5"], GROUND_TRUTH)
    assert score == 0.0, f"Expected 0.0, got {score}"
    print("[PASS] MRR: no relevant docs → 0.0")


def test_mrr_empty():
    score = compute_mrr([], GROUND_TRUTH)
    assert score == 0.0
    print("[PASS] MRR: empty ranking → 0.0")


# ===== Composite grade() Tests =====

def test_grade_returns_all_metrics():
    result = grade(IDEAL_RANKING, GROUND_TRUTH, k=3)
    assert hasattr(result, "score")
    assert hasattr(result, "ndcg")
    assert hasattr(result, "precision_at_k")
    assert hasattr(result, "mrr")
    assert abs(result.score - 1.0) < 1e-5
    assert abs(result.ndcg - 1.0) < 1e-5
    assert result.precision_at_k == 1.0
    assert result.mrr == 1.0
    print("[PASS] grade(): perfect ranking returns all metrics correctly")


def test_grade_deterministic():
    r1 = grade(WORST_RANKING, GROUND_TRUTH)
    r2 = grade(WORST_RANKING, GROUND_TRUTH)
    assert r1 == r2, "grade() must be deterministic"
    print("[PASS] grade(): deterministic across calls")


if __name__ == "__main__":
    test_ndcg_perfect()
    test_ndcg_worst()
    test_ndcg_partial()
    test_ndcg_empty_ranking()
    test_ndcg_missing_ids()
    test_ndcg_all_zero_relevance()
    test_ndcg_deterministic()

    test_precision_at_k_all_relevant()
    test_precision_at_k_none_relevant()
    test_precision_at_k_partial()
    test_precision_at_k_empty()
    test_precision_at_k_k_larger_than_list()

    test_mrr_first_is_relevant()
    test_mrr_second_is_relevant()
    test_mrr_none_relevant()
    test_mrr_empty()

    test_grade_returns_all_metrics()
    test_grade_deterministic()

    print("\n✅ All grader tests passed.")
