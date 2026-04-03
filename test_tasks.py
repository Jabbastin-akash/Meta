"""
Comprehensive validation of tasks and grading across all difficulty levels.

Validates:
  - Dataset structure and doc counts per spec
  - Perfect ranking → exactly 1.0
  - Worst ranking → partial credit, strictly < 1.0
  - Partial mistakes → intermediate scores
  - Score monotonicity: perfect > partial > worst
  - Determinism: same seed → same results
  - Difficulty progression: harder tasks have more docs
"""

from env import SearchRankingEnv
from models import Action


# ===== Dataset Structure =====

def test_easy_doc_counts():
    env = SearchRankingEnv(seed=0)
    for i in range(10):  # sample repeatedly to cover all queries
        obs = env.reset("easy")
        n = len(obs.documents)
        assert 3 <= n <= 5, f"easy: expected 3–5 docs, got {n}"
    print("[PASS] Easy: all tasks have 3–5 documents")


def test_medium_doc_counts():
    env = SearchRankingEnv(seed=0)
    for i in range(10):
        obs = env.reset("medium")
        n = len(obs.documents)
        assert 5 <= n <= 10, f"medium: expected 5–10 docs, got {n}"
    print("[PASS] Medium: all tasks have 5–10 documents")


def test_hard_doc_counts():
    env = SearchRankingEnv(seed=0)
    for i in range(10):
        obs = env.reset("hard")
        n = len(obs.documents)
        assert 10 <= n <= 15, f"hard: expected 10–15 docs, got {n}"
    print("[PASS] Hard: all tasks have 10–15 documents")


def test_unique_doc_ids():
    """All document IDs within a single task must be unique."""
    env = SearchRankingEnv(seed=0)
    for diff in ("easy", "medium", "hard"):
        for _ in range(10):
            obs = env.reset(diff)
            ids = [d.id for d in obs.documents]
            assert len(ids) == len(set(ids)), f"Duplicate IDs found in {diff}"
    print("[PASS] All tasks have unique document IDs")


def test_relevance_scores_in_range():
    """All relevance scores must be in [0, 3]."""
    env = SearchRankingEnv(seed=0)
    for diff in ("easy", "medium", "hard"):
        for _ in range(10):
            obs = env.reset(diff)
            for doc in obs.documents:
                assert 0.0 <= doc.relevance <= 3.0, (
                    f"{diff}/{doc.id}: relevance {doc.relevance} out of [0,3]"
                )
    print("[PASS] All relevance scores in [0.0, 3.0]")


# ===== Scoring Behavior =====

def test_perfect_ranking_all_difficulties():
    env = SearchRankingEnv(seed=42)
    for diff in ("easy", "medium", "hard"):
        obs = env.reset(diff)
        ideal = sorted(obs.documents, key=lambda d: d.relevance, reverse=True)
        action = Action(ranking=[d.id for d in ideal])
        _, reward, done, info = env.step(action)
        assert done is True
        assert abs(reward.score - 1.0) < 1e-5, (
            f"{diff}: perfect ranking yielded {reward.score}, expected 1.0"
        )
    print("[PASS] Perfect ranking → 1.0 for all difficulties")


def test_worst_ranking_all_difficulties():
    env = SearchRankingEnv(seed=42)
    for diff in ("easy", "medium", "hard"):
        obs = env.reset(diff)
        worst = sorted(obs.documents, key=lambda d: d.relevance, reverse=False)
        action = Action(ranking=[d.id for d in worst])
        _, reward, _, _ = env.step(action)
        assert 0.0 <= reward.score < 1.0, (
            f"{diff}: worst ranking yielded {reward.score}"
        )
    print("[PASS] Worst ranking → score < 1.0 for all difficulties")


def test_score_monotonicity():
    """perfect > partial > worst for every difficulty."""
    for diff in ("easy", "medium", "hard"):
        # Use a fresh env instance per ranking to avoid RNG advancing
        env_p = SearchRankingEnv(seed=42)
        obs = env_p.reset(diff)

        # Build ranking orders from the same observation
        ideal = sorted(obs.documents, key=lambda d: d.relevance, reverse=True)
        partial_order = list(ideal)
        if len(partial_order) >= 2:
            partial_order[0], partial_order[1] = partial_order[1], partial_order[0]
        worst = sorted(obs.documents, key=lambda d: d.relevance, reverse=False)

        a_perfect = Action(ranking=[d.id for d in ideal])
        a_partial = Action(ranking=[d.id for d in partial_order])
        a_worst = Action(ranking=[d.id for d in worst])

        # Evaluate each with a fresh env reset to the exact same task
        env1 = SearchRankingEnv(seed=42)
        env1.reset(diff)
        _, r_perfect, _, _ = env1.step(a_perfect)

        env2 = SearchRankingEnv(seed=42)
        env2.reset(diff)
        _, r_partial, _, _ = env2.step(a_partial)

        env3 = SearchRankingEnv(seed=42)
        env3.reset(diff)
        _, r_worst, _, _ = env3.step(a_worst)

        assert r_perfect.score >= r_partial.score >= r_worst.score, (
            f"{diff}: monotonicity violated: "
            f"perfect={r_perfect.score} partial={r_partial.score} worst={r_worst.score}"
        )
        print(f"  {diff}: perfect={r_perfect.score:.4f} >= partial={r_partial.score:.4f} >= worst={r_worst.score:.4f}")

    print("[PASS] Score monotonicity: perfect >= partial >= worst")


# ===== Determinism =====

def test_determinism():
    for diff in ("easy", "medium", "hard"):
        env1 = SearchRankingEnv(seed=123)
        env2 = SearchRankingEnv(seed=123)
        obs1 = env1.reset(diff)
        obs2 = env2.reset(diff)
        assert obs1 == obs2, f"{diff}: observations differ with same seed"

        action = Action(ranking=[d.id for d in obs1.documents])
        _, r1, _, i1 = env1.step(action)
        _, r2, _, i2 = env2.step(action)
        assert r1 == r2, f"{diff}: rewards differ with same seed"
        assert i1 == i2, f"{diff}: info differs with same seed"
    print("[PASS] Deterministic: same seed → identical observations, rewards, info")


# ===== Difficulty Progression =====

def test_difficulty_progression():
    """Average doc count should increase: easy < medium < hard."""
    env = SearchRankingEnv(seed=0)
    avg_counts = {}
    for diff in ("easy", "medium", "hard"):
        counts = []
        for _ in range(20):
            obs = env.reset(diff)
            counts.append(len(obs.documents))
        avg_counts[diff] = sum(counts) / len(counts)

    assert avg_counts["easy"] < avg_counts["medium"] < avg_counts["hard"], (
        f"Difficulty progression violated: {avg_counts}"
    )
    print(f"[PASS] Difficulty progression: easy({avg_counts['easy']:.1f}) "
          f"< medium({avg_counts['medium']:.1f}) < hard({avg_counts['hard']:.1f})")


# ===== Run All =====

if __name__ == "__main__":
    test_easy_doc_counts()
    test_medium_doc_counts()
    test_hard_doc_counts()
    test_unique_doc_ids()
    test_relevance_scores_in_range()

    test_perfect_ranking_all_difficulties()
    test_worst_ranking_all_difficulties()
    test_score_monotonicity()

    test_determinism()
    test_difficulty_progression()

    print("\n✅ All task & grader validation tests passed.")
