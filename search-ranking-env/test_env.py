"""
Tests for SearchRankingEnv (env.py).
Covers: reset, state, step, reward range, perfect/invalid actions, all difficulties.
"""

from env import SearchRankingEnv
from models import Action, Observation, Reward, Info


def test_reset_returns_observation():
    env = SearchRankingEnv(seed=42)
    obs = env.reset("easy")
    assert isinstance(obs, Observation), "reset() must return an Observation"
    assert obs.query, "query must be non-empty"
    assert 3 <= len(obs.documents) <= 5, f"easy should have 3–5 docs, got {len(obs.documents)}"
    print("[PASS] reset() returns valid Observation (easy)")


def test_state_matches_reset():
    env = SearchRankingEnv(seed=42)
    obs1 = env.reset("easy")
    obs2 = env.state()
    assert obs1 == obs2, "state() must match the last reset()"
    print("[PASS] state() matches reset()")


def test_step_returns_valid_tuple():
    env = SearchRankingEnv(seed=42)
    obs = env.reset("medium")
    action = Action(ranking=[doc.id for doc in obs.documents])
    result = env.step(action)
    assert len(result) == 4, "step() must return a 4-tuple"
    obs_out, reward, done, info = result
    assert isinstance(obs_out, Observation)
    assert isinstance(reward, Reward)
    assert isinstance(done, bool)
    assert isinstance(info, Info)
    print("[PASS] step() returns valid (Observation, Reward, bool, Info)")


def test_reward_in_range():
    env = SearchRankingEnv(seed=42)
    for diff in ("easy", "medium", "hard"):
        obs = env.reset(diff)
        action = Action(ranking=[doc.id for doc in obs.documents])
        _, reward, _, _ = env.step(action)
        assert 0.0 <= reward.score <= 1.0, f"reward out of range for {diff}: {reward.score}"
    print("[PASS] reward ∈ [0, 1] for all difficulties")


def test_perfect_ranking():
    env = SearchRankingEnv(seed=42)
    for diff in ("easy", "medium", "hard"):
        obs = env.reset(diff)
        # Sort by hidden relevance (descending) to form the ideal ranking
        ideal = sorted(obs.documents, key=lambda d: d.relevance, reverse=True)
        action = Action(ranking=[d.id for d in ideal])
        _, reward, done, info = env.step(action)
        assert done is True, "done must be True after step"
        assert abs(reward.score - 1.0) < 1e-5, (
            f"perfect ranking must yield 1.0, got {reward.score} ({diff})"
        )
        assert abs(info.ndcg - 1.0) < 1e-5
    print("[PASS] perfect ranking → reward = 1.0 for all difficulties")


def test_invalid_action_missing_ids():
    env = SearchRankingEnv(seed=42)
    obs = env.reset("easy")
    # Submit only the first doc — missing the rest
    action = Action(ranking=[obs.documents[0].id])
    _, reward, done, info = env.step(action)
    assert reward.score == 0.0, f"invalid action should yield 0.0, got {reward.score}"
    assert done is True
    print("[PASS] invalid action (missing IDs) → reward = 0.0")


def test_invalid_action_extra_ids():
    env = SearchRankingEnv(seed=42)
    obs = env.reset("easy")
    ids = [doc.id for doc in obs.documents] + ["bogus_id"]
    action = Action(ranking=ids)
    _, reward, done, _ = env.step(action)
    assert reward.score == 0.0
    assert done is True
    print("[PASS] invalid action (extra IDs) → reward = 0.0")


def test_difficulty_document_counts():
    env = SearchRankingEnv(seed=42)
    ranges = {"easy": (3, 5), "medium": (5, 10), "hard": (10, 15)}
    for diff, (lo, hi) in ranges.items():
        obs = env.reset(diff)
        n = len(obs.documents)
        assert lo <= n <= hi, (
            f"{diff} expected {lo}–{hi} docs, got {n}"
        )
    print("[PASS] document counts match per difficulty")


def test_invalid_difficulty():
    env = SearchRankingEnv(seed=42)
    try:
        env.reset("nightmare")
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("[PASS] invalid difficulty raises ValueError")


def test_reproducibility():
    env1 = SearchRankingEnv(seed=99)
    env2 = SearchRankingEnv(seed=99)
    obs1 = env1.reset("hard")
    obs2 = env2.reset("hard")
    assert obs1 == obs2, "Same seed must produce identical observations"
    print("[PASS] reproducibility with same seed")


if __name__ == "__main__":
    test_reset_returns_observation()
    test_state_matches_reset()
    test_step_returns_valid_tuple()
    test_reward_in_range()
    test_perfect_ranking()
    test_invalid_action_missing_ids()
    test_invalid_action_extra_ids()
    test_difficulty_document_counts()
    test_invalid_difficulty()
    test_reproducibility()
    print("\n✅ All tests passed.")
