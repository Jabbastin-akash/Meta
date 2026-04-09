"""
Test script: runs inference with a dummy LLM against all tasks,
prints every score, and asserts 0 < score < 1.
"""
import sys
import os

# Ensure server/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "server"))

from server.env import SearchRankingEnv
from server.models import Action
from server.grader import grade, _clamp_strict_0_1

SEED = 42

def test_all_tasks():
    env = SearchRankingEnv(seed=SEED)
    difficulties = ["easy", "medium", "hard"]
    all_passed = True

    for diff in difficulties:
        print(f"\n{'='*60}")
        print(f"  TASK: {diff}")
        print(f"{'='*60}")

        obs = env.reset(diff)
        doc_ids = [doc.id for doc in obs.documents]
        print(f"  Query: {obs.query}")
        print(f"  Docs:  {doc_ids}")

        # === Test 1: Perfect ranking (sorted by relevance desc) ===
        perfect = sorted(doc_ids, key=lambda d: next(
            doc.relevance for doc in obs.documents if doc.id == d
        ), reverse=True)
        action = Action(ranking=perfect)
        _, reward, done, info = env.step(action)
        print(f"\n  [Perfect ranking]")
        print(f"    reward.score     = {reward.score}")
        print(f"    info.ndcg        = {info.ndcg}")
        print(f"    info.precision   = {info.precision_at_k}")
        print(f"    info.mrr         = {info.mrr}")
        for name, val in [("reward", reward.score), ("ndcg", info.ndcg),
                          ("precision", info.precision_at_k), ("mrr", info.mrr)]:
            if not (0.0 < val < 1.0):
                print(f"    ❌ FAIL: {name} = {val} is NOT strictly between 0 and 1!")
                all_passed = False
            else:
                print(f"    ✅ OK:   {name} = {val}")

        # === Test 2: Worst ranking (reversed) ===
        obs = env.reset(diff)
        doc_ids = [doc.id for doc in obs.documents]
        worst = sorted(doc_ids, key=lambda d: next(
            doc.relevance for doc in obs.documents if doc.id == d
        ), reverse=False)
        action = Action(ranking=worst)
        _, reward, done, info = env.step(action)
        print(f"\n  [Worst ranking]")
        print(f"    reward.score     = {reward.score}")
        print(f"    info.ndcg        = {info.ndcg}")
        print(f"    info.precision   = {info.precision_at_k}")
        print(f"    info.mrr         = {info.mrr}")
        for name, val in [("reward", reward.score), ("ndcg", info.ndcg),
                          ("precision", info.precision_at_k), ("mrr", info.mrr)]:
            if not (0.0 < val < 1.0):
                print(f"    ❌ FAIL: {name} = {val} is NOT strictly between 0 and 1!")
                all_passed = False
            else:
                print(f"    ✅ OK:   {name} = {val}")

        # === Test 3: Random / as-is ranking ===
        obs = env.reset(diff)
        doc_ids = [doc.id for doc in obs.documents]
        action = Action(ranking=doc_ids)  # whatever shuffle env gave
        _, reward, done, info = env.step(action)
        print(f"\n  [Random ranking]")
        print(f"    reward.score     = {reward.score}")
        print(f"    info.ndcg        = {info.ndcg}")
        print(f"    info.precision   = {info.precision_at_k}")
        print(f"    info.mrr         = {info.mrr}")
        for name, val in [("reward", reward.score), ("ndcg", info.ndcg),
                          ("precision", info.precision_at_k), ("mrr", info.mrr)]:
            if not (0.0 < val < 1.0):
                print(f"    ❌ FAIL: {name} = {val} is NOT strictly between 0 and 1!")
                all_passed = False
            else:
                print(f"    ✅ OK:   {name} = {val}")

    # === Test 4: Edge case - clamp function directly ===
    print(f"\n{'='*60}")
    print(f"  EDGE CASE: _clamp_strict_0_1 direct tests")
    print(f"{'='*60}")
    edge_cases = [0.0, 1.0, -1.0, 2.0, 0.5, 1e-10, 1.0 - 1e-10, float('nan'), float('inf')]
    for raw in edge_cases:
        clamped = _clamp_strict_0_1(raw)
        ok = 0.0 < clamped < 1.0
        tag = "✅" if ok else "❌"
        print(f"    {tag} _clamp({raw}) = {clamped}")
        if not ok:
            all_passed = False

    print(f"\n{'='*60}")
    if all_passed:
        print("  ✅ ALL SCORES STRICTLY BETWEEN 0 AND 1 — PASSED!")
    else:
        print("  ❌ SOME SCORES FAILED — SEE ABOVE")
    print(f"{'='*60}\n")
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    test_all_tasks()
