"""
Search Ranking Environment — OpenEnv-compatible.

Manages state, validates actions, computes rewards via grading functions,
and returns structured outputs using Pydantic models.
"""

import random
from typing import Tuple, Optional

from models import Document, Observation, Action, Reward, Info
from grader import grade
from tasks.easy import TASKS as EASY_TASKS
from tasks.medium import TASKS as MEDIUM_TASKS
from tasks.hard import TASKS as HARD_TASKS


TASK_REGISTRY = {
    "easy": EASY_TASKS,
    "medium": MEDIUM_TASKS,
    "hard": HARD_TASKS,
}


class SearchRankingEnv:
    """
    OpenEnv-compatible search ranking environment.

    Interface:
        reset(difficulty)  → Observation
        state()            → Observation
        step(action)       → (Observation, Reward, bool, Info)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Optional random seed for reproducible task selection.
        """
        self._rng = random.Random(seed)
        self._current_query: Optional[str] = None
        self._current_documents: list[Document] = []
        self._ground_truth: dict[str, float] = {}
        self._current_step: int = 0
        self._max_steps: int = 1
        self._difficulty: str = "easy"

    # ------------------------------------------------------------------
    # OpenEnv interface
    # ------------------------------------------------------------------

    def reset(self, difficulty: str = "easy") -> Observation:
        """
        Initialize a new query-document episode.

        Args:
            difficulty: One of "easy", "medium", "hard".

        Returns:
            Observation with the query and shuffled candidate documents.

        Raises:
            ValueError: If difficulty is not recognized.
        """
        difficulty = difficulty.lower()
        if difficulty not in TASK_REGISTRY:
            raise ValueError(
                f"Invalid difficulty '{difficulty}'. "
                f"Choose from: {list(TASK_REGISTRY.keys())}"
            )

        self._difficulty = difficulty
        self._current_step = 0

        # Pick a task from the appropriate pool
        tasks = TASK_REGISTRY[difficulty]
        task = self._rng.choice(tasks)

        # Build Document models and ground-truth map
        self._current_query = task["query"]
        docs = [
            Document(id=d["id"], text=d["text"], relevance=d["relevance"])
            for d in task["documents"]
        ]

        # Shuffle so the agent cannot rely on ordering
        self._rng.shuffle(docs)

        self._current_documents = docs
        self._ground_truth = {doc.id: doc.relevance for doc in docs}

        return self.state()

    def state(self) -> Observation:
        """Return the current observation."""
        return Observation(
            query=self._current_query,
            documents=self._current_documents,
        )

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Info]:
        """
        Evaluate the agent's ranking and return structured results.

        Args:
            action: An Action containing the agent's ordered document IDs.

        Returns:
            A 4-tuple of (Observation, Reward, done, Info).
        """
        self._current_step += 1
        obs = self.state()

        # --- Step 1: Validate action ---
        try:
            action.validate_against_observation(obs)
        except ValueError:
            return (
                obs,
                Reward(score=0.0),
                True,
                Info(ndcg=0.0, precision_at_k=0.0, mrr=0.0),
            )

        # --- Step 2: Compute reward via grader ---
        result = grade(action.ranking, self._ground_truth, k=3)

        reward = Reward(score=result.score)
        info = Info(
            ndcg=result.ndcg,
            precision_at_k=result.precision_at_k,
            mrr=result.mrr,
        )

        # --- Step 3: Single-step episode ---
        done = True

        return obs, reward, done, info
