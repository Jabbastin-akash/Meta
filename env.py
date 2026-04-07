"""Compatibility re-export for the environment.

Some tooling/tests expect `from env import SearchRankingEnv`.
The implementation lives in `server.env`.
"""

from server.env import SearchRankingEnv  # noqa: F401

__all__ = ["SearchRankingEnv"]
