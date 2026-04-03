"""
Baseline inference script for the Search Ranking Environment.

Interacts with an LLM via the OpenAI client configured against an
OpenAI-compatible endpoint. Logs single-line [START]/[STEP]/[END]
records required by OpenEnv automated grading.

Required environment variables
------------------------------
    API_BASE_URL         - LLM endpoint (default: https://openrouter.ai/api/v1)
    MODEL_NAME           - model identifier (default: openai/gpt-4o-mini)
    OPENROUTER_API_KEY   - API key (falls back to OPENAI_API_KEY or HF_TOKEN)
"""

import os
import sys
import json

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import re
import time
from typing import List, Optional

from openai import OpenAI

from server.env import SearchRankingEnv
from server.models import Action, Observation


# ---------------------------------------------------------------------------
# Configuration (read from environment — NEVER hardcoded)
# ---------------------------------------------------------------------------

DEFAULT_API_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_MODEL_NAME = "openai/gpt-4o-mini"

API_BASE_URL: str = os.environ.get("API_BASE_URL", DEFAULT_API_BASE_URL)
MODEL_NAME: str = os.environ.get("MODEL_NAME", DEFAULT_MODEL_NAME)
API_KEY: str = (
    os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("API_KEY", "")
)

TASK_NAME: str = os.environ.get("SEARCH_RANKING_TASK") or os.environ.get("TASK_NAME", "easy")
BENCHMARK: str = os.environ.get("SEARCH_RANKING_BENCHMARK", "search-ranking-env")
SUCCESS_SCORE_THRESHOLD: float = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.5"))

SEED: int = 42                              # fixed seed for reproducibility
MAX_RETRIES: int = 1                        # retry once on transient failure
RETRY_DELAY: float = 2.0                    # seconds between retries


# ---------------------------------------------------------------------------
# Strict-format logging
# ---------------------------------------------------------------------------

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    """Create an OpenAI client pointing at the configured endpoint."""
    if not API_KEY:
        print("WARNING: OPENROUTER_API_KEY (or OPENAI_API_KEY/HF_TOKEN) not set", file=sys.stderr)

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=API_KEY,
    )


# ---------------------------------------------------------------------------
# Prompt Construction
# ---------------------------------------------------------------------------

def build_prompt(observation: Observation) -> str:
    """
    Build a clear, unambiguous prompt that asks the LLM to rank documents
    by relevance to the query and return ONLY a JSON array of document IDs.
    """
    doc_lines = []
    for doc in observation.documents:
        doc_lines.append(f"  - ID: \"{doc.id}\" | Content: \"{doc.text}\"")

    documents_block = "\n".join(doc_lines)

    # Collect all IDs for the example so the LLM sees the exact format
    all_ids = [doc.id for doc in observation.documents]
    example_ids = json.dumps(all_ids[:3]) if len(all_ids) >= 3 else json.dumps(all_ids)

    prompt = (
        f"You are a search-ranking expert.\n\n"
        f"QUERY: \"{observation.query}\"\n\n"
        f"CANDIDATE DOCUMENTS:\n{documents_block}\n\n"
        f"TASK:\n"
        f"Rank ALL documents from MOST relevant to LEAST relevant to the query.\n\n"
        f"RULES:\n"
        f"1. Return ONLY a JSON array of document IDs, nothing else.\n"
        f"2. Include every document ID exactly once.\n"
        f"3. Do NOT include any explanation, markdown, or extra text.\n\n"
        f"EXAMPLE OUTPUT FORMAT:\n"
        f"{example_ids}\n\n"
        f"Your ranking:"
    )
    return prompt


# ---------------------------------------------------------------------------
# Response Parsing (multi-strategy, deterministic)
# ---------------------------------------------------------------------------

def parse_ranking(response_text: str, valid_ids: List[str]) -> List[str]:
    """
    Parse the LLM response into an ordered list of document IDs.

    Tries multiple extraction strategies in order:
      1. Direct JSON array parse
      2. JSON array inside markdown code fence
      3. JSON array anywhere in the text (regex)
      4. Line-by-line ID extraction
      5. Fallback: original document order (deterministic)

    Args:
        response_text: Raw text from the LLM.
        valid_ids:     List of valid document IDs (used for validation
                       and as the deterministic fallback order).

    Returns:
        Ordered list of document IDs.
    """
    valid_set = set(valid_ids)
    n = len(valid_ids)

    def _is_valid(ids: List[str]) -> bool:
        return (
            isinstance(ids, list)
            and len(ids) == n
            and all(isinstance(x, str) for x in ids)
            and set(ids) == valid_set
        )

    # --- Strategy 1: direct JSON parse of entire response ----------------
    try:
        parsed = json.loads(response_text.strip())
        if _is_valid(parsed):
            return parsed
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # --- Strategy 2: extract from markdown code fence --------------------
    fence_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", response_text, re.DOTALL)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1).strip())
            if _is_valid(parsed):
                return parsed
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # --- Strategy 3: find first JSON array anywhere ----------------------
    array_match = re.search(r"\[.*?\]", response_text, re.DOTALL)
    if array_match:
        try:
            parsed = json.loads(array_match.group())
            if _is_valid(parsed):
                return parsed
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # --- Strategy 4: line-by-line ID extraction --------------------------
    found_ids: List[str] = []
    for line in response_text.split("\n"):
        line = line.strip()
        for vid in valid_ids:
            if vid in line and vid not in found_ids:
                found_ids.append(vid)
    if _is_valid(found_ids):
        return found_ids

    # --- Strategy 5: deterministic fallback (original observation order) -
    print(
        "WARNING: Could not parse LLM response, using fallback order",
        file=sys.stderr,
    )
    return list(valid_ids)


# ---------------------------------------------------------------------------
# LLM Ranking Call (with retry)
# ---------------------------------------------------------------------------

def get_llm_ranking(client: OpenAI, observation: Observation) -> List[str]:
    """
    Send the observation to the LLM and return a ranked list of document IDs.

    Retries once on transient failure; falls back to the deterministic
    document order if all attempts fail.
    """
    valid_ids = [doc.id for doc in observation.documents]
    prompt = build_prompt(observation)

    last_error = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a search ranking assistant. "
                            "Respond ONLY with a JSON array of document IDs."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
                max_tokens=1024,
            )

            response_text = (response.choices[0].message.content or "").strip()
            return parse_ranking(response_text, valid_ids)

        except Exception as exc:
            last_error = exc
            print(
                f"WARNING: LLM call failed (attempt {attempt + 1}): {exc}",
                file=sys.stderr,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)

    # All attempts exhausted — deterministic fallback
    print(
        f"WARNING: All LLM attempts failed ({last_error}), using fallback order",
        file=sys.stderr,
    )
    return list(valid_ids)


# ---------------------------------------------------------------------------
# Main Execution
# ---------------------------------------------------------------------------

def main() -> None:
    """Run a single OpenEnv episode and print strict-format logs."""
    env = SearchRankingEnv(seed=SEED)
    client = get_client()

    rewards: List[float] = []
    steps_taken = 0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        observation = env.reset(TASK_NAME)

        ranking = get_llm_ranking(client, observation)
        action = Action(ranking=ranking)
        _, reward, done, _info = env.step(action)

        steps_taken = 1
        rewards.append(reward.score)

        action_str = json.dumps(ranking, separators=(",", ":"))
        log_step(step=1, action=action_str, reward=reward.score, done=done, error=None)

        success = reward.score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"ERROR: Inference failed: {exc}", file=sys.stderr)

    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                print(f"WARNING: env.close failed: {exc}", file=sys.stderr)

        log_end(success=success, steps=steps_taken, rewards=rewards)


if __name__ == "__main__":
    main()
