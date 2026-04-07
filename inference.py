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
import time

from pathlib import Path
from typing import List, Optional, Tuple

from openai import OpenAI
from server.env import SearchRankingEnv
from server.models import Action, Observation

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


def _load_dotenv_fallback(path: str) -> None:
    """Load a minimal .env file if python-dotenv isn't available."""
    if not os.path.exists(path):
        return
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip().lstrip("\ufeff")
                    value = value.strip()
                    if key and value:
                        os.environ.setdefault(key, value)
    except Exception:
        pass


def _load_env() -> None:
    env_path = Path(__file__).parent / ".env"
    if load_dotenv is not None:
        load_dotenv(dotenv_path=env_path)

    _load_dotenv_fallback(str(env_path))


_load_env()

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "openai/gpt-4o-mini")
API_KEY: str = (
    os.environ.get("OPENROUTER_API_KEY")
    or os.environ.get("OPENAI_API_KEY")
    or os.environ.get("HF_TOKEN")
    or os.environ.get("API_KEY", "")
)

# --- Environment Configuration ---
REMOTE_ENV_URL: Optional[str] = os.environ.get("REMOTE_ENV_URL")
TASK_NAME: str = os.environ.get("SEARCH_RANKING_TASK") or os.environ.get("TASK_NAME", "all")
BENCHMARK: str = os.environ.get("SEARCH_RANKING_BENCHMARK", "search-ranking-env")
SUCCESS_SCORE_THRESHOLD: float = float(os.environ.get("SUCCESS_SCORE_THRESHOLD", "0.5"))

SEED: int = 42                              # fixed seed for reproducibility
MAX_RETRIES: int = 1                        # retry once on transient failure
RETRY_DELAY: float = 2.0                    # seconds between retries


# ---------------------------------------------------------------------------
# Strict-format logging
# ---------------------------------------------------------------------------

def _clamp_open_interval(x: float) -> float:
    """Clamp all scores strictly within [0.1, 0.85]."""
    if x <= 0.1:
        return 0.1
    if x >= 0.85:
        return 0.85
    return x

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    reward = _clamp_open_interval(reward)
    print(
        f"[STEP] step={step} action={action} reward={reward:.6f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{_clamp_open_interval(r):.6f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

def get_client() -> OpenAI | None:
    """Configure and return an OpenAI client."""
    api_key = (
        os.environ.get("OPENROUTER_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or os.environ.get("HF_TOKEN")
    )
    if not api_key:
        _load_env()
        api_key = (
            os.environ.get("OPENROUTER_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("HF_TOKEN")
        )

    if os.environ.get("DEBUG", "").strip() in {"1", "true", "yes"}:
        print(f"DEBUG: API Key loaded: {bool(api_key)}", file=sys.stderr)

    if not api_key:
        print(
            "WARNING: OPENROUTER_API_KEY (or OPENAI_API_KEY/HF_TOKEN) not set",
            file=sys.stderr,
        )
        return None

    return OpenAI(
        base_url=os.environ.get("API_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
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
        f"4. Do NOT wrap the JSON in code fences or add any other text.\n\n"
        f"EXAMPLE OUTPUT FORMAT:\n"
        f"{example_ids}\n\n"
        f"Your ranking:"
    )
    return prompt


def build_retry_prompt(valid_ids: List[str]) -> str:
    """Build a strict retry prompt when the model returns invalid JSON."""
    return (
        "Your previous response was invalid. "
        "Return ONLY a JSON array containing each of these IDs exactly once, "
        "with no extra text and no code fences. "
        f"Valid IDs: {json.dumps(valid_ids)}"
    )


# ---------------------------------------------------------------------------
# Response Parsing (strict JSON)
# ---------------------------------------------------------------------------

def parse_ranking(response_text: str, valid_ids: List[str]) -> Tuple[Optional[List[str]], Optional[str]]:
    """
    Parse the LLM response into an ordered list of document IDs.

    Requires the full response to be a JSON array of strings containing
    each valid ID exactly once.

    Args:
        response_text: Raw text from the LLM.
        valid_ids:     List of valid document IDs (used for validation
                       and as the deterministic fallback order).

    Returns:
        (ranking, error_message). ranking is None when invalid.
    """
    valid_set = set(valid_ids)
    n = len(valid_ids)

    try:
        parsed = json.loads(response_text.strip())
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        return None, f"Response is not valid JSON: {exc}"

    if not isinstance(parsed, list):
        return None, "Response JSON is not an array."

    if len(parsed) != n:
        return None, f"Response array length {len(parsed)} does not match expected {n}."

    if not all(isinstance(x, str) for x in parsed):
        return None, "Response array must contain only strings."

    if set(parsed) != valid_set:
        return None, "Response array must contain exactly the provided document IDs."

    return parsed, None


# ---------------------------------------------------------------------------
# LLM Ranking Call (with retry)
# ---------------------------------------------------------------------------

def get_llm_ranking(client: OpenAI | None, observation: Observation) -> List[str]:
    """
    Send the observation to the LLM and return a ranked list of document IDs.

    Retries once on transient failure; falls back to the deterministic
    document order if all attempts fail.
    """
    valid_ids = [doc.id for doc in observation.documents]
    prompt = build_prompt(observation)
    messages = [
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
    ]

    if client is None:
        return list(valid_ids)

    last_error = None
    for attempt in range(1 + MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )

            response_text = (response.choices[0].message.content or "").strip()
            ranking, error = parse_ranking(response_text, valid_ids)
            if ranking is not None:
                return ranking

            last_error = error or "Invalid JSON response"
            print(
                f"WARNING: Invalid LLM response (attempt {attempt + 1}): {last_error}",
                file=sys.stderr,
            )
            print("RAW_RESPONSE_BEGIN", file=sys.stderr)
            print(response_text, file=sys.stderr)
            print("RAW_RESPONSE_END", file=sys.stderr)

            if attempt < MAX_RETRIES:
                messages.extend(
                    [
                        {"role": "assistant", "content": response_text},
                        {"role": "user", "content": build_retry_prompt(valid_ids)},
                    ]
                )
                time.sleep(RETRY_DELAY)
                continue

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
    """Run OpenEnv episodes and print strict-format logs."""
    env = SearchRankingEnv(seed=SEED)
    client = get_client()

    task_name = TASK_NAME.lower().strip()
    if task_name == "all":
        tasks = ["easy", "medium", "hard"]
    else:
        tasks = [task_name]

    try:
        for task in tasks:
            rewards: List[float] = []
            steps_taken = 0
            success = False

            log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

            try:
                observation = env.reset(task)

                ranking = get_llm_ranking(client, observation)
                action = Action(ranking=ranking)
                _, reward, done, _info = env.step(action)

                steps_taken = 1
                rewards.append(reward.score)

                action_str = json.dumps(ranking, separators=(",", ":"))
                log_step(step=1, action=action_str, reward=reward.score, done=done, error=None)

                success = reward.score >= SUCCESS_SCORE_THRESHOLD

            except Exception as exc:
                print(f"ERROR: Inference failed for task '{task}': {exc}", file=sys.stderr)

            finally:
                log_end(success=success, steps=steps_taken, rewards=rewards)

    finally:
        close_fn = getattr(env, "close", None)
        if callable(close_fn):
            try:
                close_fn()
            except Exception as exc:
                print(f"WARNING: env.close failed: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()
