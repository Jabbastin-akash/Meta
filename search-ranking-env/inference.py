"""
Baseline inference script for the Search Ranking Environment.

Interacts with an LLM via the OpenAI client configured against an
OpenRouter-compatible endpoint.  Runs every difficulty level (easy,
medium, hard), logs results in the strict format required by OpenEnv
automated grading.

Required environment variables
------------------------------
  API_BASE_URL   – LLM endpoint  (e.g. https://openrouter.ai/api/v1)
  MODEL_NAME     – model identifier
  OPENAI_API_KEY – API / OpenRouter key
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
from typing import List

from openai import OpenAI

from env import SearchRankingEnv
from models import Action, Observation


# ---------------------------------------------------------------------------
# Configuration (read from environment — NEVER hardcoded)
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "")
OPENAI_API_KEY: str = os.environ.get("OPENAI_API_KEY", "")

SEED: int = 42                              # fixed seed for reproducibility
DIFFICULTIES: List[str] = ["easy", "medium", "hard"]
MAX_RETRIES: int = 1                        # retry once on transient failure
RETRY_DELAY: float = 2.0                    # seconds between retries


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------

def get_client() -> OpenAI:
    """Create an OpenAI client pointing at the configured endpoint."""
    if not API_BASE_URL:
        print("WARNING: API_BASE_URL not set", file=sys.stderr)
    if not OPENAI_API_KEY:
        print("WARNING: OPENAI_API_KEY not set", file=sys.stderr)

    return OpenAI(
        base_url=API_BASE_URL,
        api_key=OPENAI_API_KEY,
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

            response_text = response.choices[0].message.content.strip()
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
    """Run the full inference pipeline and print strict-format logs."""
    env = SearchRankingEnv(seed=SEED)
    client = get_client()

    print("[START]")

    for difficulty in DIFFICULTIES:
        # 1. Reset environment
        observation = env.reset(difficulty)

        # 2. Get ranking from LLM (or fallback)
        ranking = get_llm_ranking(client, observation)

        # 3. Create action and step
        action = Action(ranking=ranking)
        _, reward, done, info = env.step(action)

        # 4. Log in strict format
        print()
        print("[STEP]")
        print(f"task: {difficulty}")
        print(f"query: {observation.query}")
        print(f"ranking: {ranking}")
        print(f"reward: {reward.score}")
        print(f"ndcg: {info.ndcg}")
        print(f"precision_at_k: {info.precision_at_k}")
        print(f"mrr: {info.mrr}")

    print()
    print("[END]")


if __name__ == "__main__":
    main()
