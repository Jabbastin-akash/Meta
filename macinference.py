import json
import random
import sys
from typing import List

from datasets import load_dataset
from openai import OpenAI

# Import from the existing infrastructure
from models import Document, Observation, Action
from grader import grade
from inference import get_client, get_llm_ranking

def _clamp(x: float) -> float:
    """Clamp all scores strictly within (0, 1)."""
    if x <= 0.0:
        return 0.001
    if x >= 1.0:
        return 0.999
    return x

def main():
    print("Loading MS MARCO validation dataset (first 100 items) to build tasks...")
    # Load a small slice to find good rankable questions
    dataset = load_dataset("ms_marco", "v1.1", split="validation[:100]")
    
    # Extract good validation tasks (queries that have at least 1 relevant doc and 4+ total docs)
    tasks = []
    for row in dataset:
        query = row["query"]
        passages = row["passages"]
        
        # We need both selected and non-selected for a good ranking task
        has_positive = any(p == 1 for p in passages["is_selected"])
        if not has_positive or len(passages["passage_text"]) < 5:
            continue
            
        docs = []
        ground_truth = {}
        for i, text in enumerate(passages["passage_text"]):
            doc_id = f"doc_{i}"
            # Relevance in MS MARCO is binary here (1 selected, 0 not selected)
            relevance = float(passages["is_selected"][i])
            
            # Truncate text slightly if it's too massive, to save on token limits, but MS MARCO passages are usually short
            docs.append(Document(id=doc_id, text=text, relevance=relevance)) 
            ground_truth[doc_id] = relevance
        
        # Shuffle so the LLM cannot rely on any default dataset ordering
        random.shuffle(docs)
        tasks.append((query, docs, ground_truth))
        
        # We limit to 3 tasks to match the speed of the original inference script (easy, medium, hard)
        if len(tasks) >= 3:
            break
            
    if not tasks:
        print("Could not find suitable tasks in the slice.", file=sys.stderr)
        return

    client = get_client()

    print("\n[START]")
    
    for idx, (query, docs, ground_truth) in enumerate(tasks):
        observation = Observation(query=query, documents=docs)
        
        # Call LLM logic completely reused from your existing inference.py
        ranking = get_llm_ranking(client, observation)
        
        action = Action(ranking=ranking)
        
        # Validate action
        try:
            action.validate_against_observation(observation)
            
            # Use original grader against ground truth
            result = grade(action.ranking, ground_truth, k=3)
            reward_score = result.score
            ndcg = result.ndcg
            precision = result.precision_at_k
            mrr = result.mrr
            
        except ValueError as e:
            print(f"WARNING: Invalid action taking default zero reward - {e}", file=sys.stderr)
            reward_score = 0.0
            ndcg = 0.0
            precision = 0.0
            mrr = 0.0

        # Clamp all scores strictly within [0.1, 0.85]
        reward_score = _clamp(reward_score)
        ndcg = _clamp(ndcg)
        precision = _clamp(precision)
        mrr = _clamp(mrr)

        # Print using exact same format as expected by OpenEnv grader
        print(f"\n[STEP]")
        print(f"task: ms_marco_eval_{idx+1}")
        print(f"query: {query}")
        print(f"ranking: {ranking}")
        print(f"reward: {reward_score}")
        print(f"ndcg: {ndcg}")
        print(f"precision_at_k: {precision}")
        print(f"mrr: {mrr}")

    print("\n[END]")

if __name__ == "__main__":
    main()
