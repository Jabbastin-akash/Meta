---
title: Search Ranking Env
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# Search Ranking Optimization Environment

An OpenEnv-compatible environment designed to simulate a real-world search ranking system. 
This environment is used to train and evaluate agents (e.g., LLMs) on their ability to assess the relevance of documents to a user's search query and rank them accordingly.

## 1. Environment Description

Search ranking formulation is the core of modern search engines and recommendation systems. Typically, a system narrows down millions of results to a small set of "candidate documents" and relies on a heavy, secondary ranker to evaluate precisely which document should be shown at position 1 vs position 10.

This environment perfectly models that downstream ranking task:
- **Input**: A user query and a short list of candidate documents.
- **Output**: An ordered list of those documents.
- **Goal**: Maximize user satisfaction by placing the most relevant documents at the very top.

## 2. Observation Space

At each step, the agent receives an **Observation** JSON containing:
- `query` (string): The search query provided by the simulated user.
- `documents` (array): A shuffled list of candidate objects, each containing:
  - `id` (string): A unique identifier for the document.
  - `text` (string): The content or title of the document.

## 3. Action Space

The agent must respond with an **Action** JSON. The action space consists of:
- `ranking` (array of strings): An ordered list of document IDs, from most relevant to least relevant.

**Constraints:** The ranking array must contain exactly every `id` provided in the observation exactly once. Missing IDs, extra IDs, or duplicates will result in a validation failure and a score of 0.

## 4. Reward Function

The environment utilizes continuous evaluation using **NDCG (Normalized Discounted Cumulative Gain)**.

- **Range**: `[0.0, 1.0]`
- **Explanation**: NDCG is the industry standard for evaluating ranking quality. It heavily discounts the value of relevant results that are placed lower in the ranking list. The metric compares the agent's ranking against the "ideal" theoretical ranking.
- **Partial Credit**: Yes. The agent will receive partial credit if relevant documents are placed high, even if the ordering is imperfect. 
- **Auxiliary Info**: Beside NDCG, `step()` also returns `precision_at_k` and `mrr` (Mean Reciprocal Rank).

## 5. Tasks (Difficulty Levels)

Three tasks are available via `reset(task=...)`:

### Easy
- **Documents**: 3–5 per query
- **Description**: Features strong separation between highly relevant and completely irrelevant items. It serves as a baseline sanity check.

### Medium
- **Documents**: 5–10 per query
- **Description**: Introduces overlapping relevance scores and subtle distractor documents. The agent must distinguish between closely related, yet differently relevant concepts.

### Hard
- **Documents**: 10–15 per query
- **Description**: Features high ambiguity, very similar relevance levels, and noisy data. Tests fine-grained ranking ability.

## 6. Setup Instructions

### Environment Variables

For the baseline inference script, configure the following:
```bash
export API_BASE_URL="https://openrouter.ai/api/v1" # Or any OpenAI-compatible API
export MODEL_NAME="your-model-name"
export OPENAI_API_KEY="your-api-key"
```

### Running with Docker

Build and run the OpenEnv HTTP API server (which powers the Hugging Face Deployment):
```bash
docker build -t search-env .
docker run -p 7860:7860 search-env
```
The API responds to `/reset` and `/step`.

### Running Inference

Install the dependencies, and simply run the inference script. It will evaluate your LLM on all difficulties.
```bash
pip install -r requirements.txt
python inference.py
```

## 7. Baseline Results

When the API lacks a valid key, the script performs a deterministic fallback gracefully using the shuffled observation order to avoid crashes. 

**Fallback (Random) Baseline Scores**:
- **Easy**: `0.6609`
- **Medium**: `0.6765`
- **Hard**: `0.6124`

A production LLM (e.g. GPT-4o) should easily achieve > `0.95` on Easy and > `0.85` on Hard.
# Meta
