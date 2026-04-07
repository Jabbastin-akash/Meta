import math
import random
from typing import List, Dict, Any, Tuple, Optional


def _safe_score(score: float) -> float:
    """Clamp to be strictly within (0, 1)."""
    epsilon = 1e-6
    if score <= 0.0:
        return epsilon
    if score >= 1.0:
        return 1.0 - epsilon
    return score

class SearchRankingEnv:
    """
    An OpenEnv-compatible environment for simulating a search ranking system.
    
    The agent acts by providing a ranked list of document IDs based on a query
    and receives a reward proportional to the NDCG of the provided ranking.
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            random.seed(seed)
        self.current_query = None
        self.candidate_documents = []
        self._ground_truth = {}
        self.difficulty = "easy"
        self._step_count = 0
        
    def reset(self, difficulty: str = "easy") -> Dict[str, Any]:
        """
        Initializes a new query-document set based on the difficulty level.
        Returns the initial state observation.
        """
        self.difficulty = difficulty.lower()
        self._step_count = 0
        self._generate_scenario()
        return self.state()
        
    def _generate_scenario(self):
        """
        Generates a query, candidate documents, and their ground truth relevances.
        """
        if self.difficulty == "easy":
            self.current_query = "best budget smartphone"
            base_docs = [
                {"id": "doc_1", "title": "Top 10 Budget Smartphones in 2023", "relevance": 3},
                {"id": "doc_2", "title": "A Review of the Latest Cheap Phones", "relevance": 2},
                {"id": "doc_3", "title": "Premium Smartphones Extravaganza", "relevance": 0},
                {"id": "doc_4", "title": "Best High-end Cameras", "relevance": 0},
                {"id": "doc_5", "title": "Affordable Mobile Phones Guide", "relevance": 1},
            ]
        elif self.difficulty == "medium":
            self.current_query = "healthy breakfast recipes"
            base_docs = [
                {"id": f"doc_{i}", "title": f"Healthy Breakfast Idea {i}", "relevance": random.choice([0, 1, 2])}
                for i in range(2, 11)
            ]
            base_docs.append({"id": "doc_1", "title": "15 Quick and Healthy Breakfast Recipes", "relevance": 3})
            
        else: # hard
            self.current_query = "machine learning vs deep learning tutorials"
            base_docs = [
                {"id": f"doc_{i}", "title": f"AI and ML Topic {i}", "relevance": random.choice([0, 0, 1, 1, 2])}
                for i in range(3, 21)
            ]
            base_docs.append({"id": "doc_1", "title": "Machine Learning vs Deep Learning: A Complete Tutorial", "relevance": 3})
            base_docs.append({"id": "doc_2", "title": "Intro to Deep Learning Concepts", "relevance": 2})
            
        # Shuffle documents so the agent is forced to rank them
        random.shuffle(base_docs)
        
        # Populate agent-visible states and hidden states
        self.candidate_documents = [
            {
                "id": d["id"], 
                "title": d["title"], 
                "metadata": {"popularity_score": round(random.uniform(0, 10), 2)}
            } for d in base_docs
        ]
        self._ground_truth = {d["id"]: d["relevance"] for d in base_docs}

    def state(self) -> Dict[str, Any]:
        """
        Returns the current observation space.
        """
        return {
            "query": self.current_query,
            "documents": self.candidate_documents
        }

    def step(self, action: List[str]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """
        Evaluates the agent's ranking.
        Action is a list of document IDs ordered by predicted relevance.
        
        Returns:
            observation (dict): The current state.
            reward (float): The NDCG score between 0.0 and 1.0.
            done (bool): Whether the episode has terminated.
            info (dict): Auxiliary metrics like Precision@K and MRR.
        """
        self._step_count += 1
        
        # Ensure the action contains exactly the same IDs as the candidate documents
        expected_ids = set(doc["id"] for doc in self.candidate_documents)
        provided_ids = set(action)
        
        if expected_ids != provided_ids:
            # If invalid action, return 0 reward and terminate
            return (
                self.state(),
                _safe_score(0.0),
                True,
                {"error": "Action must include exactly all candidate document IDs."},
            )
            
        # Calculate Reward (NDCG)
        reward = self._calculate_ndcg(action)
        reward = _safe_score(reward)
        
        # Auxiliary metrics
        info = {
            "ndcg": reward,
            "precision_at_3": _safe_score(self._calculate_precision_at_k(action, k=3)),
            "mrr": _safe_score(self._calculate_mrr(action)),
        }
        
        # Search ranking is inherently a single-step episode per query
        done = True 
        
        return self.state(), reward, done, info
        
    def _calculate_ndcg(self, predicted_ranking: List[str]) -> float:
        """
        Calculates the Normalized Discounted Cumulative Gain (NDCG).
        """
        # Calculate DCG for the predicted ranking
        dcg = 0.0
        for i, doc_id in enumerate(predicted_ranking):
            rel = self._ground_truth.get(doc_id, 0)
            dcg += (2 ** rel - 1) / math.log2(i + 2) # i+2 corresponds to standard rank positions 1, 2, 3...
            
        # Calculate Ideal DCG (IDCG)
        ideal_ranking = sorted(self._ground_truth.keys(), key=lambda x: self._ground_truth[x], reverse=True)
        idcg = 0.0
        for i, doc_id in enumerate(ideal_ranking):
            rel = self._ground_truth.get(doc_id, 0)
            idcg += (2 ** rel - 1) / math.log2(i + 2)
            
        if idcg == 0.0:
            raw = 1.0 if dcg == 0.0 else 0.0
            return _safe_score(raw)
            
        return _safe_score(dcg / idcg)

    def _calculate_precision_at_k(self, predicted_ranking: List[str], k: int) -> float:
        """
        Calculates precision at K (considering relevance > 0 as relevant).
        """
        if not predicted_ranking:
            return _safe_score(0.0)
        
        k = min(k, len(predicted_ranking))
        relevant_count = sum(1 for doc_id in predicted_ranking[:k] if self._ground_truth.get(doc_id, 0) > 0)
        
        return _safe_score(relevant_count / k)

    def _calculate_mrr(self, predicted_ranking: List[str]) -> float:
        """
        Calculates Mean Reciprocal Rank for the first remotely relevant document.
        """
        for i, doc_id in enumerate(predicted_ranking):
            if self._ground_truth.get(doc_id, 0) > 0:
                return _safe_score(1.0 / (i + 1))
        return _safe_score(0.0)
