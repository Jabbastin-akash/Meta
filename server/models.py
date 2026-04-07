from typing import List

from pydantic import BaseModel, Field, model_validator

class Document(BaseModel):
    """Represents a candidate item to be ranked."""
    id: str = Field(..., description="Unique identifier for the document")
    text: str = Field(..., description="Content or title of the document")
    relevance: float = Field(..., description="Ground truth relevance score (used internally for evaluation only)")

class Observation(BaseModel):
    """Represents the state returned to the agent."""
    query: str = Field(..., description="The user search query")
    documents: List[Document] = Field(..., description="List of candidate documents")

class Action(BaseModel):
    """Represents the agent's output."""
    ranking: List[str] = Field(..., description="Ordered list of document IDs")

    @model_validator(mode='after')
    def check_unique_ranking(self):
        # Validate that the ranking list contains no duplicate IDs
        if len(self.ranking) != len(set(self.ranking)):
            raise ValueError("Ranking contains duplicate document IDs.")
        return self
        
    def validate_against_observation(self, obs: Observation) -> bool:
        """Cross-validate that ranking contains exactly the observation's document IDs."""
        obs_ids = {doc.id for doc in obs.documents}
        rank_ids = set(self.ranking)
        
        if obs_ids != rank_ids:
            raise ValueError("Action.ranking must contain exactly all document IDs from Observation.")
        return True

class Reward(BaseModel):
    """Represents the score returned by the environment."""
    score: float = Field(..., ge=0.0, le=1.0, description="A normalized value between 0.0 and 1.0")

class Info(BaseModel):
    """Provides additional evaluation metrics."""
    ndcg: float = Field(..., ge=0.0, le=1.0, description="Normalized Discounted Cumulative Gain")
    precision_at_k: float = Field(..., ge=0.0, le=1.0, description="Precision at top K results")
    mrr: float = Field(..., ge=0.0, le=1.0, description="Mean Reciprocal Rank")
