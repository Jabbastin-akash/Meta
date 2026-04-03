from models import Document, Observation, Action, Reward, Info

def test_models():
    # 1. Create a sample Document
    doc1 = Document(id="d1", text="Sample Doc 1", relevance=0.8)
    doc2 = Document(id="d2", text="Sample Doc 2", relevance=0.2)
    print("Documents created successfully.")

    # 2. Create an Observation
    obs = Observation(query="test query", documents=[doc1, doc2])
    print("Observation created successfully.")

    # 3. Create an Action with a valid ranking
    action = Action(ranking=["d2", "d1"])
    print("Action created successfully.")
    
    # Check valid validation against observation
    action.validate_against_observation(obs)
    print("Action validated against Observation successfully.")

    # 4. Create a Reward with a valid score
    reward = Reward(score=0.95)
    print("Reward created successfully.")
    
    # 5. Create an Info metric
    info = Info(ndcg=0.9, precision_at_k=1.0, mrr=1.0)
    print("Info created successfully.")
    
    # Test Validation checks
    try:
        invalid_reward = Reward(score=1.5)
        print("ERROR: Invalid reward should have raised error")
    except ValueError:
        print("Invalid reward caught successfully.")
        
    try:
        invalid_action = Action(ranking=["d1", "d1"])
        print("ERROR: Duplicate action should have raised error")
    except ValueError:
        print("Invalid action (duplicates) caught successfully.")
        
    try:
        invalid_action_match = Action(ranking=["d1", "d3"])
        invalid_action_match.validate_against_observation(obs)
        print("ERROR: Mismatched action should have raised error")
    except ValueError:
        print("Invalid action (mismatch with obs) caught successfully.")

if __name__ == "__main__":
    test_models()
