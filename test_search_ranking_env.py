import unittest
from search_ranking_env import SearchRankingEnv

class TestSearchRankingEnv(unittest.TestCase):
    def setUp(self):
        self.env = SearchRankingEnv()

    def test_environment_interfaces(self):
        # Test easy difficulty
        state = self.env.reset(difficulty="easy")
        
        self.assertIn("query", state)
        self.assertIn("documents", state)
        self.assertEqual(len(state["documents"]), 5)
        
        # Test basic info return values
        docs = state["documents"]
        action = [doc["id"] for doc in docs] # arbitrary order
        
        next_state, reward, done, info = self.env.step(action)
        
        self.assertIsInstance(next_state, dict)
        self.assertIsInstance(reward, float)
        self.assertTrue(done)
        self.assertIn("ndcg", info)
        self.assertIn("precision_at_3", info)
        self.assertIn("mrr", info)
        
    def test_optimal_ranking_reward(self):
        state = self.env.reset(difficulty="medium")
        
        # Get ground truth
        ground_truth = self.env._ground_truth
        
        # Create an optimal ranking manually
        optimal_ranking = sorted(ground_truth.keys(), key=lambda x: ground_truth[x], reverse=True)
        
        _, reward, _, info = self.env.step(optimal_ranking)
        
        # Expect NDCG = 1.0 for optimal
        self.assertAlmostEqual(reward, 1.0, places=5)
        self.assertAlmostEqual(info["ndcg"], 1.0, places=5)
        
    def test_suboptimal_ranking_reward(self):
        state = self.env.reset(difficulty="hard")
        
        ground_truth = self.env._ground_truth
        
        # Create a completely reversed (pessimal) ranking
        pessimal_ranking = sorted(ground_truth.keys(), key=lambda x: ground_truth[x], reverse=False)
        
        _, reward, _, _ = self.env.step(pessimal_ranking)
        
        # Expect NDCG to be less than 1.0, but greater than or equal to 0.0 depending on DCG math
        self.assertTrue(0.0 <= reward < 1.0)
        
    def test_invalid_action(self):
        self.env.reset(difficulty="easy")
        
        # pass incomplete list
        _, reward, done, info = self.env.step(["doc_1"])
        
        self.assertEqual(reward, 0.0)
        self.assertTrue(done)
        self.assertIn("error", info)

if __name__ == "__main__":
    unittest.main()
