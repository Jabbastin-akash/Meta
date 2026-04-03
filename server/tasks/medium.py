"""
Medium difficulty task dataset.

Characteristics:
  - 5–10 documents per query
  - Overlapping relevance scores among several documents
  - Some noise and distractor documents
  - Queries with multiple valid interpretations
  - Relevance scale: 0 (irrelevant) to 3 (highly relevant)
"""

TASKS = [
    {
        "query": "healthy breakfast recipes for weight loss",
        "documents": [
            {"id": "m1_d1", "text": "15 Quick and Healthy Breakfast Recipes for Busy Mornings", "relevance": 3.0},
            {"id": "m1_d2", "text": "Overnight Oats: 5 Low-Calorie Variations", "relevance": 2.0},
            {"id": "m1_d3", "text": "Smoothie Bowls for a Nutritious Start to the Day", "relevance": 2.0},
            {"id": "m1_d4", "text": "Egg-Based Breakfasts: Omelettes and Frittatas", "relevance": 2.0},
            {"id": "m1_d5", "text": "Best Brunch Spots in New York City", "relevance": 0.0},
            {"id": "m1_d6", "text": "Healthy Lunch and Dinner Ideas for Weight Loss", "relevance": 1.0},
            {"id": "m1_d7", "text": "Pancake Recipes from Around the World", "relevance": 1.0},
            {"id": "m1_d8", "text": "The Science of Intermittent Fasting", "relevance": 1.0},
            {"id": "m1_d9", "text": "High-Protein Breakfast Ideas for Athletes", "relevance": 2.0},
            {"id": "m1_d10", "text": "How to Stock a Healthy Kitchen Pantry", "relevance": 0.0},
        ],
    },
    {
        "query": "remote work productivity tips",
        "documents": [
            {"id": "m2_d1", "text": "10 Proven Strategies for Remote Work Productivity", "relevance": 3.0},
            {"id": "m2_d2", "text": "Home Office Setup Guide for Maximum Focus", "relevance": 2.0},
            {"id": "m2_d3", "text": "Pomodoro Technique: A Time Management Deep Dive", "relevance": 2.0},
            {"id": "m2_d4", "text": "How to Avoid Burnout When Working from Home", "relevance": 2.0},
            {"id": "m2_d5", "text": "Best Standing Desks Under $500", "relevance": 1.0},
            {"id": "m2_d6", "text": "Slack vs Microsoft Teams: Feature Comparison", "relevance": 1.0},
            {"id": "m2_d7", "text": "Dealing with Loneliness While Working Remotely", "relevance": 1.0},
            {"id": "m2_d8", "text": "Office Lease Rates in Downtown Chicago", "relevance": 0.0},
        ],
    },
    {
        "query": "best running shoes for beginners",
        "documents": [
            {"id": "m3_d1", "text": "Top 10 Running Shoes for Beginners — Expert Picks 2024", "relevance": 3.0},
            {"id": "m3_d2", "text": "How to Choose Your First Pair of Running Shoes", "relevance": 2.0},
            {"id": "m3_d3", "text": "Nike vs Asics: Which Brand is Better for New Runners?", "relevance": 2.0},
            {"id": "m3_d4", "text": "Trail Running Shoes vs Road Running Shoes Explained", "relevance": 1.0},
            {"id": "m3_d5", "text": "Couch to 5K: A Complete Beginner's Training Plan", "relevance": 1.0},
            {"id": "m3_d6", "text": "Basketball Shoes Buying Guide", "relevance": 0.0},
            {"id": "m3_d7", "text": "How to Prevent Common Running Injuries", "relevance": 1.0},
            {"id": "m3_d8", "text": "Fashion Sneaker Trends for 2024", "relevance": 0.0},
            {"id": "m3_d9", "text": "Minimalist Running Shoes: Pros and Cons", "relevance": 2.0},
        ],
    },
]
