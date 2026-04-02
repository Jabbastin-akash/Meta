"""
Easy difficulty task dataset.

Characteristics:
  - 3–5 documents per query
  - Strong separation between relevant and irrelevant items
  - Minimal ambiguity — obvious intent, clearly distinguishable relevance
  - Relevance scale: 0 (irrelevant) to 3 (highly relevant)
"""

TASKS = [
    {
        "query": "best budget smartphone 2024",
        "documents": [
            {"id": "e1_d1", "text": "Top 10 Budget Smartphones of 2024 — Full Buyer's Guide", "relevance": 3.0},
            {"id": "e1_d2", "text": "Affordable Android Phones Under $200: A Detailed Comparison", "relevance": 2.0},
            {"id": "e1_d3", "text": "Premium Flagship Smartphones Over $1000", "relevance": 0.0},
            {"id": "e1_d4", "text": "Best DSLR Cameras for Professional Photography", "relevance": 0.0},
            {"id": "e1_d5", "text": "Budget Mobile Phone Accessories and Cases", "relevance": 1.0},
        ],
    },
    {
        "query": "how to learn python programming",
        "documents": [
            {"id": "e2_d1", "text": "Python for Absolute Beginners: Step-by-Step Tutorial", "relevance": 3.0},
            {"id": "e2_d2", "text": "Python Tips and Tricks for New Developers", "relevance": 2.0},
            {"id": "e2_d3", "text": "Learn JavaScript in 30 Days", "relevance": 0.0},
            {"id": "e2_d4", "text": "History of Programming Languages: From FORTRAN to Rust", "relevance": 0.0},
        ],
    },
    {
        "query": "best coffee makers for home",
        "documents": [
            {"id": "e3_d1", "text": "Top 5 Drip Coffee Makers for Home Use — 2024 Reviews", "relevance": 3.0},
            {"id": "e3_d2", "text": "Espresso vs Drip: Which Brewing Method Suits You?", "relevance": 2.0},
            {"id": "e3_d3", "text": "Kitchen Blender Buying Guide", "relevance": 0.0},
            {"id": "e3_d4", "text": "The Global History of Coffee Cultivation", "relevance": 0.0},
            {"id": "e3_d5", "text": "How to Clean and Maintain Your Coffee Machine", "relevance": 1.0},
        ],
    },
    {
        "query": "weather in new york today",
        "documents": [
            {"id": "e4_d1", "text": "New York City Weather Forecast: Today and This Week", "relevance": 3.0},
            {"id": "e4_d2", "text": "Historical Climate Data for New York State", "relevance": 1.0},
            {"id": "e4_d3", "text": "Best Restaurants in Manhattan", "relevance": 0.0},
        ],
    },
]
