"""
Hard difficulty task dataset.

Characteristics:
  - 10–15 documents per query
  - Very similar relevance levels across many documents
  - High ambiguity — queries that can be interpreted multiple ways
  - Noisy data with misleading or borderline-relevant items
  - Relevance scale: 0 (irrelevant) to 3 (highly relevant)
"""

TASKS = [
    {
        "query": "machine learning vs deep learning",
        "documents": [
            {"id": "h1_d1", "text": "Machine Learning vs Deep Learning: A Comprehensive Tutorial", "relevance": 3.0},
            {"id": "h1_d2", "text": "Introduction to Deep Learning Architectures", "relevance": 2.0},
            {"id": "h1_d3", "text": "Supervised Learning Algorithms Explained Simply", "relevance": 2.0},
            {"id": "h1_d4", "text": "Neural Networks from Scratch in Python", "relevance": 2.0},
            {"id": "h1_d5", "text": "Gradient Descent Optimizers: SGD, Adam, RMSProp", "relevance": 1.0},
            {"id": "h1_d6", "text": "Convolutional Neural Networks for Image Classification", "relevance": 1.0},
            {"id": "h1_d7", "text": "Recurrent Neural Networks and LSTMs Explained", "relevance": 1.0},
            {"id": "h1_d8", "text": "Natural Language Processing with Transformers", "relevance": 1.0},
            {"id": "h1_d9", "text": "TensorFlow vs PyTorch: Framework Comparison", "relevance": 1.0},
            {"id": "h1_d10", "text": "Feature Engineering Best Practices for ML Pipelines", "relevance": 1.0},
            {"id": "h1_d11", "text": "Reinforcement Learning: A Beginner's Guide", "relevance": 1.0},
            {"id": "h1_d12", "text": "The History of Artificial Intelligence", "relevance": 0.0},
            {"id": "h1_d13", "text": "Blockchain Technology and Cryptocurrency Explained", "relevance": 0.0},
            {"id": "h1_d14", "text": "Best Laptops for Software Developers 2024", "relevance": 0.0},
            {"id": "h1_d15", "text": "Cloud Computing Services Comparison: AWS vs GCP", "relevance": 0.0},
        ],
    },
    {
        "query": "sustainable energy solutions for homes",
        "documents": [
            {"id": "h2_d1", "text": "Complete Guide to Residential Solar Panel Installation", "relevance": 3.0},
            {"id": "h2_d2", "text": "Home Battery Storage: Tesla Powerwall vs Alternatives", "relevance": 2.0},
            {"id": "h2_d3", "text": "Heat Pump Technology for Energy-Efficient Heating", "relevance": 2.0},
            {"id": "h2_d4", "text": "Small Wind Turbines for Residential Properties", "relevance": 2.0},
            {"id": "h2_d5", "text": "Solar Water Heater Installation Step-by-Step", "relevance": 2.0},
            {"id": "h2_d6", "text": "Smart Thermostat Buying Guide 2024", "relevance": 1.0},
            {"id": "h2_d7", "text": "Energy-Efficient Windows and Insulation Tips", "relevance": 1.0},
            {"id": "h2_d8", "text": "Government Rebates for Green Energy Adoption", "relevance": 1.0},
            {"id": "h2_d9", "text": "LED Lighting: How Much Can You Really Save?", "relevance": 1.0},
            {"id": "h2_d10", "text": "Geothermal Heating Systems Explained", "relevance": 1.0},
            {"id": "h2_d11", "text": "Nuclear Energy: Global Pros and Cons", "relevance": 0.0},
            {"id": "h2_d12", "text": "Oil and Gas Industry Market Report 2024", "relevance": 0.0},
            {"id": "h2_d13", "text": "Large-Scale Wind Farm Engineering", "relevance": 0.0},
            {"id": "h2_d14", "text": "Cryptocurrency Mining Energy Consumption", "relevance": 0.0},
        ],
    },
    {
        "query": "best strategies for personal finance management",
        "documents": [
            {"id": "h3_d1", "text": "Complete Personal Finance Guide: Budgeting, Saving, Investing", "relevance": 3.0},
            {"id": "h3_d2", "text": "How to Build an Emergency Fund in 6 Months", "relevance": 2.0},
            {"id": "h3_d3", "text": "Index Fund Investing for Beginners", "relevance": 2.0},
            {"id": "h3_d4", "text": "50/30/20 Budget Rule Explained with Examples", "relevance": 2.0},
            {"id": "h3_d5", "text": "Debt Snowball vs Debt Avalanche: Which Works Better?", "relevance": 2.0},
            {"id": "h3_d6", "text": "Credit Score: How It Works and How to Improve It", "relevance": 1.0},
            {"id": "h3_d7", "text": "Roth IRA vs Traditional IRA: Retirement Account Comparison", "relevance": 1.0},
            {"id": "h3_d8", "text": "Side Hustles to Boost Your Income in 2024", "relevance": 1.0},
            {"id": "h3_d9", "text": "Understanding Mortgage Rates and Home Loans", "relevance": 1.0},
            {"id": "h3_d10", "text": "Tax Filing Tips for Freelancers", "relevance": 1.0},
            {"id": "h3_d11", "text": "Day Trading Strategies for Volatile Markets", "relevance": 0.0},
            {"id": "h3_d12", "text": "Luxury Watch Collecting as an Investment", "relevance": 0.0},
            {"id": "h3_d13", "text": "Corporate Accounting Standards Overview", "relevance": 0.0},
        ],
    },
]
