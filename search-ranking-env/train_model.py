import os
import torch
import numpy as np
from datasets import load_dataset
from sentence_transformers import CrossEncoder, InputExample
from torch.utils.data import DataLoader
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator
import math

def main():
    print("Loading MS MARCO training dataset...")
    # 0.1% of 82326 train examples is approx 82
    train_dataset = load_dataset("ms_marco", "v1.1", split="train[:82]")
    
    print("Loading MS MARCO validation dataset for evaluation...")
    # 1% of 10047 validation examples is approx 100
    val_dataset = load_dataset("ms_marco", "v1.1", split="validation[:100]")
    
    train_examples = []
    print(f"Loaded {len(train_dataset)} training queries. Preparing examples...")
    for row in train_dataset:
        query = row['query']
        passages = row['passages']
        for i in range(len(passages['passage_text'])):
            text = passages['passage_text'][i]
            is_selected = passages['is_selected'][i]
            # Convert label to float (1.0 for relevant, 0.0 for non-relevant)
            label = float(is_selected)
            train_examples.append(InputExample(texts=[query, text], label=label))
            
    val_examples = []
    print(f"Loaded {len(val_dataset)} validation queries. Preparing examples...")
    for row in val_dataset:
        query = row['query']
        passages = row['passages']
        for i in range(len(passages['passage_text'])):
            text = passages['passage_text'][i]
            is_selected = passages['is_selected'][i]
            label = float(is_selected)
            val_examples.append(InputExample(texts=[query, text], label=label))
            
    print(f"Total training pairs: {len(train_examples)}")
    print(f"Total validation pairs: {len(val_examples)}")
    
    # We use a very small, fast model for demonstration
    model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    print(f"Initializing CrossEncoder model: {model_name}")
    model = CrossEncoder(model_name, num_labels=1)
    
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    
    # Use evaluator
    evaluator = CEBinaryClassificationEvaluator.from_input_examples(val_examples, name='ms-marco-dev')
    
    # Evaluate before training
    print("Evaluating before training...")
    score_before = evaluator(model)
    print(f"Score before training: {score_before}")
    
    print("Starting training...")
    
    epochs = 1
    # We use roughly 10% of train data for warmup
    warmup_steps = math.ceil(len(train_dataloader) * epochs * 0.1)
    
    model.fit(
        train_dataloader=train_dataloader,
        evaluator=evaluator,
        epochs=epochs,
        evaluation_steps=50,
        warmup_steps=warmup_steps,
        output_path="trained_msmarco_model"
    )
    
    print("Training complete!")
    print("Evaluating after training...")
    score_after = evaluator(model)
    print(f"Score after training: {score_after}")
    
    # Check predictions for a test case
    test_query = "What is the capital of France?"
    test_passages = [
        "Paris is the capital and most populous city of France.",
        "London is the capital and largest city of England and the United Kingdom.",
        "Berlin is the capital and largest city of Germany."
    ]
    
    print(f"\nTesting with query: '{test_query}'")
    for cp in test_passages:
        score = model.predict([test_query, cp])
        print(f"Score for Passage '{cp}': {score:.4f}")

if __name__ == "__main__":
    main()
