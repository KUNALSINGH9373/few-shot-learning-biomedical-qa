
import os
from datasets import load_dataset
import json
import pandas as pd
from collections import Counter

def download_and_explore_pubmedqa():
    """Download PubMedQA and create initial data exploration"""
    
    print("Downloading PubMedQA dataset...")
    # Load the labeled version
    dataset = load_dataset("pubmed_qa", "pqa_labeled")
    
    print(f"Dataset structure: {dataset}")
    print(f"Train set size: {len(dataset['train'])}")
    
    # Explore first few examples
    print("\n" + "="*50)
    print("SAMPLE EXAMPLES:")
    print("="*50)
    
    for i in range(3):
        example = dataset['train'][i]
        print(f"\nExample {i+1}:")
        print(f"Question: {example['question']}")
        print(f"Answer: {example['final_decision']}")
        print(f"Abstract length: {len(example['context']['contexts'][0])} chars")
        print("-" * 30)
    
    # Analyze label distribution
    labels = [example['final_decision'] for example in dataset['train']]
    label_counts = Counter(labels)
    
    print(f"\nLabel distribution:")
    for label, count in label_counts.items():
        print(f"{label}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Save dataset info
    dataset_info = {
        'total_examples': len(dataset['train']),
        'label_distribution': dict(label_counts),
        'sample_questions': [dataset['train'][i]['question'] for i in range(5)]
    }
    
    with open('data/dataset_info.json', 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Save dataset locally for faster access
    dataset.save_to_disk('data/pubmedqa_cached')
    print(f"\nDataset cached locally in data/pubmedqa_cached/")
    
    return dataset

if __name__ == "__main__":
    dataset = download_and_explore_pubmedqa()