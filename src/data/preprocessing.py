
import json
import random
from datasets import load_from_disk
from transformers import GPT2Tokenizer
import os

class PubMedQAPreprocessor:
    def __init__(self, max_length=800):  # More conservative limit
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length  # Leave room for few-shot examples
        
    def format_example(self, example):
        """Format a single example into prompt structure"""
        question = example['question'].strip()
        # Take first context (abstract)
        abstract = example['context']['contexts'][0].strip()
        answer = example['final_decision'].strip()
        
        return {
            'question': question,
            'abstract': abstract,
            'answer': answer,
            'formatted_prompt': f"Question: {question}\nAbstract: {abstract}\nAnswer: {answer}"
        }
    
    def check_length(self, formatted_example):
        """Check if example fits within token limit"""
        prompt = formatted_example['formatted_prompt']
        tokens = self.tokenizer.encode(prompt)
        return len(tokens) <= self.max_length
    
    def create_splits(self, dataset, test_size=350, val_size=100, few_shot_pool=50):
        """Create train/val/test splits for the experiment"""
        
        # Get all examples and shuffle
        all_examples = [self.format_example(example) for example in dataset['train']]
        
        # Filter by length
        valid_examples = [ex for ex in all_examples if self.check_length(ex)]
        print(f"Filtered from {len(all_examples)} to {len(valid_examples)} examples (within token limit)")
        
        # Shuffle with fixed seed for reproducibility
        random.seed(42)
        random.shuffle(valid_examples)
        
        # Create splits
        test_set = valid_examples[:test_size]
        val_set = valid_examples[test_size:test_size + val_size]
        few_shot_pool_set = valid_examples[test_size + val_size:test_size + val_size + few_shot_pool]
        train_set = valid_examples[test_size + val_size + few_shot_pool:]
        
        splits = {
            'train': train_set,
            'validation': val_set,
            'test': test_set,
            'few_shot_pool': few_shot_pool_set
        }
        
        # Print split info
        print(f"\nData splits created:")
        for split_name, split_data in splits.items():
            print(f"{split_name}: {len(split_data)} examples")
            
            # Check label distribution
            labels = [ex['answer'] for ex in split_data]
            label_dist = {label: labels.count(label) for label in set(labels)}
            print(f"  Label distribution: {label_dist}")
        
        return splits
    
    def create_few_shot_examples(self, few_shot_pool, n_shots):
        """Create few-shot examples for prompting"""
        if n_shots == 0:
            return []
        
        # Sample n_shots examples
        selected = random.sample(few_shot_pool, min(n_shots, len(few_shot_pool)))
        return selected
    
    def save_splits(self, splits, output_dir='data/processed'):
        """Save processed splits to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, split_data in splits.items():
            output_file = f"{output_dir}/{split_name}.json"
            with open(output_file, 'w') as f:
                json.dump(split_data, f, indent=2)
            print(f"Saved {split_name} to {output_file}")

def main():
    """Main preprocessing pipeline"""
    # Load cached dataset
    print("Loading cached dataset...")
    dataset = load_from_disk('data/pubmedqa_cached')
    
    # Initialize preprocessor
    preprocessor = PubMedQAPreprocessor()
    
    # Create splits
    splits = preprocessor.create_splits(dataset)
    
    # Save splits
    preprocessor.save_splits(splits)
    
    # Create example few-shot prompts for testing
    print(f"\nCreating example few-shot prompts...")
    for n_shots in [1, 5, 10]:
        examples = preprocessor.create_few_shot_examples(splits['few_shot_pool'], n_shots)
        
        # Create a sample prompt
        test_example = splits['test'][0]
        few_shot_text = "\n\n".join([ex['formatted_prompt'] for ex in examples])
        
        full_prompt = f"""Answer biomedical questions with Yes, No, or Maybe based on the abstract:

{few_shot_text}

Question: {test_example['question']}
Abstract: {test_example['abstract']}
Answer:"""
        
        print(f"\n{n_shots}-shot prompt preview (first 500 chars):")
        print(full_prompt[:500] + "..." if len(full_prompt) > 500 else full_prompt)
        
        # Check token count
        token_count = len(preprocessor.tokenizer.encode(full_prompt))
        print(f"Token count: {token_count}")
    
    print("\nPreprocessing complete!")

if __name__ == "__main__":
    main()