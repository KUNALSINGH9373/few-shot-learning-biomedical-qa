
import json
import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import re

class FewShotPrompter:
    def __init__(self, model_name='gpt2'):
        print(f"Loading {model_name} model and tokenizer...")
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded on {self.device}")
    
    def load_data(self, data_dir='data/processed'):
        """Load preprocessed data splits"""
        splits = {}
        for split in ['train', 'validation', 'test', 'few_shot_pool']:
            with open(f"{data_dir}/{split}.json", 'r') as f:
                splits[split] = json.load(f)
        return splits
    
    def create_prompt(self, question, abstract, few_shot_examples=None):
        """Create prompt with optional few-shot examples, ensuring it fits within token limits"""
        
        if few_shot_examples is None or len(few_shot_examples) == 0:
            # 0-shot prompt
            prompt = f"""Answer biomedical questions with Yes, No, or Maybe based on the abstract:

Question: {question}
Abstract: {abstract}
Answer:"""
        else:
            # Few-shot prompt with length management
            base_prompt = f"""Answer biomedical questions with Yes, No, or Maybe based on the abstract:

Question: {question}
Abstract: {abstract}
Answer:"""
            
            # Calculate available space for examples
            base_tokens = len(self.tokenizer.encode(base_prompt))
            max_example_tokens = 950 - base_tokens  # Leave some buffer
            
            # Add examples until we run out of space
            examples_text = ""
            used_examples = []
            
            for example in few_shot_examples:
                example_text = f"Question: {example['question']}\nAbstract: {example['abstract']}\nAnswer: {example['answer']}\n\n"
                example_tokens = len(self.tokenizer.encode(example_text))
                
                if len(self.tokenizer.encode(examples_text + example_text)) <= max_example_tokens:
                    examples_text += example_text
                    used_examples.append(example)
                else:
                    break
            
            prompt = f"""Answer biomedical questions with Yes, No, or Maybe based on the abstract:

{examples_text.strip()}

Question: {question}
Abstract: {abstract}
Answer:"""
        
        return prompt
    
    def generate_answer(self, prompt, max_new_tokens=10, temperature=0.1):
        """Generate answer for a given prompt with proper length management"""
        
        # Tokenize input with strict length limits
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=1000)
        
        # Safety check
        if inputs.shape[1] > 1000:
            print(f"[WARNING] Prompt length ({inputs.shape[1]}) exceeds safe limit, truncating")
            inputs = inputs[:, :1000]
        
        inputs = inputs.to(self.device)
        
        # Generate with error handling
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    max_length=1024  # Hard limit for GPT-2
                )
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return "maybe"  # Default fallback
        
        # Decode the generated part only
        generated_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return generated_text.strip()
    
    def extract_answer(self, generated_text):
        """Extract Yes/No/Maybe from generated text"""
        
        # Clean the text
        text = generated_text.lower().strip()
        
        # Look for exact matches first
        if text.startswith('yes'):
            return 'yes'
        elif text.startswith('no'):
            return 'no'
        elif text.startswith('maybe'):
            return 'maybe'
        
        # Look for patterns
        if 'yes' in text and 'no' not in text:
            return 'yes'
        elif 'no' in text and 'yes' not in text:
            return 'no'
        elif 'maybe' in text:
            return 'maybe'
        
        # Default fallback
        return 'maybe'
    
    def evaluate_few_shot(self, test_data, few_shot_examples, n_shots, max_eval=100):
        """Evaluate model with n-shot prompting"""
        
        print(f"Evaluating {n_shots}-shot prompting on {min(len(test_data), max_eval)} examples...")
        
        correct = 0
        total = 0
        predictions = []
        actual_shots_used = 0
        
        # Limit evaluation for faster iteration during development
        eval_data = test_data[:max_eval]
        
        for example in tqdm(eval_data, desc=f"{n_shots}-shot evaluation"):
            # Create prompt with length management
            prompt = self.create_prompt(
                example['question'], 
                example['abstract'], 
                few_shot_examples[:n_shots] if n_shots > 0 else None
            )
            
            # Check actual prompt length and count examples used
            if n_shots > 0 and actual_shots_used == 0:  # Only check once
                prompt_tokens = len(self.tokenizer.encode(prompt))
                # Count how many examples were actually included
                for i in range(1, n_shots + 1):
                    test_prompt = self.create_prompt(
                        example['question'], 
                        example['abstract'], 
                        few_shot_examples[:i]
                    )
                    if len(self.tokenizer.encode(test_prompt)) < 1000:
                        actual_shots_used = i
                    else:
                        break
                
                if actual_shots_used < n_shots:
                    print(f"[WARNING] Could only fit {actual_shots_used}/{n_shots} examples due to length limits")
            
            # Generate answer
            generated = self.generate_answer(prompt)
            predicted = self.extract_answer(generated)
            actual = example['answer'].lower()
            
            # Check if correct
            is_correct = predicted == actual
            correct += is_correct
            total += 1
            
            predictions.append({
                'question': example['question'],
                'actual': actual,
                'predicted': predicted,
                'correct': is_correct,
                'generated_text': generated,
                'prompt_length': len(self.tokenizer.encode(prompt))
            })
        
        accuracy = correct / total
        
        results = {
            'n_shots': n_shots,
            'actual_shots_used': max(actual_shots_used, n_shots) if n_shots > 0 else 0,
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions
        }
        
        shots_info = f" (used {actual_shots_used})" if actual_shots_used > 0 and actual_shots_used < n_shots else ""
        print(f"{n_shots}-shot{shots_info} accuracy: {accuracy:.3f} ({correct}/{total})")
        
        return results
    
    def run_few_shot_experiments(self, shot_counts=[0, 1, 2, 5, 10], max_eval=100):
        """Run experiments across different shot counts"""
        
        # Load data
        splits = self.load_data()
        test_data = splits['test']
        few_shot_pool = splits['few_shot_pool']
        
        print(f"Running experiments on {len(test_data)} test examples")
        print(f"Few-shot pool size: {len(few_shot_pool)}")
        
        # Set seed for reproducible few-shot example selection
        random.seed(42)
        few_shot_examples = random.sample(few_shot_pool, min(20, len(few_shot_pool)))
        
        all_results = {}
        
        for n_shots in shot_counts:
            print(f"\n" + "=" * 50)
            print(f"EVALUATING {n_shots}-SHOT")
            print(f"=" * 50)
            
            results = self.evaluate_few_shot(test_data, few_shot_examples, n_shots, max_eval)
            all_results[f"{n_shots}_shot"] = results
        
        # Save results
        output_file = 'results/few_shot_results.json'
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\nResults saved to {output_file}")
        
        # Print summary
        print(f"\nEXPERIMENT SUMMARY")
        print(f"=" * 50)
        
        for n_shots in shot_counts:
            accuracy = all_results[f"{n_shots}_shot"]['accuracy']
            print(f"{n_shots}-shot: {accuracy:.3f}")
        
        return all_results

def main():
    """Run few-shot prompting experiments"""
    
    # Initialize prompter
    prompter = FewShotPrompter()
    
    # Run experiments
    results = prompter.run_few_shot_experiments(
        shot_counts=[0, 1, 2, 5, 10],
        max_eval=50  # Start with 50 for quick iteration
    )

if __name__ == "__main__":
    main()