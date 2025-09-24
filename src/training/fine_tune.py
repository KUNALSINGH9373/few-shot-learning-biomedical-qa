
import json
import torch
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from torch.utils.data import Dataset
import os
from tqdm import tqdm
import numpy as np
from pathlib import Path

class PubMedQADataset(Dataset):
    """Dataset class for PubMedQA fine-tuning"""
    
    def __init__(self, data, tokenizer, max_length=1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        example = self.data[idx]
        
        # Create formatted text for causal language modeling
        text = f"Question: {example['question']}\nAbstract: {example['abstract']}\nAnswer: {example['answer']}<|endoftext|>"
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze()
        }

class GPT2FineTuner:
    def __init__(self, model_name='gpt2', output_dir='models/fine_tuned'):
        self.model_name = model_name
        self.output_dir = output_dir
        
        print(f"Loading {model_name} for fine-tuning...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
        # Add padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        print(f"Model loaded on {self.device}")
    
    def load_data(self, data_dir='data/processed'):
        """Load preprocessed data splits"""
        splits = {}
        for split in ['train', 'validation', 'test']:
            file_path = f"{data_dir}/{split}.json"
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    splits[split] = json.load(f)
                print(f"Loaded {len(splits[split])} examples for {split}")
            else:
                print(f"Warning: {file_path} not found")
        return splits
    
    def create_datasets(self, splits, max_length=1024):
        """Create PyTorch datasets for training"""
        datasets = {}
        
        for split_name, split_data in splits.items():
            if split_data:
                datasets[split_name] = PubMedQADataset(
                    split_data, 
                    self.tokenizer, 
                    max_length=max_length
                )
                print(f"Created {split_name} dataset with {len(datasets[split_name])} examples")
        
        return datasets
    
    def fine_tune(self, config):
        """Fine-tune GPT-2 on PubMedQA data"""
        
        # Load and create datasets
        splits = self.load_data()
        datasets = self.create_datasets(
            splits, 
            max_length=config['data']['max_length']
        )
        
        if 'train' not in datasets or 'validation' not in datasets:
            raise ValueError("Training and validation datasets are required")
        
        # Create data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal language modeling, not masked LM
        )
        
        # Setup training arguments
        experiment_dir = config['output']['experiment_dir']
        model_save_dir = config['output']['model_save_dir']
        
        os.makedirs(experiment_dir, exist_ok=True)
        os.makedirs(model_save_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=experiment_dir,
            overwrite_output_dir=True,
            num_train_epochs=config['training']['num_epochs'],
            per_device_train_batch_size=config['training']['batch_size'],
            per_device_eval_batch_size=config['training']['batch_size'],
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            learning_rate=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            warmup_steps=config['training']['warmup_steps'],
            logging_steps=50,
            logging_dir=f"{experiment_dir}/logs",
            eval_strategy="epoch",
            save_strategy="epoch",
            save_steps=500,
            eval_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # Disable wandb/tensorboard for now
            seed=42,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=datasets['train'],
            eval_dataset=datasets['validation'],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        print("Starting fine-tuning...")
        print(f"Training examples: {len(datasets['train'])}")
        print(f"Validation examples: {len(datasets['validation'])}")
        print(f"Training for {config['training']['num_epochs']} epochs")
        
        # Fine-tune the model
        trainer.train()
        
        # Save the final model
        final_model_path = f"{model_save_dir}/final"
        trainer.save_model(final_model_path)
        print(f"Final model saved to: {final_model_path}")
        
        # Evaluate on validation set
        eval_results = trainer.evaluate()
        print(f"Final validation loss: {eval_results['eval_loss']:.4f}")
        
        # Save training results
        results = {
            'training_args': training_args.to_dict(),
            'final_eval_results': eval_results,
            'model_path': final_model_path,
            'training_dataset_size': len(datasets['train']),
            'validation_dataset_size': len(datasets['validation'])
        }
        
        results_file = f"{experiment_dir}/training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Training results saved to: {results_file}")
        
        return trainer, results

class FineTunedEvaluator:
    """Evaluate fine-tuned model on test set"""
    
    def __init__(self, model_path, tokenizer_name='gpt2'):
        self.tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loading fine-tuned model from: {model_path}")
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Fine-tuned model loaded on {self.device}")
    
    def generate_answer(self, question, abstract, max_new_tokens=10, temperature=0.1):
        """Generate answer using fine-tuned model"""
        
        # Create prompt in same format as training
        prompt = f"Question: {question}\nAbstract: {abstract}\nAnswer:"
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors='pt', truncation=True, max_length=1000)
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode generated part
        generated_text = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return generated_text.strip()
    
    def extract_answer(self, generated_text):
        """Extract Yes/No/Maybe from generated text"""
        
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
    
    def evaluate_on_test_set(self, test_data, max_eval=None):
        """Evaluate fine-tuned model on test data"""
        
        if max_eval:
            test_data = test_data[:max_eval]
        
        print(f"Evaluating fine-tuned model on {len(test_data)} test examples...")
        
        correct = 0
        total = 0
        predictions = []
        
        for example in tqdm(test_data, desc="Fine-tuned evaluation"):
            # Generate answer
            generated = self.generate_answer(example['question'], example['abstract'])
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
                'generated_text': generated
            })
        
        accuracy = correct / total
        
        results = {
            'model_type': 'fine_tuned',
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions
        }
        
        print(f"Fine-tuned model accuracy: {accuracy:.3f} ({correct}/{total})")
        
        return results

def main():
    """Test fine-tuning implementation"""
    
    # Simple test configuration
    test_config = {
        'data': {'max_length': 1024},
        'training': {
            'num_epochs': 1,  # Quick test
            'batch_size': 2,
            'gradient_accumulation_steps': 2,
            'learning_rate': 5e-5,
            'weight_decay': 0.01,
            'warmup_steps': 10
        },
        'output': {
            'experiment_dir': 'results/experiments/fine_tuning_test',
            'model_save_dir': 'models/fine_tuned'
        }
    }
    
    # Initialize and run fine-tuning
    fine_tuner = GPT2FineTuner()
    trainer, results = fine_tuner.fine_tune(test_config)
    
    print("Fine-tuning test completed!")

if __name__ == "__main__":
    main()