import json
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any
import requests
from datasets import load_dataset
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
import logging
from datetime import datetime

class CrossDomainTransferEvaluator:
    def __init__(self):
        self.setup_logging()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.results = {}
        
    def setup_logging(self):
        """Set up logging for the experiment"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('results/experiments/cross_domain_transfer.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('cross_domain_transfer')
        
    def create_covid_dataset(self) -> List[Dict]:
        """Create COVID-19 biomedical QA dataset"""
        self.logger.info("Creating COVID-19 biomedical QA dataset...")
        
        # Sample COVID-19 biomedical questions (in practice, you'd use CORD-19 or similar)
        covid_questions = [
            {
                "question": "Does COVID-19 cause cardiovascular complications?",
                "abstract": "SARS-CoV-2 infection has been associated with various cardiovascular manifestations including myocarditis, arrhythmias, and thrombotic events. Studies show increased troponin levels and ECG abnormalities in COVID-19 patients, indicating direct cardiac involvement.",
                "answer": "Yes"
            },
            {
                "question": "Is hydroxychloroquine effective for treating COVID-19?",
                "abstract": "Multiple randomized controlled trials have shown that hydroxychloroquine does not significantly reduce mortality or improve clinical outcomes in COVID-19 patients. The WHO discontinued hydroxychloroquine trials due to lack of efficacy evidence.",
                "answer": "No"
            },
            {
                "question": "Can COVID-19 vaccines prevent severe disease?",
                "abstract": "Clinical trials and real-world data demonstrate that COVID-19 vaccines significantly reduce the risk of severe disease, hospitalization, and death. Vaccine effectiveness against severe outcomes remains high even with variant emergence.",
                "answer": "Yes"
            },
            {
                "question": "Does COVID-19 affect neurological function?",
                "abstract": "Neurological symptoms including loss of taste and smell, headaches, and cognitive impairment have been reported in COVID-19 patients. Some studies suggest potential long-term neurological effects, though mechanisms remain under investigation.",
                "answer": "Yes"
            },
            {
                "question": "Is remdesivir effective against COVID-19?",
                "abstract": "Clinical trials show mixed results for remdesivir in COVID-19 treatment. While some studies indicate modest reduction in recovery time for hospitalized patients, evidence for mortality benefit is limited and controversial.",
                "answer": "Maybe"
            },
            {
                "question": "Can children transmit COVID-19 effectively?",
                "abstract": "Studies indicate that children can transmit SARS-CoV-2, though transmission rates may be lower than adults. School-age children show variable transmission patterns, with transmission more common in household settings than schools.",
                "answer": "Yes"
            },
            {
                "question": "Does COVID-19 cause permanent lung damage?",
                "abstract": "Follow-up studies of COVID-19 survivors show that while most patients recover lung function, some develop persistent pulmonary fibrosis and reduced exercise capacity. Long-term outcomes are still being studied as the pandemic is relatively recent.",
                "answer": "Maybe"
            },
            {
                "question": "Are COVID-19 antibody tests reliable for immunity assessment?",
                "abstract": "Antibody tests can detect past SARS-CoV-2 infection but correlation with immunity levels varies. Antibody levels decline over time, and the relationship between antibody presence and protection from reinfection is complex and not fully established.",
                "answer": "Maybe"
            },
            {
                "question": "Does vitamin D deficiency increase COVID-19 risk?",
                "abstract": "Observational studies suggest associations between vitamin D deficiency and increased COVID-19 severity, but causal relationships remain unclear. Randomized controlled trials on vitamin D supplementation for COVID-19 prevention show inconsistent results.",
                "answer": "Maybe"
            },
            {
                "question": "Can COVID-19 cause diabetes?",
                "abstract": "Emerging evidence suggests COVID-19 may trigger new-onset diabetes in some patients. Proposed mechanisms include direct pancreatic beta cell damage by SARS-CoV-2 and inflammatory responses affecting glucose metabolism, but long-term studies are needed.",
                "answer": "Maybe"
            },
            {
                "question": "Is COVID-19 primarily transmitted through airborne particles?",
                "abstract": "Scientific evidence strongly supports that SARS-CoV-2 spreads primarily through respiratory droplets and airborne particles. Indoor transmission in poorly ventilated spaces is particularly efficient, leading to airborne precautions in healthcare settings.",
                "answer": "Yes"
            },
            {
                "question": "Do face masks prevent COVID-19 transmission?",
                "abstract": "Multiple studies demonstrate that face masks significantly reduce transmission of SARS-CoV-2. Both surgical and cloth masks provide source control by reducing droplet emission, with N95 respirators offering highest protection levels.",
                "answer": "Yes"
            },
            {
                "question": "Can COVID-19 reinfection occur frequently?",
                "abstract": "Reinfection with SARS-CoV-2 can occur but appears relatively uncommon in the first year after initial infection. Most reinfections are milder than primary infections, though severity can vary with viral variants and individual immune responses.",
                "answer": "No"
            },
            {
                "question": "Does COVID-19 vaccination cause fertility problems?",
                "abstract": "Large-scale studies and surveillance data show no evidence that COVID-19 vaccines affect fertility in men or women. Pregnancy outcomes in vaccinated individuals are similar to unvaccinated populations, and vaccination is recommended during pregnancy.",
                "answer": "No"
            },
            {
                "question": "Is COVID-19 mortality higher in older adults?",
                "abstract": "Age is the strongest risk factor for severe COVID-19 outcomes. Case fatality rates increase dramatically with age, with adults over 65 years having significantly higher rates of hospitalization, ICU admission, and death compared to younger populations.",
                "answer": "Yes"
            }
        ]
        
        self.logger.info(f"Created COVID-19 dataset with {len(covid_questions)} questions")
        return covid_questions
    
    def load_existing_models(self):
        """Load the existing fine-tuned model and setup few-shot evaluator"""
        self.logger.info("Loading existing models...")
        
        # Load fine-tuned model
        model_path = "models/fine_tuned/final"
        if Path(model_path).exists():
            self.fine_tuned_model = GPT2LMHeadModel.from_pretrained(model_path)
            self.fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.fine_tuned_model.to(self.device)
            self.logger.info("Fine-tuned model loaded successfully")
        else:
            self.logger.error(f"Fine-tuned model not found at {model_path}")
            return False
            
        # Load base model for few-shot
        self.base_model = GPT2LMHeadModel.from_pretrained("gpt2")
        self.base_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        if self.base_tokenizer.pad_token is None:
            self.base_tokenizer.pad_token = self.base_tokenizer.eos_token
        self.base_model.to(self.device)
        self.logger.info("Base model loaded successfully")
        
        return True
    
    def evaluate_fine_tuned_transfer(self, covid_data: List[Dict]) -> float:
        """Evaluate fine-tuned model on COVID-19 data"""
        self.logger.info("Evaluating fine-tuned model on COVID-19 domain...")
        
        correct = 0
        total = len(covid_data)
        predictions = []
        
        for item in covid_data:
            # Format input like training data
            input_text = f"Question: {item['question']}\nAbstract: {item['abstract']}\nAnswer:"
            
            # Tokenize
            inputs = self.fine_tuned_tokenizer.encode(input_text, return_tensors="pt").to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.fine_tuned_model.generate(
                    inputs, 
                    max_new_tokens=5,
                    temperature=0.1,
                    pad_token_id=self.fine_tuned_tokenizer.eos_token_id,
                    do_sample=False
                )
            
            # Decode prediction
            prediction = self.fine_tuned_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            prediction = prediction.strip().split()[0] if prediction.strip() else ""
            
            # Clean prediction
            prediction = re.sub(r'[^\w]', '', prediction).lower()
            actual = item['answer'].lower()
            
            # Check if correct
            is_correct = prediction.startswith(actual[:2])  # Match first 2 chars (ye, no, ma)
            if is_correct:
                correct += 1
                
            predictions.append({
                'question': item['question'],
                'actual': item['answer'],
                'predicted': prediction,
                'correct': is_correct
            })
        
        accuracy = correct / total
        self.logger.info(f"Fine-tuned transfer accuracy: {accuracy:.3f} ({correct}/{total})")
        
        return accuracy, predictions
    
    def evaluate_few_shot_transfer(self, covid_data: List[Dict], shot_count: int = 5) -> float:
        """Evaluate few-shot approach on COVID-19 data"""
        self.logger.info(f"Evaluating {shot_count}-shot on COVID-19 domain...")
        
        # Load original PubMedQA examples for few-shot prompts
        original_examples = self.load_original_few_shot_examples()[:shot_count]
        
        correct = 0
        total = len(covid_data)
        predictions = []
        
        for item in covid_data:
            # Create few-shot prompt
            prompt = "Answer biomedical questions with Yes, No, or Maybe based on the abstract:\n\n"
            
            # Add examples
            for example in original_examples:
                prompt += f"Question: {example['question']}\n"
                prompt += f"Abstract: {example['abstract']}\n"
                prompt += f"Answer: {example['answer']}\n\n"
            
            # Add target question
            prompt += f"Question: {item['question']}\n"
            prompt += f"Abstract: {item['abstract']}\n"
            prompt += f"Answer:"
            
            # Tokenize
            inputs = self.base_tokenizer.encode(prompt, return_tensors="pt", max_length=1024, truncation=True).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.base_model.generate(
                    inputs, 
                    max_new_tokens=5,
                    temperature=0.1,
                    pad_token_id=self.base_tokenizer.eos_token_id,
                    do_sample=False
                )
            
            # Decode prediction
            prediction = self.base_tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
            prediction = prediction.strip().split()[0] if prediction.strip() else ""
            
            # Clean prediction
            prediction = re.sub(r'[^\w]', '', prediction).lower()
            actual = item['answer'].lower()
            
            # Check if correct
            is_correct = prediction.startswith(actual[:2])  # Match first 2 chars
            if is_correct:
                correct += 1
                
            predictions.append({
                'question': item['question'],
                'actual': item['answer'],
                'predicted': prediction,
                'correct': is_correct
            })
        
        accuracy = correct / total
        self.logger.info(f"Few-shot transfer accuracy: {accuracy:.3f} ({correct}/{total})")
        
        return accuracy, predictions
    
    def load_original_few_shot_examples(self) -> List[Dict]:
        """Load original PubMedQA examples for few-shot prompts"""
        # Load from your data directory or recreate sample examples
        examples = [
            {
                "question": "Does exercise improve cardiovascular health?",
                "abstract": "Regular aerobic exercise training has been consistently shown to improve cardiovascular outcomes including reduced blood pressure, improved lipid profiles, and enhanced endothelial function in both healthy individuals and those with cardiovascular disease.",
                "answer": "Yes"
            },
            {
                "question": "Is vitamin C effective for preventing common cold?",
                "abstract": "Multiple systematic reviews and meta-analyses have shown that regular vitamin C supplementation has minimal impact on common cold incidence in the general population, though it may reduce duration and severity in some cases.",
                "answer": "Maybe"
            },
            {
                "question": "Does smoking cause lung cancer?",
                "abstract": "Extensive epidemiological evidence demonstrates a strong causal relationship between tobacco smoking and lung cancer, with risk increasing proportionally to smoking duration and intensity. Smoking cessation significantly reduces lung cancer risk over time.",
                "answer": "Yes"
            },
            {
                "question": "Can meditation treat depression?",
                "abstract": "Clinical studies on mindfulness-based interventions show mixed results for depression treatment. While some trials demonstrate benefits comparable to other psychotherapies, evidence quality varies and meditation may be more effective as an adjunct rather than standalone treatment.",
                "answer": "Maybe"
            },
            {
                "question": "Do antibiotics work against viral infections?",
                "abstract": "Antibiotics are specifically designed to target bacterial pathogens and have no direct antiviral activity. Use of antibiotics for viral infections provides no clinical benefit and contributes to antibiotic resistance development.",
                "answer": "No"
            }
        ]
        return examples
    
    def run_cross_domain_experiment(self):
        """Run the complete cross-domain transfer experiment"""
        self.logger.info("Starting Cross-Domain Transfer Experiment")
        self.logger.info("="*60)
        
        # Create COVID-19 dataset
        covid_data = self.create_covid_dataset()
        
        # Load existing models
        if not self.load_existing_models():
            self.logger.error("Failed to load models")
            return
        
        # Evaluate both approaches
        fine_tuned_acc, ft_predictions = self.evaluate_fine_tuned_transfer(covid_data)
        few_shot_acc, fs_predictions = self.evaluate_few_shot_transfer(covid_data)
        
        # Store results
        self.results = {
            "experiment_type": "cross_domain_transfer",
            "source_domain": "PubMedQA (general biomedical)",
            "target_domain": "COVID-19 biomedical QA",
            "dataset_size": len(covid_data),
            "fine_tuned_accuracy": fine_tuned_acc,
            "few_shot_accuracy": few_shot_acc,
            "performance_difference": few_shot_acc - fine_tuned_acc,
            "relative_improvement": ((few_shot_acc - fine_tuned_acc) / fine_tuned_acc * 100) if fine_tuned_acc > 0 else 0,
            "winner": "few_shot" if few_shot_acc > fine_tuned_acc else "fine_tuned",
            "fine_tuned_predictions": ft_predictions,
            "few_shot_predictions": fs_predictions,
            "timestamp": datetime.now().isoformat()
        }
        
        # Save results
        self.save_results()
        
        # Create analysis
        self.create_transfer_analysis()
        
        return self.results
    
    def save_results(self):
        """Save experiment results"""
        output_dir = Path("results/experiments/cross_domain_transfer")
        output_dir.mkdir(exist_ok=True)
        
        # Save main results
        with open(output_dir / "transfer_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        self.logger.info(f"Results saved to {output_dir}/transfer_results.json")
    
    def create_transfer_analysis(self):
        """Create analysis comparing original vs transfer performance"""
        # Load original results for comparison
        original_few_shot_path = Path("results/experiments/few_shot_baseline/few_shot_results.json")
        original_fine_tuned_path = Path("results/experiments/fine_tuning_baseline/fine_tuning_results.json")
        
        if original_few_shot_path.exists() and original_fine_tuned_path.exists():
            with open(original_few_shot_path) as f:
                original_fs_results = json.load(f)
            with open(original_fine_tuned_path) as f:
                original_ft_results = json.load(f)
            
            # Get best few-shot from original
            best_original_fs = max(original_fs_results.values(), key=lambda x: x.get('accuracy', 0) if isinstance(x, dict) else 0)
            original_ft_acc = original_ft_results.get('accuracy', 0)
            
            # Create comparison visualization
            self.create_transfer_visualization(best_original_fs['accuracy'], original_ft_acc)
    
    def create_transfer_visualization(self, original_fs_acc: float, original_ft_acc: float):
        """Create visualization comparing original vs transfer performance"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Performance comparison
        methods = ['Fine-Tuning', 'Few-Shot (5-shot)']
        original_scores = [original_ft_acc, original_fs_acc]
        transfer_scores = [self.results['fine_tuned_accuracy'], self.results['few_shot_accuracy']]
        
        x = range(len(methods))
        width = 0.35
        
        bars1 = ax1.bar([i - width/2 for i in x], original_scores, width, label='Original (PubMedQA)', alpha=0.8)
        bars2 = ax1.bar([i + width/2 for i in x], transfer_scores, width, label='Transfer (COVID-19)', alpha=0.8)
        
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Domain Transfer Performance Comparison')
        ax1.set_xticks(x)
        ax1.set_xticklabels(methods)
        ax1.legend()
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                        f'{height:.3f}', ha='center', va='bottom')
        
        # Performance drop analysis
        ft_drop = original_ft_acc - self.results['fine_tuned_accuracy']
        fs_drop = original_fs_acc - self.results['few_shot_accuracy']
        
        drops = [ft_drop, fs_drop]
        colors = ['red' if x > 0 else 'green' for x in drops]
        
        bars = ax2.bar(methods, drops, color=colors, alpha=0.7)
        ax2.set_ylabel('Performance Drop (Original - Transfer)')
        ax2.set_title('Domain Transfer Robustness')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Add value labels
        for bar, drop in zip(bars, drops):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.01 if height > 0 else -0.02),
                    f'{height:+.3f}', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        
        # Save plot
        output_path = Path("results/experiments/cross_domain_transfer/transfer_analysis.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        self.logger.info(f"Transfer analysis plot saved to {output_path}")
        
        plt.show()
    
    def print_summary(self):
        """Print experiment summary"""
        print("\n" + "="*60)
        print("CROSS-DOMAIN TRANSFER EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Source Domain: {self.results['source_domain']}")
        print(f"Target Domain: {self.results['target_domain']}")
        print(f"Test Questions: {self.results['dataset_size']}")
        print(f"")
        print(f"TRANSFER PERFORMANCE:")
        print(f"  Fine-tuned model: {self.results['fine_tuned_accuracy']:.3f}")
        print(f"  Few-shot (5-shot): {self.results['few_shot_accuracy']:.3f}")
        print(f"  Performance gap: {self.results['performance_difference']:+.3f}")
        print(f"  Relative improvement: {self.results['relative_improvement']:+.1f}%")
        print(f"")
        print(f"WINNER: {self.results['winner'].replace('_', ' ').title()}")
        print("="*60)

# Main execution
if __name__ == "__main__":
    evaluator = CrossDomainTransferEvaluator()
    results = evaluator.run_cross_domain_experiment()
    evaluator.print_summary()