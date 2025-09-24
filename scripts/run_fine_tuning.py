#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import load_config, merge_configs
from utils.seed import set_seed
from utils.logging_utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Run fine-tuning experiments")
    parser.add_argument("--config", default="configs/fine_tuning_experiments.yaml",
                       help="Configuration file path")
    parser.add_argument("--base_config", default="configs/base_config.yaml",
                       help="Base configuration file path")
    parser.add_argument("--eval_only", action="store_true",
                       help="Only evaluate existing model, skip training")
    
    args = parser.parse_args()
    
    # Load configurations
    config = merge_configs(args.base_config, args.config)
    
    # Setup logging
    logger = setup_logger("fine_tuning_experiment", 
                         log_file="results/logs/fine_tuning_experiment.log")
    
    # Set random seed
    set_seed(config.get("random_seed", 42))
    
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    
    try:
        from training.fine_tune import GPT2FineTuner, FineTunedEvaluator
        
        if not args.eval_only:
            # Fine-tune the model
            logger.info("Starting fine-tuning process...")
            
            fine_tuner = GPT2FineTuner(
                model_name=config['model']['name'],
                output_dir=config['output']['model_save_dir']
            )
            
            trainer, training_results = fine_tuner.fine_tune(config)
            
            logger.info("Fine-tuning completed successfully")
            logger.info(f"Final validation loss: {training_results['final_eval_results']['eval_loss']:.4f}")
            
            model_path = training_results['model_path']
        else:
            # Use existing model
            model_path = f"{config['output']['model_save_dir']}/final"
            logger.info(f"Using existing model: {model_path}")
        
        # Evaluate the fine-tuned model
        logger.info("Starting evaluation on test set...")
        
        evaluator = FineTunedEvaluator(model_path)
        
        # Load test data
        with open('data/processed/test.json', 'r') as f:
            test_data = json.load(f)
        
        # Evaluate
        max_eval = getattr(config.get('experiment_settings', {}), 'max_eval_samples', 100)
        results = evaluator.evaluate_on_test_set(test_data, max_eval=max_eval)
        
        logger.info(f"Fine-tuned model accuracy: {results['accuracy']:.3f}")
        
        # Save results
        experiment_dir = config['output']['experiment_dir']
        Path(experiment_dir).mkdir(parents=True, exist_ok=True)
        
        results_file = f"{experiment_dir}/fine_tuning_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to: {results_file}")
        
        # Print summary
        print("\\nFINE-TUNING EXPERIMENT SUMMARY:")
        print("=" * 40)
        print(f"Model: {config['model']['name']}")
        print(f"Training examples: {len(json.load(open('data/processed/train.json')))}")
        print(f"Test accuracy: {results['accuracy']:.3f} ({results['correct']}/{results['total']})")
        
        # Load few-shot results for comparison
        try:
            with open('results/few_shot_results.json', 'r') as f:
                few_shot_results = json.load(f)
            
            print("\\nCOMPARISON WITH FEW-SHOT:")
            print("-" * 30)
            
            # Find best few-shot result
            best_few_shot = 0
            best_shots = 0
            for key, result in few_shot_results.items():
                if '_shot' in key and result['accuracy'] > best_few_shot:
                    best_few_shot = result['accuracy']
                    best_shots = result['n_shots']
            
            print(f"Best few-shot ({best_shots}-shot): {best_few_shot:.3f}")
            print(f"Fine-tuned model: {results['accuracy']:.3f}")
            
            improvement = results['accuracy'] - best_few_shot
            if improvement > 0:
                print(f"Fine-tuning advantage: +{improvement:.3f} (+{improvement/best_few_shot*100:.1f}%)")
            else:
                print(f"Few-shot advantage: {abs(improvement):.3f} ({abs(improvement)/results['accuracy']*100:.1f}%)")
                
        except FileNotFoundError:
            print("Few-shot results not found - run few-shot experiments first for comparison")
        
        logger.info("Fine-tuning experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Fine-tuning experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()