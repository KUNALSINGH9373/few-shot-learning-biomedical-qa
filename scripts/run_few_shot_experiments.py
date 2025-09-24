#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import load_config, merge_configs
from utils.seed import set_seed
from utils.logging_utils import setup_logger

def main():
    parser = argparse.ArgumentParser(description="Run few-shot learning experiments")
    parser.add_argument("--config", default="configs/few_shot_experiments.yaml",
                       help="Configuration file path")
    parser.add_argument("--base_config", default="configs/base_config.yaml",
                       help="Base configuration file path")
    
    args = parser.parse_args()
    
    # Load configurations
    config = merge_configs(args.base_config, args.config)
    
    # Setup logging
    logger = setup_logger("few_shot_experiment", 
                         log_file=f"results/logs/few_shot_experiment.log")
    
    # Set random seed
    set_seed(config.get("random_seed", 42))
    
    logger.info(f"Starting experiment: {config['experiment']['name']}")
    
    # Import and run experiment with correct method
    try:
        from models.few_shot_model import FewShotPrompter
        
        # Extract parameters from config
        shot_counts = config['experiment_settings']['shot_counts']
        max_eval = config['experiment_settings']['max_eval_samples']
        
        logger.info(f"Running few-shot experiments with shots: {shot_counts}")
        logger.info(f"Max evaluation samples: {max_eval}")
        
        # Initialize and run with correct method name
        prompter = FewShotPrompter()
        results = prompter.run_few_shot_experiments(
            shot_counts=shot_counts,
            max_eval=max_eval
        )
        
        logger.info("Few-shot experiments completed successfully")
        logger.info("Results saved to results/few_shot_results.json")
        
        # Print summary
        print("\nEXPERIMENT SUMMARY:")
        print("=" * 40)
        for shot_count in shot_counts:
            if f"{shot_count}_shot" in results:
                accuracy = results[f"{shot_count}_shot"]['accuracy']
                print(f"{shot_count:2d}-shot: {accuracy:.3f}")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise

if __name__ == "__main__":
    main()
