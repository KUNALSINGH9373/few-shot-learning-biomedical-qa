#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path
import subprocess

def main():
    """Run the complete experimental pipeline"""
    
    print("RUNNING COMPLETE FEW-SHOT LEARNING PIPELINE")
    print("=" * 60)
    
    steps = [
        {
            "name": "Few-Shot Experiments",
            "script": "scripts/run_few_shot_experiments.py",
            "config": "configs/few_shot_experiments.yaml"
        },
        {
            "name": "Fine-Tuning Experiments", 
            "script": "scripts/run_fine_tuning.py",
            "config": "configs/fine_tuning_experiments.yaml"
        },
        {
            "name": "Evaluation & Comparison",
            "script": "scripts/run_evaluation.py", 
            "config": "configs/evaluation_config.yaml"
        }
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"\n[STEP {i}] {step['name']}")
        print("-" * 40)
        
        cmd = ["python", step["script"], "--config", step["config"]]
        
        try:
            result = subprocess.run(cmd, check=True)
            print(f"[SUCCESS] {step['name']} completed")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {step['name']} failed: {e}")
            break
    
    print("\nPipeline completed!")

if __name__ == "__main__":
    main()
