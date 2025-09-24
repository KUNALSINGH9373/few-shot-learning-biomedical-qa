
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd

class ExperimentAnalyzer:
    def __init__(self, results_file='results/few_shot_results.json'):
        with open(results_file, 'r') as f:
            self.results = json.load(f)
    
    def analyze_accuracy_trends(self):
        """Analyze how accuracy changes with shot count"""
        
        shot_counts = []
        accuracies = []
        
        for key, result in self.results.items():
            if '_shot' in key:
                n_shots = result['n_shots']
                accuracy = result['accuracy']
                shot_counts.append(n_shots)
                accuracies.append(accuracy)
        
        # Sort by shot count
        sorted_data = sorted(zip(shot_counts, accuracies))
        shot_counts, accuracies = zip(*sorted_data)
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        plt.plot(shot_counts, accuracies, 'b-o', linewidth=2, markersize=8)
        plt.xlabel('Number of Few-shot Examples')
        plt.ylabel('Accuracy')
        plt.title('Few-shot Learning Performance vs Number of Examples')
        plt.grid(True, alpha=0.3)
        
        # Add value labels
        for x, y in zip(shot_counts, accuracies):
            plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                        xytext=(0,10), ha='center')
        
        plt.tight_layout()
        plt.savefig('results/accuracy_vs_shots.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return shot_counts, accuracies
    
    def analyze_error_patterns(self):
        """Analyze error patterns across different shot counts"""
        
        error_analysis = {}
        
        for key, result in self.results.items():
            if '_shot' in key:
                n_shots = result['n_shots']
                predictions = result['predictions']
                
                # Confusion matrix
                confusion = defaultdict(lambda: defaultdict(int))
                errors_by_type = defaultdict(list)
                
                for pred in predictions:
                    actual = pred['actual']
                    predicted = pred['predicted']
                    confusion[actual][predicted] += 1
                    
                    if not pred['correct']:
                        errors_by_type[f"{actual}_to_{predicted}"].append({
                            'question': pred['question'],
                            'generated': pred['generated_text']
                        })
                
                error_analysis[n_shots] = {
                    'confusion_matrix': dict(confusion),
                    'error_examples': dict(errors_by_type)
                }
        
        return error_analysis
    
    def create_summary_report(self):
        """Create a summary report of the experiments"""
        
        print("=" * 60)
        print("FEW-SHOT LEARNING EXPERIMENT SUMMARY")
        print("=" * 60)
        
        # Overall performance trend
        shot_counts, accuracies = self.analyze_accuracy_trends()
        
        print(f"\nPerformance by Shot Count:")
        print("-" * 30)
        for shots, acc in zip(shot_counts, accuracies):
            print(f"{shots:2d}-shot: {acc:.3f} ({acc*100:.1f}%)")
        
        # Performance improvement analysis
        print(f"Performance Improvements:")
        print("-" * 30)
        baseline_acc = accuracies[0]  # 0-shot
        for i, (shots, acc) in enumerate(zip(shot_counts[1:], accuracies[1:])):
            improvement = acc - baseline_acc
            relative_improvement = (acc - baseline_acc) / baseline_acc * 100
            print(f"{shots:2d}-shot vs 0-shot: +{improvement:.3f} (+{relative_improvement:.1f}%)")
        
        # Best performance
        best_idx = accuracies.index(max(accuracies))
        best_shots = shot_counts[best_idx]
        best_acc = accuracies[best_idx]
        print(f"Best Performance: {best_shots}-shot with {best_acc:.3f} accuracy")
        
        # Error analysis for best model
        error_analysis = self.analyze_error_patterns()
        if best_shots in error_analysis:
            confusion = error_analysis[best_shots]['confusion_matrix']
            print(f"Confusion Matrix for {best_shots}-shot:")
            print("-" * 30)
            
            all_labels = set()
            for actual_label in confusion:
                all_labels.update(confusion[actual_label].keys())
                all_labels.add(actual_label)
            all_labels = sorted(list(all_labels))
            
            # Print confusion matrix
            print(f"{'':>8}", end="")
            for label in all_labels:
                print(f"{label:>8}", end="")
            print()
            
            for actual in all_labels:
                print(f"{actual:>8}", end="")
                for predicted in all_labels:
                    count = confusion[actual].get(predicted, 0)
                    print(f"{count:>8}", end="")
                print()
        
        # Sample errors
        print(f"\nSample Errors from {best_shots}-shot:")
        print("-" * 30)
        if best_shots in error_analysis:
            error_examples = error_analysis[best_shots]['error_examples']
            for error_type, examples in list(error_examples.items())[:3]:
                if examples:
                    print(f"\n{error_type.upper()}:")
                    example = examples[0]
                    print(f"Q: {example['question'][:100]}...")
                    print(f"Generated: '{example['generated']}'")
    
    def save_analysis(self, output_file='results/experiment_analysis.json'):
        """Save detailed analysis to file"""
        
        analysis = {
            'accuracy_trends': {},
            'error_patterns': self.analyze_error_patterns(),
            'summary_stats': {}
        }
        
        # Add accuracy trends
        shot_counts, accuracies = self.analyze_accuracy_trends()
        for shots, acc in zip(shot_counts, accuracies):
            analysis['accuracy_trends'][f"{shots}_shot"] = acc
        
        # Add summary stats
        analysis['summary_stats'] = {
            'best_performance': {
                'shots': shot_counts[accuracies.index(max(accuracies))],
                'accuracy': max(accuracies)
            },
            'baseline_accuracy': accuracies[0],
            'max_improvement': max(accuracies) - accuracies[0]
        }
        
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Analysis saved to {output_file}")

def main():
    """Run analysis on experiment results"""
    
    try:
        analyzer = ExperimentAnalyzer()
        analyzer.create_summary_report()
        analyzer.save_analysis()
    except FileNotFoundError:
        print("[ERROR] No results file found. Run few_shot_prompting.py first!")

if __name__ == "__main__":
    main()