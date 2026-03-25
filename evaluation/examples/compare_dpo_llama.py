#!/usr/bin/env python3
"""
Compare DPO-Llama with other models using the enhanced evaluation framework
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from model_comparison_evaluator import ModelComparisonEvaluator

def main():
    print("="*80)
    print("DPO-Llama Model Comparison")
    print("="*80)
    
    # Model configurations for comparison
    model_configs = [
        {
            "name": "DPO-Llama",
            "path": "/Users/ronel/Downloads/llm twin/dpo_llm_twin"
        },
        {
            "name": "DPO-Llama-Improved",
            "path": "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved"
        }
    ]
    
    print("Comparing models:")
    for config in model_configs:
        print(f"  • {config['name']}: {config['path']}")
    
    print("\nStarting model comparison evaluation...")
    
    # Initialize evaluator
    evaluator = ModelComparisonEvaluator(model_configs)
    
    # Run comparison
    results = evaluator.evaluate_dataset(
        evaluator.load_comparison_dataset(), 
        output_dir="dpo_llama_comparison"
    )
    
    print("\n" + "="*80)
    print("DPO-Llama Comparison Results")
    print("="*80)
    
    # Overall rankings
    print("\nOverall Model Rankings:")
    sorted_models = sorted(results['overall_model_scores'].items(), key=lambda x: x[1], reverse=True)
    for rank, (model_name, score) in enumerate(sorted_models, 1):
        win_rate = results['win_rates'][model_name]
        print(f"  {rank}. {model_name}: {score:.3f} (Win Rate: {win_rate:.1%})")
    
    # Performance by task type
    print("\nPerformance by Task Type:")
    for task_type, avg_score in sorted(results['task_type_averages'].items(), 
                                         key=lambda x: x[1], reverse=True):
        print(f"  {task_type}: {avg_score:.3f}")
    
    # DPO-Llama specific performance
    dpo_score = results['overall_model_scores'].get('DPO-Llama', 0.0)
    
    print(f"\nDPO-Llama Performance:")
    print(f"  Overall Score: {dpo_score:.3f}")
    print(f"  Win Rate: {results['win_rates'].get('DPO-Llama', 0.0):.1%}")
    
    # Assessment
    print(f"\nAssessment:")
    if dpo_score > 0.7:
        print("🏆 EXCELLENT: DPO-Llama demonstrates superior performance")
    elif dpo_score > 0.6:
        print("🎯 GOOD: DPO-Llama shows strong capabilities")
    elif dpo_score > 0.5:
        print("📊 FAIR: DPO-Llama has adequate performance")
    else:
        print("⚠️  NEEDS IMPROVEMENT: DPO-Llama requires optimization")
    
    # Comparison insights
    print(f"\nKey Insights:")
    print(f"• DPO-Llama vs Base-Llama: {dpo_score - results['overall_model_scores'].get('Base-Llama', 0.0):.3f} points difference")
    print(f"• DPO-Llama vs Meta-Llama: {dpo_score - results['overall_model_scores'].get('Meta-Llama', 0.0):.3f} points difference")
    
    # Recommendations
    print(f"\nRecommendations:")
    if dpo_score < 0.6:
        print("• Consider additional fine-tuning on task-specific datasets")
        print("• Review prompt engineering for better instruction following")
        print("• Analyze specific task types where performance is low")
    
    print("="*80)

if __name__ == "__main__":
    main()
