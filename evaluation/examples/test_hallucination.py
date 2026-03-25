#!/usr/bin/env python3
"""
Quick test for Hallucination Leaderboard evaluation
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from hallucination_evaluator import HallucinationEvaluator

def main():
    print("="*60)
    print("Hallucination Leaderboard Quick Test")
    print("="*60)
    
    model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin"
    
    # Test Hallucination evaluation
    print("\nTesting Hallucination Leaderboard...")
    halluc_eval = HallucinationEvaluator(model_path)
    dataset = halluc_eval.load_hallucination_dataset()
    
    # Test with small sample (2 tasks from each category)
    results = halluc_eval.evaluate_dataset(dataset, num_samples=10, output_dir="test_results/hallucination")
    
    print(f"\nHallucination Test Results:")
    print(f"Overall Reliability: {results['overall_reliability']:.3f}/1.0")
    print(f"Hallucination Score: {results['overall_hallucination_score']:.3f}/1.0 (lower is better)")
    
    # Category breakdown
    print(f"\nCategory Performance:")
    for category, reliability in sorted(results['category_averages'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {category}: {reliability:.3f}")
    
    # Task type breakdown
    print(f"\nTask Type Performance:")
    for task_type, reliability in sorted(results['task_type_averages'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {task_type}: {reliability:.3f}")
    
    print(f"\nHallucination Risk Analysis:")
    if results['overall_reliability'] > 0.8:
        print("✅ Excellent reliability - Low hallucination risk")
    elif results['overall_reliability'] > 0.6:
        print("📈 Good reliability - Moderate hallucination risk")
    elif results['overall_reliability'] > 0.4:
        print("⚠️  Fair reliability - Notable hallucination risk")
    else:
        print("❌ Poor reliability - High hallucination risk")
    
    print("="*60)

if __name__ == "__main__":
    main()
