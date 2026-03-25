#!/usr/bin/env python3
"""
Quick test for ROUGE Summarization evaluation
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from rouge_summarization_evaluator import SummarizationEvaluator

def main():
    print("="*60)
    print("ROUGE Summarization Quick Test")
    print("="*60)
    
    model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin"
    
    # Test ROUGE Summarization evaluation
    print("\nTesting ROUGE Summarization...")
    sum_eval = SummarizationEvaluator(model_path)
    dataset = sum_eval.load_summarization_dataset()
    
    # Test with small sample (5 examples)
    results = sum_eval.evaluate_dataset(dataset, num_samples=5, output_dir="test_results/rouge_summarization")
    
    print(f"\nROUGE Summarization Test Results:")
    print(f"Overall Score: {results['overall_score']:.3f}/1.0")
    
    # ROUGE scores
    print(f"\nROUGE Performance:")
    for rouge_type, score in results['rouge_averages'].items():
        print(f"  {rouge_type.upper()}: {score:.3f}")
    
    # Category breakdown
    print(f"\nCategory Performance:")
    for category, scores in sorted(results['category_averages'].items(), key=lambda x: x[1]['overall'], reverse=True):
        print(f"  {category}: {scores['overall']:.3f} (ROUGE-1: {scores['rouge']['rouge-1']:.3f})")
    
    # Performance assessment
    rouge_1 = results['rouge_averages']['rouge-1']
    rouge_2 = results['rouge_averages']['rouge-2']
    rouge_l = results['rouge_averages']['rouge-l']
    
    print(f"\nSummarization Quality Assessment:")
    if rouge_1 > 0.4:
        print("✅ Excellent ROUGE-1 score - Good content overlap")
    elif rouge_1 > 0.3:
        print("📈 Good ROUGE-1 score - Decent content overlap")
    elif rouge_1 > 0.2:
        print("⚠️  Fair ROUGE-1 score - Limited content overlap")
    else:
        print("❌ Poor ROUGE-1 score - Low content overlap")
    
    if rouge_2 > 0.2:
        print("✅ Strong ROUGE-2 score - Good phrase overlap")
    elif rouge_2 > 0.1:
        print("📈 Moderate ROUGE-2 score - Some phrase overlap")
    else:
        print("⚠️  Weak ROUGE-2 score - Limited phrase overlap")
    
    if rouge_l > 0.35:
        print("✅ Strong ROUGE-L score - Good sentence structure")
    elif rouge_l > 0.25:
        print("📈 Moderate ROUGE-L score - Decent sentence structure")
    else:
        print("⚠️  Weak ROUGE-L score - Limited sentence structure")
    
    # Show example
    best_result = max(results["detailed_results"], key=lambda x: x["overall_score"])
    print(f"\nBest Summary Example:")
    print(f"Category: {best_result['category']}")
    print(f"Generated: {best_result['generated_summary']}")
    print(f"Reference: {best_result['reference_summary']}")
    print(f"ROUGE-1: {best_result['rouge_scores']['rouge-1']:.3f}")
    
    print("="*60)

if __name__ == "__main__":
    main()
