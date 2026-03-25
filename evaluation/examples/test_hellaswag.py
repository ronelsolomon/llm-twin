#!/usr/bin/env python3
"""
Quick test for HellaSwag evaluation
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from hellaswag_evaluator import HellaSwagEvaluator

def main():
    print("="*50)
    print("HellaSwag Quick Test")
    print("="*50)
    
    # Test with small sample
    evaluator = HellaSwagEvaluator("/Users/ronel/Downloads/llm twin/dpo_llm_twin")
    dataset = evaluator.load_hellaswag_dataset()
    
    # Test just 10 questions
    results = evaluator.evaluate_dataset(dataset, num_samples=10, output_dir="hellaswag_test")
    
    print(f"\nTest completed! Accuracy: {results['overall_accuracy']:.2%}")

if __name__ == "__main__":
    main()
