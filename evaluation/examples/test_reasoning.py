#!/usr/bin/env python3
"""
Quick test for all reasoning benchmarks
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from hellaswag_evaluator import HellaSwagEvaluator
from arc_c_evaluator import ARCCEvaluator
from winogrande_evaluator import WinograndeEvaluator
from piqa_evaluator import PIQAEvaluator

def main():
    print("="*60)
    print("Reasoning Benchmarks Quick Test")
    print("="*60)
    
    model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin"
    
    # Test HellaSwag
    print("\n1. Testing HellaSwag...")
    hellaswag_eval = HellaSwagEvaluator(model_path)
    hellaswag_dataset = hellaswag_eval.load_hellaswag_dataset()
    hellaswag_results = hellaswag_eval.evaluate_dataset(hellaswag_dataset, num_samples=10, output_dir="test_results/hellaswag")
    print(f"   HellaSwag Accuracy: {hellaswag_results['overall_accuracy']:.2%}")
    
    # Test ARC-C
    print("\n2. Testing ARC-C...")
    arc_c_eval = ARCCEvaluator(model_path)
    arc_c_dataset = arc_c_eval.load_arc_c_dataset()
    arc_c_results = arc_c_eval.evaluate_dataset(arc_c_dataset, num_samples=10, output_dir="test_results/arc_c")
    print(f"   ARC-C Accuracy: {arc_c_results['overall_accuracy']:.2%}")
    
    # Test Winogrande
    print("\n3. Testing Winogrande...")
    winogrande_eval = WinograndeEvaluator(model_path)
    winogrande_dataset = winogrande_eval.load_winogrande_dataset()
    winogrande_results = winogrande_eval.evaluate_dataset(winogrande_dataset, num_samples=10, output_dir="test_results/winogrande")
    print(f"   Winogrande Accuracy: {winogrande_results['overall_accuracy']:.2%}")
    
    # Test PIQA
    print("\n4. Testing PIQA...")
    piqa_eval = PIQAEvaluator(model_path)
    piqa_dataset = piqa_eval.load_piqa_dataset()
    piqa_results = piqa_eval.evaluate_dataset(piqa_dataset, num_samples=10, output_dir="test_results/piqa")
    print(f"   PIQA Accuracy: {piqa_results['overall_accuracy']:.2%}")
    
    # Summary
    print("\n" + "="*60)
    print("Reasoning Test Summary")
    print("="*60)
    print(f"HellaSwag (Commonsense): {hellaswag_results['overall_accuracy']:.2%}")
    print(f"ARC-C (Science Reasoning): {arc_c_results['overall_accuracy']:.2%}")
    print(f"Winogrande (Pronoun Resolution): {winogrande_results['overall_accuracy']:.2%}")
    print(f"PIQA (Physical Commonsense): {piqa_results['overall_accuracy']:.2%}")
    
    # Calculate average
    avg_accuracy = (hellaswag_results['overall_accuracy'] + 
                   arc_c_results['overall_accuracy'] + 
                   winogrande_results['overall_accuracy'] + 
                   piqa_results['overall_accuracy']) / 4
    print(f"\nAverage Reasoning Accuracy: {avg_accuracy:.2%}")
    
    print("="*60)

if __name__ == "__main__":
    main()
