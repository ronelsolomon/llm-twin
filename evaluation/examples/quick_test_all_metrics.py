#!/usr/bin/env python3
"""
Quick Test Script - Run all evaluation metrics with 10 samples each
Tests your DPO model across all available benchmarks quickly
"""

import sys
import os
from pathlib import Path
import subprocess
import json

# Add the evaluation directory to the path
sys.path.append(str(Path(__file__).parent.parent))

def run_quick_evaluation(model_path: str, output_dir: str = "quick_test_results"):
    """Run all evaluation metrics with 10 samples each"""
    
    print("🚀 Quick Evaluation Test - 10 samples per metric")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    # List of all evaluation types that support num_samples
    evaluations = [
        ("mmlu", "MMLU-Pro"),
        ("hellaswag", "HellaSwag"),
        ("arc-c", "ARC-C"),
        ("winogrande", "Winogrande"),
        ("piqa", "PIQA"),
        ("ifeval", "IFEval"),
        ("alpaca", "AlpacaEval"),
        ("mt-bench", "MT-Bench"),
        ("gaia", "GAIA"),
        ("hallucination", "Hallucination"),
        ("summarization", "Summarization"),
        ("ragas", "RAGAS"),
        ("ares", "ARES"),
        ("safety", "Safety")
    ]
    
    results = {}
    
    for eval_type, eval_name in evaluations:
        print(f"\n📊 Running {eval_name} evaluation...")
        
        try:
            # Build command
            cmd = [
                "python", "run_evaluation.py",
                f"--eval-type", eval_type,
                f"--mmlu-model", model_path,  # Most evals use this parameter
                f"--{eval_type.replace('-', '_')}-samples", "10",
                "--output", output_dir
            ]
            
            # Handle special cases
            if eval_type in ["mmlu", "hellaswag", "arc-c", "winogrande", "piqa", "ifeval", "alpaca", "mt-bench", "gaia", "hallucination", "summarization", "ragas", "ares", "safety"]:
                # These use --mmlu-model parameter
                pass
            else:
                print(f"  ⚠️  {eval_name} may need different parameters")
                continue
            
            # Run evaluation
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"  ✅ {eval_name} completed successfully")
                results[eval_type] = {"status": "success", "output": result.stdout}
            else:
                print(f"  ❌ {eval_name} failed: {result.stderr}")
                results[eval_type] = {"status": "failed", "error": result.stderr}
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {eval_name} timed out")
            results[eval_type] = {"status": "timeout"}
        except Exception as e:
            print(f"  ❌ {eval_name} error: {str(e)}")
            results[eval_type] = {"status": "error", "error": str(e)}
    
    # Save results
    results_file = Path(output_dir) / "quick_test_results.json"
    results_file.parent.mkdir(exist_ok=True)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 Quick Test Summary")
    print("="*60)
    
    successful = sum(1 for r in results.values() if r["status"] == "success")
    failed = sum(1 for r in results.values() if r["status"] in ["failed", "error", "timeout"])
    
    print(f"✅ Successful: {successful}")
    print(f"❌ Failed: {failed}")
    print(f"📁 Results saved to: {results_file}")
    
    if successful > 0:
        print("\n🏆 Successful Evaluations:")
        for eval_type, result in results.items():
            if result["status"] == "success":
                print(f"  ✅ {eval_type}")
    
    if failed > 0:
        print("\n❌ Failed Evaluations:")
        for eval_type, result in results.items():
            if result["status"] != "success":
                print(f"  ❌ {eval_type}: {result['status']}")
    
    return results

def run_individual_tests(model_path: str):
    """Run individual evaluation tests for better debugging"""
    
    print("\n🔧 Running Individual Tests (10 samples each)")
    print("="*50)
    
    # Test core evaluations first
    core_tests = [
        ("mmlu", "MMLU-Pro"),
        ("hellaswag", "HellaSwag"),
        ("safety", "Safety"),
        ("ifeval", "IFEval")
    ]
    
    for eval_type, eval_name in core_tests:
        print(f"\n🧪 Testing {eval_name}...")
        
        try:
            cmd = [
                "python", "run_evaluation.py",
                f"--eval-type", eval_type,
                f"--mmlu-model", model_path,
                f"--{eval_type.replace('-', '_')}-samples", "10",
                "--output", f"test_{eval_type}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                print(f"  ✅ {eval_name} - SUCCESS")
            else:
                print(f"  ❌ {eval_name} - FAILED")
                print(f"     Error: {result.stderr[:200]}...")
                
        except Exception as e:
            print(f"  ❌ {eval_name} - ERROR: {str(e)}")

def main():
    """Main function"""
    
    # Default to your DPO model
    default_model = "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"
    
    print("🎯 Quick Evaluation Test - All Metrics (10 samples)")
    print("="*60)
    print("This script will test all evaluation metrics with just 10 samples each.")
    print("It's perfect for quickly verifying your model works across all benchmarks.")
    print()
    
    # Check if model exists
    model_path = default_model
    if not Path(model_path).exists():
        print(f"❌ Model not found: {model_path}")
        print("Please update the model path in the script")
        return
    
    print(f"📂 Using model: {model_path}")
    
    # Option to run individual tests first
    choice = input("\nRun individual core tests first? (y/n): ").lower().strip()
    
    if choice == 'y':
        run_individual_tests(model_path)
        
        continue_choice = input("\nContinue with full evaluation? (y/n): ").lower().strip()
        if continue_choice != 'y':
            print("👋 Exiting...")
            return
    
    # Run full quick evaluation
    results = run_quick_evaluation(model_path)
    
    print("\n🎉 Quick evaluation completed!")
    print("\nNext steps:")
    print("1. Check the results in quick_test_results.json")
    print("2. For successful evaluations, run full tests with more samples")
    print("3. For failed evaluations, check the error messages and debug")

if __name__ == "__main__":
    main()
