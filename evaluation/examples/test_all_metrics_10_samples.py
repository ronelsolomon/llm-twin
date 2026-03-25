#!/usr/bin/env python3
"""
Simple function to test all evaluation metrics with 10 samples each
Usage: python test_all_metrics_10_samples.py
"""

import subprocess
import sys
from pathlib import Path

def test_all_metrics_10_samples(model_path: str = None):
    """
    Test all evaluation metrics with 10 samples each
    
    Args:
        model_path: Path to your model (defaults to DPO LLM Twin)
    """
    
    if model_path is None:
        model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"
    
    print("🚀 Testing All Evaluation Metrics - 10 Samples Each")
    print("="*60)
    print(f"Model: {model_path}")
    print("="*60)
    
    # All available evaluation types with their sample parameter names
    evaluations = [
        ("mmlu", "mmlu_samples", "MMLU-Pro"),
        ("hellaswag", "hellaswag_samples", "HellaSwag"),
        ("arc-c", "arc_c_samples", "ARC-C"),
        ("winogrande", "winogrande_samples", "Winogrande"),
        ("piqa", "piqa_samples", "PIQA"),
        ("ifeval", "ifeval_samples", "IFEval"),
        ("alpaca", "alpaca_samples", "AlpacaEval"),
        ("mt-bench", "mt_bench_samples", "MT-Bench"),
        ("gaia", "gaia_samples", "GAIA"),
        ("hallucination", "hallucination_samples", "Hallucination"),
        ("summarization", "summarization_samples", "Summarization"),
        ("ragas", "ragas_samples", "RAGAS"),
        ("ares", "ares_samples", "ARES"),
        ("safety", "safety_samples", "Safety")
    ]
    
    success_count = 0
    total_count = len(evaluations)
    
    for eval_type, sample_param, display_name in evaluations:
        print(f"\n📊 Testing {display_name}...")
        
        try:
            cmd = [
                "python", "run_evaluation.py",
                "--eval-type", eval_type,
                "--mmlu-model", model_path,  # Most evaluations use this parameter
                f"--{sample_param}", "10",
                "--output", f"quick_test_{eval_type}"
            ]
            
            # Run the evaluation from the correct directory
            result = subprocess.run(
                cmd, 
                cwd="/Users/ronel/Downloads/llm twin/evaluation",
                capture_output=True, 
                text=True, 
                timeout=180  # 3 minutes timeout per evaluation
            )
            
            if result.returncode == 0:
                print(f"  ✅ {display_name} - SUCCESS")
                success_count += 1
            else:
                print(f"  ❌ {display_name} - FAILED")
                if result.stderr:
                    print(f"     Error: {result.stderr[:150]}...")
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {display_name} - TIMEOUT")
        except Exception as e:
            print(f"  ❌ {display_name} - ERROR: {str(e)}")
    
    # Summary
    print("\n" + "="*60)
    print("🎯 SUMMARY")
    print("="*60)
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 All evaluations passed! Your model is ready for full testing.")
    elif success_count > total_count // 2:
        print("📈 Most evaluations passed. Check failed ones individually.")
    else:
        print("⚠️  Many evaluations failed. Check model and dependencies.")
    
    return success_count, total_count

if __name__ == "__main__":
    # You can specify a different model path here
    test_all_metrics_10_samples()
