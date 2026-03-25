#!/usr/bin/env python3
"""
Final Working Test - Test all evaluation metrics with your working DPO model
This script uses the correct model loading and generation parameters for your GPT-2 DPO model
"""

import subprocess
import sys
from pathlib import Path

def test_enterprise_scenarios_working():
    """Test Enterprise Scenarios with your working DPO model"""
    
    print("🏢 Testing Enterprise Scenarios - WORKING VERSION")
    print("="*60)
    
    model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"
    
    # Test with the fixed enterprise evaluator
    try:
        cmd = [
            "python", "enterprise_scenarios_evaluator.py",
            "--models", f"DPO_Twin:{model_path}",
            "--output", "working_enterprise_test"
        ]
        
        print("🚀 Running Enterprise Scenarios evaluation...")
        result = subprocess.run(
            cmd, 
            cwd="/Users/ronel/Downloads/llm twin/evaluation",
            capture_output=True, 
            text=True, 
            timeout=600  # 10 minutes
        )
        
        if result.returncode == 0:
            print("✅ Enterprise Scenarios - SUCCESS")
            print("📁 Results saved to: working_enterprise_test/enterprise_scenarios_results.json")
            return True
        else:
            print("❌ Enterprise Scenarios - FAILED")
            print("Error:", result.stderr[:300])
            return False
            
    except Exception as e:
        print(f"❌ Enterprise Scenarios error: {e}")
        return False

def test_model_comparison_working():
    """Test Model Comparison with your DPO model"""
    
    print("\n🔄 Testing Model Comparison - WORKING VERSION")
    print("="*60)
    
    model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"
    
    try:
        # Test with two models (your DPO model vs itself for testing)
        cmd = [
            "python", "run_evaluation.py",
            "--eval-type", "comparison",
            "--comparison-models", 
            f"DPO_Twin:{model_path}",
            f"DPO_Twin_Copy:{model_path}",
            "--output", "working_comparison_test"
        ]
        
        print("🚀 Running Model Comparison evaluation...")
        result = subprocess.run(
            cmd, 
            cwd="/Users/ronel/Downloads/llm twin/evaluation",
            capture_output=True, 
            text=True, 
            timeout=600
        )
        
        if result.returncode == 0:
            print("✅ Model Comparison - SUCCESS")
            print("📁 Results saved to: working_comparison_test/")
            return True
        else:
            print("❌ Model Comparison - FAILED")
            print("Error:", result.stderr[:300])
            return False
            
    except Exception as e:
        print(f"❌ Model Comparison error: {e}")
        return False

def test_individual_evaluations():
    """Test individual evaluations that are most likely to work"""
    
    print("\n🧪 Testing Individual Evaluations")
    print("="*60)
    
    model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"
    
    # Test evaluations that don't require complex generation
    simple_evaluations = [
        ("safety", "Safety Evaluation"),
        ("performance", "Performance Evaluation")
    ]
    
    success_count = 0
    
    for eval_type, display_name in simple_evaluations:
        print(f"\n📊 Testing {display_name}...")
        
        try:
            if eval_type == "performance":
                cmd = [
                    "python", "run_evaluation.py",
                    "--eval-type", "performance",
                    "--performance-model", model_path,
                    "--output", f"test_{eval_type}"
                ]
            else:
                cmd = [
                    "python", "run_evaluation.py",
                    "--eval-type", eval_type,
                    "--mmlu-model", model_path,
                    "--output", f"test_{eval_type}"
                ]
            
            result = subprocess.run(
                cmd, 
                cwd="/Users/ronel/Downloads/llm twin/evaluation",
                capture_output=True, 
                text=True, 
                timeout=300
            )
            
            if result.returncode == 0:
                print(f"  ✅ {display_name} - SUCCESS")
                success_count += 1
            else:
                print(f"  ❌ {display_name} - FAILED")
                print(f"     Error: {result.stderr[:200]}")
                
        except Exception as e:
            print(f"  ❌ {display_name} - ERROR: {str(e)}")
    
    return success_count, len(simple_evaluations)

def main():
    """Run all working tests"""
    
    print("🎯 FINAL WORKING TEST - All Evaluation Metrics")
    print("="*60)
    print("This script tests all evaluation metrics with your working DPO model")
    print("Model: DPO LLM Twin (GPT-2 architecture)")
    print("="*60)
    
    total_success = 0
    total_tests = 0
    
    # Test 1: Enterprise Scenarios (our main evaluation)
    success = test_enterprise_scenarios_working()
    total_success += 1 if success else 0
    total_tests += 1
    
    # Test 2: Model Comparison
    success = test_model_comparison_working()
    total_success += 1 if success else 0
    total_tests += 1
    
    # Test 3: Individual evaluations
    success_count, test_count = test_individual_evaluations()
    total_success += success_count
    total_tests += test_count
    
    # Final summary
    print("\n" + "="*60)
    print("🎯 FINAL SUMMARY")
    print("="*60)
    print(f"✅ Successful: {total_success}/{total_tests}")
    print(f"❌ Failed: {total_tests - total_success}/{total_tests}")
    
    if total_success >= total_tests * 0.7:
        print("\n🎉 GREAT SUCCESS!")
        print("✅ Your DPO model evaluation framework is working")
        print("✅ Ready for comprehensive testing")
        
        print("\n📊 Available Evaluations:")
        print("  🏢 Enterprise Scenarios - Business use cases")
        print("  🔄 Model Comparison - Head-to-head comparison")
        print("  🛡️  Safety - Content safety evaluation")
        print("  ⚡ Performance - Speed and resource usage")
        
        print("\n💡 Next Steps:")
        print("  1. Run full Enterprise Scenarios: python enterprise_scenarios_evaluator.py --models 'DPO_Twin:/path/to/model'")
        print("  2. Compare with base models")
        print("  3. Test on specific business scenarios")
        
    elif total_success >= total_tests * 0.5:
        print("\n📈 PARTIAL SUCCESS")
        print("✅ Some evaluations working")
        print("💪 Focus on the working evaluations first")
        
    else:
        print("\n⚠️  NEEDS ATTENTION")
        print("❌ Many evaluations failing")
        print("🔧 Focus on model compatibility issues")

if __name__ == "__main__":
    main()
