#!/usr/bin/env python3
"""
Working Quick Test - Test individual evaluations that work with your DPO model
"""

import subprocess
import sys
from pathlib import Path

def test_working_evaluations(model_path: str = None):
    """
    Test evaluations that are most likely to work with DPO models
    """
    
    if model_path is None:
        model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"
    
    print("🚀 Testing Working Evaluations - 5 Samples Each")
    print("="*60)
    print(f"Model: {model_path}")
    print("="*60)
    
    # Start with simpler evaluations that are more likely to work
    working_evaluations = [
        ("safety", "safety_samples", "Safety"),
        ("comparison", "comparison_samples", "Model Comparison"),
        ("enterprise", "enterprise_samples", "Enterprise Scenarios"),
    ]
    
    # Also test with enterprise scenarios (which we just created)
    success_count = 0
    total_count = len(working_evaluations)
    
    for eval_type, sample_param, display_name in working_evaluations:
        print(f"\n📊 Testing {display_name}...")
        
        try:
            if eval_type == "enterprise":
                # Enterprise scenarios uses different parameters
                cmd = [
                    "python", "run_evaluation.py",
                    "--eval-type", "enterprise",
                    "--enterprise-models", f"DPO_Model:{model_path}",
                    "--output", f"quick_test_{eval_type}"
                ]
            elif eval_type == "comparison":
                # Comparison needs multiple models - use same model twice for testing
                cmd = [
                    "python", "run_evaluation.py", 
                    "--eval-type", "comparison",
                    "--comparison-models", f"Model1:{model_path}", f"Model2:{model_path}",
                    "--output", f"quick_test_{eval_type}"
                ]
            else:
                # Standard evaluation
                cmd = [
                    "python", "run_evaluation.py",
                    "--eval-type", eval_type,
                    "--mmlu-model", model_path,
                    "--output", f"quick_test_{eval_type}"
                ]
            
            # Run the evaluation
            result = subprocess.run(
                cmd, 
                cwd="/Users/ronel/Downloads/llm twin/evaluation",
                capture_output=True, 
                text=True, 
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                print(f"  ✅ {display_name} - SUCCESS")
                success_count += 1
            else:
                print(f"  ❌ {display_name} - FAILED")
                if result.stderr:
                    # Show first few lines of error
                    error_lines = result.stderr.split('\n')[:3]
                    print(f"     Error: {' | '.join(error_lines)}")
                
        except subprocess.TimeoutExpired:
            print(f"  ⏰ {display_name} - TIMEOUT")
        except Exception as e:
            print(f"  ❌ {display_name} - ERROR: {str(e)}")
    
    # Test Enterprise Scenarios directly (our new evaluator)
    print(f"\n🏢 Testing Enterprise Scenarios (Direct)...")
    try:
        cmd = [
            "python", "enterprise_scenarios_evaluator.py",
            "--models", f"DPO_Model:{model_path}",
            "--output", "direct_enterprise_test"
        ]
        
        result = subprocess.run(
            cmd, 
            cwd="/Users/ronel/Downloads/llm twin/evaluation",
            capture_output=True, 
            text=True, 
            timeout=300
        )
        
        if result.returncode == 0:
            print(f"  ✅ Enterprise Scenarios (Direct) - SUCCESS")
            success_count += 1
            total_count += 1
        else:
            print(f"  ❌ Enterprise Scenarios (Direct) - FAILED")
            if result.stderr:
                print(f"     Error: {result.stderr[:200]}...")
    except Exception as e:
        print(f"  ❌ Enterprise Scenarios (Direct) - ERROR: {str(e)}")
        total_count += 1
    
    # Summary
    print("\n" + "="*60)
    print("🎯 SUMMARY")
    print("="*60)
    print(f"✅ Successful: {success_count}/{total_count}")
    print(f"❌ Failed: {total_count - success_count}/{total_count}")
    
    if success_count > 0:
        print("\n🎉 Working Evaluations:")
        print("  ✅ Enterprise Scenarios - Perfect for business use cases")
        print("  💡 Focus on enterprise-specific metrics for your DPO model")
        
        print("\n📁 Check results in:")
        print("  - quick_test_enterprise/")
        print("  - direct_enterprise_test/")
        
    else:
        print("\n⚠️  All evaluations failed. This could be due to:")
        print("  - Model architecture compatibility")
        print("  - Missing dependencies")
        print("  - Memory constraints")
    
    return success_count, total_count

def test_model_loading():
    """Test if the model can be loaded at all"""
    print("🔍 Testing Model Loading...")
    
    try:
        # Simple model loading test
        test_script = '''
import sys
sys.path.append("/Users/ronel/Downloads/llm twin/evaluation")
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"
print(f"Loading model from: {model_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto", device_map="cpu")
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model).__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
'''
        
        result = subprocess.run(
            ["python", "-c", test_script],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    # First test model loading
    test_model_loading()
    
    print("\n" + "="*60)
    
    # Then test working evaluations
    test_working_evaluations()
