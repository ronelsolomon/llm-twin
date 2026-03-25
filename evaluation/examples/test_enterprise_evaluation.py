#!/usr/bin/env python3
"""
Quick test script to verify Enterprise Scenarios evaluation works with your DPO models
"""

import sys
import os
from pathlib import Path

# Add the evaluation directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from enterprise_scenarios_evaluator import EnterpriseScenariosEvaluator

def test_enterprise_evaluation():
    """Test with your actual DPO models"""
    
    print("🧪 Testing Enterprise Scenarios Evaluation with DPO Models")
    print("="*60)
    
    # Use your actual DPO models
    model_configs = [
        {
            "name": "DPO_Fine_Tuned",
            "path": "/Users/ronel/Downloads/llm twin/dpo_llm_twin_merged"
        },
        {
            "name": "DPO_LLM_Twin_Improved", 
            "path": "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"
        }
    ]
    
    # Verify model paths exist
    for config in model_configs:
        model_path = Path(config["path"])
        if not model_path.exists():
            print(f"❌ Model path not found: {model_path}")
            return False
        print(f"✅ Model found: {config['name']} at {model_path}")
    
    try:
        # Initialize evaluator
        print("\n🚀 Initializing Enterprise Scenarios Evaluator...")
        evaluator = EnterpriseScenariosEvaluator(model_configs, device="auto")
        
        # Test with minimal dataset (1 item per scenario for quick test)
        print("\n📊 Running quick test evaluation (1 item per scenario)...")
        
        # Override the dataset loading to use smaller test sets
        original_load = evaluator.load_enterprise_datasets
        def load_test_datasets():
            datasets = original_load()
            # Reduce to 1 item per scenario for testing
            for scenario_name in datasets:
                datasets[scenario_name] = datasets[scenario_name][:1]
            return datasets
        
        evaluator.load_enterprise_datasets = load_test_datasets
        
        # Run evaluation
        output_dir = "test_enterprise_results"
        results = evaluator.evaluate_all_scenarios(output_dir)
        
        print("\n✅ Test evaluation completed successfully!")
        
        # Show quick results
        aggregate = results["aggregate_results"]
        print("\n📈 Quick Results:")
        for model_name, score in aggregate["overall_model_scores"].items():
            print(f"  {model_name}: {score:.3f}")
        
        print(f"\n📁 Results saved to: {output_dir}/enterprise_scenarios_results.json")
        return True
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enterprise_evaluation()
    if success:
        print("\n🎉 Enterprise Scenarios evaluation is ready to use!")
        print("\nNext steps:")
        print("1. Run the full comparison: python compare_dpo_enterprise.py")
        print("2. Or use the main script: python run_evaluation.py --eval-type enterprise --enterprise-models ...")
    else:
        print("\n❌ Please fix the errors before running the full evaluation")
