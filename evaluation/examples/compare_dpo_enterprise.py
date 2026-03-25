#!/usr/bin/env python3
"""
Example script to compare DPO LLM Twin models with base models using Enterprise Scenarios Leaderboard
This script evaluates:
1. Base model (e.g., Llama-7B)
2. DPO fine-tuned model
3. DPO LLM Twin (your personalized model)
"""

import sys
import os
from pathlib import Path

# Add the evaluation directory to the path
sys.path.append(str(Path(__file__).parent.parent))

from enterprise_scenarios_evaluator import EnterpriseScenariosEvaluator

def main():
    # Define model configurations - update these paths to match your actual model locations
    model_configs = [
        {
            "name": "Base_Llama-7B",
            "path": "/path/to/base/llama-7b"  # Update with actual base model path
        },
        {
            "name": "DPO_Fine_Tuned",
            "path": "/Users/ronel/Downloads/llm twin/dpo_llm_twin_merged"  # Your DPO fine-tuned model
        },
        {
            "name": "DPO_LLM_Twin",
            "path": "/Users/ronel/Downloads/llm twin/dpo_llm_twin_improved_merged"  # Your improved DPO LLM Twin
        }
    ]
    
    # Check if model paths exist
    for config in model_configs:
        model_path = Path(config["path"])
        if not model_path.exists():
            print(f"⚠️  Warning: Model path does not exist: {model_path}")
            print(f"   Please update the path for {config['name']} in the script")
            return
    
    print("🚀 Starting Enterprise Scenarios Leaderboard Evaluation")
    print("="*60)
    print("Models being compared:")
    for config in model_configs:
        print(f"  • {config['name']}: {config['path']}")
    print("="*60)
    
    # Initialize evaluator
    evaluator = EnterpriseScenariosEvaluator(model_configs, device="auto")
    
    # Run evaluation
    output_dir = "enterprise_comparison_results"
    results = evaluator.evaluate_all_scenarios(output_dir)
    
    # Print summary comparison
    print("\n" + "="*80)
    print("DPO LLM TWIN VS BASE MODEL COMPARISON")
    print("="*80)
    
    aggregate = results["aggregate_results"]
    
    print("\n📊 OVERALL ENTERPRISE PERFORMANCE:")
    sorted_models = sorted(aggregate["overall_model_scores"].items(), key=lambda x: x[1], reverse=True)
    
    for rank, (model_name, score) in enumerate(sorted_models, 1):
        emoji = "🏆" if rank == 1 else "🥈" if rank == 2 else "🥉"
        improvement = ""
        if "DPO" in model_name and "Base" in sorted_models[0][0]:
            base_score = aggregate["overall_model_scores"].get("Base_Llama-7B", 0)
            if base_score > 0:
                improvement_pct = ((score - base_score) / base_score) * 100
                improvement = f" (+{improvement_pct:.1f}% vs base)"
        
        print(f"  {emoji} {rank}. {model_name}: {score:.3f}{improvement}")
    
    print("\n📈 SCENARIO-BY-SCENARIO ANALYSIS:")
    scenarios = ["financebench", "legal_confidentiality", "writing_prompts", 
                 "customer_support", "toxic_prompts", "enterprise_pii"]
    
    for scenario in scenarios:
        if scenario in aggregate["scenario_averages"][sorted_models[0][0]]:
            print(f"\n  {scenario.upper().replace('_', ' ')}:")
            
            scenario_scores = {}
            for model_name in aggregate["scenario_averages"].keys():
                if scenario in aggregate["scenario_averages"][model_name]:
                    scenario_scores[model_name] = aggregate["scenario_averages"][model_name][scenario]
            
            sorted_scenario = sorted(scenario_scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (model_name, score) in enumerate(sorted_scenario, 1):
                emoji = "🏆" if rank == 1 else "  "
                improvement = ""
                if "DPO" in model_name and "Base_Llama-7B" in scenario_scores:
                    base_score = scenario_scores.get("Base_Llama-7B", 0)
                    if base_score > 0:
                        improvement_pct = ((score - base_score) / base_score) * 100
                        improvement = f" (+{improvement_pct:.1f}% vs base)"
                
                print(f"    {emoji} {rank}. {model_name}: {score:.3f}{improvement}")
    
    print("\n🎯 ENTERPRISE READINESS ASSESSMENT:")
    for model_name, overall_score in aggregate["overall_model_scores"].items():
        if overall_score >= 0.8:
            readiness = "🟢 ENTERPRISE READY"
            description = "Suitable for production enterprise use"
        elif overall_score >= 0.6:
            readiness = "🟡 NEEDS IMPROVEMENT"
            description = "Requires additional fine-tuning for enterprise deployment"
        else:
            readiness = "🔴 NOT READY"
            description = "Significant improvements needed before enterprise use"
        
        print(f"  {model_name}: {readiness}")
        print(f"    {description} ({overall_score:.3f})")
    
    print("\n🔍 KEY INSIGHTS:")
    
    # Compare DPO vs Base
    dpo_models = [name for name in aggregate["overall_model_scores"].keys() if "DPO" in name]
    base_models = [name for name in aggregate["overall_model_scores"].keys() if "Base" in name]
    
    if dpo_models and base_models:
        best_dpo = max(dpo_models, key=lambda x: aggregate["overall_model_scores"][x])
        best_base = max(base_models, key=lambda x: aggregate["overall_model_scores"][x])
        
        dpo_score = aggregate["overall_model_scores"][best_dpo]
        base_score = aggregate["overall_model_scores"][best_base]
        
        if dpo_score > base_score:
            improvement = ((dpo_score - base_score) / base_score) * 100
            print(f"  ✅ DPO fine-tuning shows {improvement_pct:.1f}% improvement over base model")
            print(f"  🎯 Best DPO model: {best_dpo}")
        else:
            print(f"  ⚠️  DPO fine-tuning shows {((base_score - dpo_score) / base_score) * 100:.1f}% decrease vs base")
    
    # Find strongest scenarios for DPO models
    if dpo_models:
        print(f"\n  📊 DPO MODEL STRENGTHS:")
        for scenario in scenarios:
            if scenario in aggregate["scenario_averages"][dpo_models[0]]:
                dpo_avg = aggregate["scenario_averages"][dpo_models[0]][scenario]
                if base_models and scenario in aggregate["scenario_averages"][base_models[0]]:
                    base_avg = aggregate["scenario_averages"][base_models[0]][scenario]
                    if dpo_avg > base_avg:
                        improvement = ((dpo_avg - base_avg) / base_avg) * 100
                        print(f"    • {scenario.replace('_', ' ').title()}: +{improvement:.1f}% improvement")
    
    print("\n" + "="*80)
    print("📁 Detailed results saved to:")
    print(f"   {output_dir}/enterprise_scenarios_results.json")
    print("="*80)
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS:")
    
    best_model = sorted_models[0][0]
    best_score = sorted_models[0][1]
    
    if best_score >= 0.8:
        print(f"  🎉 {best_model} is ready for enterprise deployment!")
        print(f"  📈 Consider using this model for production workloads")
    elif best_score >= 0.6:
        print(f"  🔧 {best_model} shows promise but needs refinement")
        print(f"  💪 Focus on improving weaker scenarios before deployment")
    else:
        print(f"  🚧 {best_model} requires significant improvements")
        print(f"  📚 Consider additional fine-tuning on enterprise-specific data")
    
    if "DPO_LLM_Twin" in aggregate["overall_model_scores"]:
        twin_score = aggregate["overall_model_scores"]["DPO_LLM_Twin"]
        if "DPO_Fine_Tuned" in aggregate["overall_model_scores"]:
            dpo_score = aggregate["overall_model_scores"]["DPO_Fine_Tuned"]
            if twin_score > dpo_score:
                print(f"  ✨ Your personalized LLM Twin outperforms standard DPO fine-tuning!")
            else:
                print(f"  📊 Standard DPO fine-tuning performs better than personalized twin")

if __name__ == "__main__":
    main()
