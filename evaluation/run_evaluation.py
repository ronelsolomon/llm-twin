"""
Main evaluation script that runs both MMLU and Chatbot Arena evaluations
"""

import argparse
import json
from pathlib import Path
from mmlu_evaluator import MMLUProEvaluator
from chatbot_arena_evaluator import ChatbotArenaEvaluator
from hellaswag_evaluator import HellaSwagEvaluator
from arc_c_evaluator import ARCCEvaluator
from winogrande_evaluator import WinograndeEvaluator
from piqa_evaluator import PIQAEvaluator
import pandas as pd
from datetime import datetime

def run_mmlu_evaluation(model_path: str, output_dir: str, num_samples: int = None):
    """Run MMLU-Pro evaluation"""
    print("="*50)
    print("Running MMLU-Pro Evaluation")
    print("="*50)
    
    evaluator = MMLUProEvaluator(model_path)
    dataset = evaluator.load_mmlu_pro_dataset()
    results = evaluator.evaluate_dataset(dataset, num_samples, output_dir)
    
    return results

def run_hellaswag_evaluation(model_path: str, output_dir: str, num_samples: int = None):
    """Run HellaSwag evaluation"""
    print("="*50)
    print("Running HellaSwag Evaluation")
    print("="*50)
    
    evaluator = HellaSwagEvaluator(model_path)
    dataset = evaluator.load_hellaswag_dataset()
    results = evaluator.evaluate_dataset(dataset, num_samples, output_dir)
    
    return results

def run_arc_c_evaluation(model_path: str, output_dir: str, num_samples: int = None):
    """Run ARC-C evaluation"""
    print("="*50)
    print("Running ARC-C Evaluation")
    print("="*50)
    
    evaluator = ARCCEvaluator(model_path)
    dataset = evaluator.load_arc_c_dataset()
    results = evaluator.evaluate_dataset(dataset, num_samples, output_dir)
    
    return results

def run_winogrande_evaluation(model_path: str, output_dir: str, num_samples: int = None):
    """Run Winogrande evaluation"""
    print("="*50)
    print("Running Winogrande Evaluation")
    print("="*50)
    
    evaluator = WinograndeEvaluator(model_path)
    dataset = evaluator.load_winogrande_dataset()
    results = evaluator.evaluate_dataset(dataset, num_samples, output_dir)
    
    return results

def run_piqa_evaluation(model_path: str, output_dir: str, num_samples: int = None):
    """Run PIQA evaluation"""
    print("="*50)
    print("Running PIQA Evaluation")
    print("="*50)
    
    evaluator = PIQAEvaluator(model_path)
    dataset = evaluator.load_piqa_dataset()
    results = evaluator.evaluate_dataset(dataset, num_samples, output_dir)
    
    return results

def run_arena_evaluation(models: dict, output_dir: str, rounds: int = 1):
    """Run Chatbot Arena evaluation"""
    print("="*50)
    print("Running Chatbot Arena Evaluation")
    print("="*50)
    
    evaluator = ChatbotArenaEvaluator(models)
    prompts = evaluator.load_test_prompts()
    results = evaluator.run_tournament(prompts, rounds, output_dir)
    
    return results

def generate_combined_report(mmlu_results: dict, hellaswag_results: dict, arena_results: list, output_dir: str):
    """Generate combined evaluation report"""
    report = {
        "evaluation_timestamp": datetime.now().isoformat(),
        "mmlu_pro_results": {
            "model": mmlu_results.get("model_path", "Unknown"),
            "overall_accuracy": mmlu_results.get("overall_accuracy", 0),
            "total_questions": mmlu_results.get("total_questions", 0),
            "correct_answers": mmlu_results.get("correct_answers", 0),
            "subject_accuracies": mmlu_results.get("subject_accuracies", {})
        },
        "hellaswag_results": {
            "model": hellaswag_results.get("model_path", "Unknown"),
            "overall_accuracy": hellaswag_results.get("overall_accuracy", 0),
            "total_questions": hellaswag_results.get("total_questions", 0),
            "correct_answers": hellaswag_results.get("correct_answers", 0),
            "domain_accuracies": hellaswag_results.get("domain_accuracies", {})
        },
        "chatbot_arena_results": {
            "model_rankings": {},
            "total_battles": len(arena_results) if isinstance(arena_results, list) else 0,
            "top_model": None
        }
    }
    
    # Handle arena results - could be list or dict
    if isinstance(arena_results, dict) and "model_ratings" in arena_results:
        # New format
        report["chatbot_arena_results"]["model_rankings"] = arena_results["model_ratings"]
        report["chatbot_arena_results"]["total_battles"] = len(arena_results.get("battle_results", []))
        
        if arena_results.get("model_ratings"):
            top_model = max(arena_results["model_ratings"].items(), 
                           key=lambda x: x[1]["elo_rating"])
            report["chatbot_arena_results"]["top_model"] = {
                "name": top_model[0],
                "elo_rating": top_model[1]["elo_rating"],
                "win_rate": top_model[1]["win_rate"]
            }
    elif isinstance(arena_results, list) and arena_results:
        # Old format - just battle results
        report["chatbot_arena_results"]["total_battles"] = len(arena_results)
        # Extract basic stats from battle results
        model_stats = {}
        for battle in arena_results:
            for model in [battle.model_a, battle.model_b]:
                if model and model not in model_stats:
                    model_stats[model] = {"battles": 0, "wins": 0}
                if model:
                    model_stats[model]["battles"] += 1
                    if battle.winner == "A" and model == battle.model_a:
                        model_stats[model]["wins"] += 1
                    elif battle.winner == "B" and model == battle.model_b:
                        model_stats[model]["wins"] += 1
        
        # Calculate simple rankings
        for model, stats in model_stats.items():
            win_rate = stats["wins"] / stats["battles"] if stats["battles"] > 0 else 0
            report["chatbot_arena_results"]["model_rankings"][model] = {
                "elo_rating": 1400 + (win_rate - 0.5) * 100,  # Simple elo approximation
                "win_rate": win_rate,
                "battles": stats["battles"]
            }
        
        if model_stats:
            top_model = max(model_stats.items(), key=lambda x: x[1]["wins"] / x[1]["battles"])
            top_stats = report["chatbot_arena_results"]["model_rankings"][top_model[0]]
            report["chatbot_arena_results"]["top_model"] = {
                "name": top_model[0],
                "elo_rating": top_stats["elo_rating"],
                "win_rate": top_stats["win_rate"]
            }
    
    # Save combined report
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    with open(output_path / "combined_evaluation_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Generate markdown report
    generate_markdown_report(report, output_path)
    
    return report

def generate_markdown_report(report: dict, output_path: Path):
    """Generate markdown report for easy reading"""
    mmlu = report["mmlu_pro_results"]
    hellaswag = report["hellaswag_results"]
    arena = report["chatbot_arena_results"]
    
    markdown = f"""# LLM Twin Evaluation Report

**Evaluation Date:** {report['evaluation_timestamp']}

## MMLU-Pro Results

### Overall Performance
- **Model:** {mmlu['model']}
- **Accuracy:** {mmlu['overall_accuracy']:.2%}
- **Questions:** {mmlu['correct_answers']}/{mmlu['total_questions']}

### Subject-wise Performance
| Subject | Accuracy |
|---------|----------|
"""
    
    for subject, accuracy in sorted(mmlu['subject_accuracies'].items(), 
                                   key=lambda x: x[1], reverse=True):
        markdown += f"| {subject} | {accuracy:.2%} |\n"
    
    markdown += f"""
## HellaSwag Results

### Overall Performance
- **Model:** {hellaswag['model']}
- **Accuracy:** {hellaswag['overall_accuracy']:.2%}
- **Questions:** {hellaswag['correct_answers']}/{hellaswag['total_questions']}

### Domain-wise Performance
| Domain | Accuracy |
|--------|----------|
"""
    
    for domain, accuracy in sorted(hellaswag['domain_accuracies'].items(), 
                                  key=lambda x: x[1], reverse=True):
        markdown += f"| {domain} | {accuracy:.2%} |\n"
    
    markdown += f"""
## Chatbot Arena Results

### Tournament Statistics
- **Total Battles:** {arena['total_battles']}
"""
    
    if arena['top_model']:
        top = arena['top_model']
        markdown += f"""
### Top Model
- **Model:** {top['name']}
- **Elo Rating:** {top['elo_rating']:.1f}
- **Win Rate:** {top['win_rate']:.2%}

### Model Rankings
| Rank | Model | Elo Rating | Win Rate | Battles |
|------|-------|------------|----------|---------|
"""
        
        rankings = sorted(arena['model_rankings'].items(), 
                         key=lambda x: x[1]['elo_rating'], reverse=True)
        
        for rank, (name, stats) in enumerate(rankings, 1):
            markdown += f"| {rank} | {name} | {stats['elo_rating']:.1f} | {stats['win_rate']:.2%} | {stats['battles']} |\n"
    
    markdown += f"""
## Summary

Your LLM twin achieved:
- **MMLU-Pro Accuracy:** {mmlu['overall_accuracy']:.2%} across {len(mmlu['subject_accuracies'])} subjects
- **HellaSwag Accuracy:** {hellaswag['overall_accuracy']:.2%} across {len(hellaswag['domain_accuracies'])} domains
"""
    
    if arena['top_model']:
        markdown += f"- **Arena Performance:** {arena['top_model']['name']} leads with {arena['top_model']['elo_rating']:.1f} Elo rating\n"
    
    # Performance analysis
    markdown += f"""
## Performance Analysis

### Knowledge & Reasoning
- **MMLU-Pro**: Tests academic knowledge across diverse subjects
- **HellaSwag**: Tests commonsense reasoning and situational understanding

### Practical Performance
- **Chatbot Arena**: Real-world conversational ability and response quality

### Overall Assessment
"""
    
    # Add performance assessment
    mmlu_score = mmlu['overall_accuracy']
    hellaswag_score = hellaswag['overall_accuracy']
    
    if mmlu_score > 0.5:
        markdown += f"✅ Strong academic performance ({mmlu_score:.1%} MMLU accuracy)\n"
    else:
        markdown += f"⚠️  Room for improvement in academic knowledge ({mmlu_score:.1%} MMLU accuracy)\n"
    
    if hellaswag_score > 0.7:
        markdown += f"✅ Excellent commonsense reasoning ({hellaswag_score:.1%} HellaSwag accuracy)\n"
    elif hellaswag_score > 0.5:
        markdown += f"📈 Good reasoning ability ({hellaswag_score:.1%} HellaSwag accuracy)\n"
    else:
        markdown += f"⚠️  Reasoning skills need development ({hellaswag_score:.1%} HellaSwag accuracy)\n"
    
    # Save markdown report
    with open(output_path / "evaluation_report.md", 'w') as f:
        f.write(markdown)

def main():
    parser = argparse.ArgumentParser(description="Run comprehensive LLM evaluation")
    parser.add_argument("--mmlu-model", type=str, help="Model path for MMLU evaluation")
    parser.add_argument("--arena-models", nargs="+", help="Model paths for arena evaluation")
    parser.add_argument("--arena-names", nargs="+", help="Model names for arena evaluation")
    parser.add_argument("--mmlu-samples", type=int, help="Number of MMLU samples (default: all)")
    parser.add_argument("--hellaswag-samples", type=int, help="Number of HellaSwag samples (default: all)")
    parser.add_argument("--arc-c-samples", type=int, help="Number of ARC-C samples (default: all)")
    parser.add_argument("--winogrande-samples", type=int, help="Number of Winogrande samples (default: all)")
    parser.add_argument("--piqa-samples", type=int, help="Number of PIQA samples (default: all)")
    parser.add_argument("--arena-rounds", type=int, default=1, help="Arena tournament rounds")
    parser.add_argument("--output", type=str, default="evaluation_results", help="Output directory")
    parser.add_argument("--eval-type", choices=["mmlu", "hellaswag", "arc-c", "winogrande", "piqa", "arena", "reasoning", "all"], default="all", 
                       help="Type of evaluation to run")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    mmlu_results = None
    hellaswag_results = None
    arc_c_results = None
    winogrande_results = None
    piqa_results = None
    arena_results = None
    
    # Run MMLU evaluation
    if args.eval_type in ["mmlu", "all"]:
        if not args.mmlu_model:
            raise ValueError("--mmlu-model required for MMLU evaluation")
        
        mmlu_results = run_mmlu_evaluation(
            args.mmlu_model, 
            str(output_dir / "mmlu"), 
            args.mmlu_samples
        )
    
    # Run HellaSwag evaluation
    if args.eval_type in ["hellaswag", "reasoning", "all"]:
        if not args.mmlu_model:
            raise ValueError("--mmlu-model required for HellaSwag evaluation")
        
        hellaswag_results = run_hellaswag_evaluation(
            args.mmlu_model, 
            str(output_dir / "hellaswag"), 
            args.hellaswag_samples
        )
    
    # Run ARC-C evaluation
    if args.eval_type in ["arc-c", "reasoning", "all"]:
        if not args.mmlu_model:
            raise ValueError("--mmlu-model required for ARC-C evaluation")
        
        arc_c_results = run_arc_c_evaluation(
            args.mmlu_model, 
            str(output_dir / "arc_c"), 
            args.arc_c_samples
        )
    
    # Run Winogrande evaluation
    if args.eval_type in ["winogrande", "reasoning", "all"]:
        if not args.mmlu_model:
            raise ValueError("--mmlu-model required for Winogrande evaluation")
        
        winogrande_results = run_winogrande_evaluation(
            args.mmlu_model, 
            str(output_dir / "winogrande"), 
            args.winogrande_samples
        )
    
    # Run PIQA evaluation
    if args.eval_type in ["piqa", "reasoning", "all"]:
        if not args.mmlu_model:
            raise ValueError("--mmlu-model required for PIQA evaluation")
        
        piqa_results = run_piqa_evaluation(
            args.mmlu_model, 
            str(output_dir / "piqa"), 
            args.piqa_samples
        )
    
    # Run Arena evaluation
    if args.eval_type in ["arena", "all"]:
        if not args.arena_models:
            raise ValueError("--arena-models required for arena evaluation")
        
        # Create model mapping
        if args.arena_names:
            if len(args.arena_names) != len(args.arena_models):
                raise ValueError("Number of arena names must match number of models")
            models = dict(zip(args.arena_names, args.arena_models))
        else:
            models = {f"model_{i}": path for i, path in enumerate(args.arena_models)}
        
        arena_results = run_arena_evaluation(
            models, 
            str(output_dir / "arena"), 
            args.arena_rounds
        )
    
    # Generate combined report
    if mmlu_results and hellaswag_results and arena_results:
        generate_combined_report(mmlu_results, hellaswag_results, arena_results, str(output_dir))
        print(f"\nCombined evaluation report saved to {output_dir}")
    elif mmlu_results and hellaswag_results:
        # Generate report without arena
        print(f"\nMMLU and HellaSwag evaluation results saved to {output_dir}")
    elif mmlu_results:
        print(f"\nMMLU evaluation results saved to {output_dir / 'mmlu'}")
    elif hellaswag_results:
        print(f"\nHellaSwag evaluation results saved to {output_dir / 'hellaswag'}")
    elif arena_results:
        print(f"\nArena evaluation results saved to {output_dir / 'arena'}")

if __name__ == "__main__":
    main()
