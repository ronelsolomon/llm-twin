"""
Integration script to test enhanced retrieval with the LLM Twin evaluation pipeline
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.hybrid_vector_store import HybridVectorStore, HybridStoreConfig
from src.ingestion.embedder import EmbeddingGenerator
from evaluation.retrieval_evaluator import RetrievalEvaluator, RetrievalEvaluationConfig


def setup_enhanced_retrieval_system():
    """Initialize the enhanced retrieval system"""
    print("Setting up enhanced retrieval system...")
    
    # Configure hybrid vector store
    vector_config = HybridStoreConfig(
        qdrant_host="localhost",
        qdrant_port=6333,
        embedding_dimension=384,
        distance_metric="cosine"
    )
    
    # Initialize vector store
    vector_store = HybridVectorStore(vector_config)
    
    # Initialize embedder
    embedder = EmbeddingGenerator(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_type="sentence_transformers"
    )
    
    print("✓ Enhanced retrieval system initialized")
    return vector_store, embedder


def run_retrieval_evaluation(vector_store, embedder, output_dir: str = "evaluation_results"):
    """Run the retrieval evaluation"""
    print("\nRunning retrieval evaluation...")
    
    # Configure evaluation
    eval_config = RetrievalEvaluationConfig(
        output_dir=output_dir
    )
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(vector_store, embedder, eval_config)
    
    # Run evaluation
    results = evaluator.run_evaluation()
    
    print(f"\n✓ Evaluation complete! Results saved to {output_dir}")
    return results


def integrate_with_model_comparison(model_configs: List[Dict], output_dir: str):
    """Integrate enhanced retrieval into model comparison evaluation"""
    print("\nIntegrating enhanced retrieval with model comparison...")
    
    # Setup retrieval system
    vector_store, embedder = setup_enhanced_retrieval_system()
    
    # Import model comparison evaluator
    from evaluation.model_comparison_evaluator import ModelComparisonEvaluator
    
    # Create enhanced model comparison evaluator
    class EnhancedModelComparisonEvaluator(ModelComparisonEvaluator):
        """Enhanced model comparison with retrieval augmentation"""
        
        def __init__(self, model_configs, device="auto", use_speculative=True):
            super().__init__(model_configs, device, use_speculative)
            
            # Initialize enhanced retrieval
            self.enhanced_pipeline = None
            try:
                from src.ingestion.enhanced_retrieval import EnhancedRetrievalPipeline, RetrievalConfig
                retrieval_config = RetrievalConfig(
                    expansion_strategies=["synonym", "paraphrase", "domain_specific"],
                    max_expanded_queries=3,  # Reduced for faster evaluation
                    top_k_per_query=5,
                    final_top_k=3
                )
                self.enhanced_pipeline = EnhancedRetrievalPipeline(
                    vector_store, embedder, retrieval_config
                )
                print("✓ Enhanced retrieval pipeline integrated")
            except Exception as e:
                print(f"⚠ Enhanced retrieval integration failed: {e}")
        
        def generate_response_with_retrieval(self, model_name: str, instruction: str, max_tokens: int = 512) -> str:
            """Generate response with retrieval augmentation"""
            if not self.enhanced_pipeline:
                return self.generate_response(model_name, instruction, max_tokens)
            
            try:
                # Retrieve relevant context
                retrieval_result = self.enhanced_pipeline.retrieve(instruction)
                context_docs = retrieval_result.get("final_results", [])
                
                if context_docs:
                    # Format context
                    context_text = "\n\n".join([
                        f"[{doc.get('document_type', 'document')}]: {doc.get('text', '')}"
                        for doc in context_docs[:3]  # Use top 3 results
                    ])
                    
                    # Augment instruction with context
                    augmented_instruction = f"""Context:
{context_text}

Based on the above context, please respond to the following:
{instruction}"""
                    
                    print(f"  Retrieved {len(context_docs)} context documents")
                    return self.generate_response(model_name, augmented_instruction, max_tokens)
                else:
                    return self.generate_response(model_name, instruction, max_tokens)
                    
            except Exception as e:
                print(f"  Retrieval augmentation failed: {e}, using original instruction")
                return self.generate_response(model_name, instruction, max_tokens)
        
        def compare_models_on_task_with_retrieval(self, task: Dict) -> Dict:
            """Compare models on a task with retrieval augmentation"""
            results = {}
            
            for model_name in self.models.keys():
                response = self.generate_response_with_retrieval(
                    model_name, task["instruction"]
                )
                quality_scores = self.evaluate_response_quality(
                    response, task["task_type"], task["criteria"]
                )
                
                results[model_name] = {
                    "response": response,
                    "quality_scores": quality_scores,
                    "overall_score": quality_scores.get("overall", 0.0),
                    "used_retrieval": self.enhanced_pipeline is not None
                }
            
            return {
                "task": task,
                "model_results": results,
                "winner": max(results.keys(), key=lambda k: results[k]["overall_score"]),
                "ranking": sorted(results.keys(), key=lambda k: results[k]["overall_score"], reverse=True)
            }
    
    # Run enhanced model comparison
    enhanced_evaluator = EnhancedModelComparisonEvaluator(model_configs)
    dataset = enhanced_evaluator.load_comparison_dataset()
    
    print(f"Running enhanced model comparison on {len(dataset)} tasks...")
    results = enhanced_evaluator.evaluate_dataset(dataset, f"{output_dir}/enhanced_model_comparison")
    
    # Compare with baseline
    baseline_evaluator = ModelComparisonEvaluator(model_configs)
    baseline_results = baseline_evaluator.evaluate_dataset(dataset, f"{output_dir}/baseline_model_comparison")
    
    # Generate comparison report
    comparison_report = {
        "enhanced_results": results,
        "baseline_results": baseline_results,
        "improvement_analysis": analyze_model_comparison_improvements(baseline_results, results)
    }
    
    # Save comparison report
    with open(f"{output_dir}/retrieval_augmentation_comparison.json", 'w') as f:
        json.dump(comparison_report, f, indent=2, default=str)
    
    print(f"✓ Enhanced model comparison complete! Results saved to {output_dir}")
    return comparison_report


def analyze_model_comparison_improvements(baseline_results: Dict, enhanced_results: Dict) -> Dict:
    """Analyze improvements from retrieval augmentation"""
    analysis = {
        "overall_score_improvements": {},
        "task_type_improvements": {},
        "retrieval_utilization": {}
    }
    
    # Overall score improvements
    baseline_scores = baseline_results.get("overall_model_scores", {})
    enhanced_scores = enhanced_results.get("overall_model_scores", {})
    
    for model_name in baseline_scores:
        if model_name in enhanced_scores:
            baseline_score = baseline_scores[model_name]
            enhanced_score = enhanced_scores[model_name]
            improvement = (enhanced_score - baseline_score) / baseline_score if baseline_score > 0 else 0
            
            analysis["overall_score_improvements"][model_name] = {
                "baseline": baseline_score,
                "enhanced": enhanced_score,
                "improvement_percent": improvement * 100
            }
    
    # Win rate improvements
    baseline_win_rates = baseline_results.get("win_rates", {})
    enhanced_win_rates = enhanced_results.get("win_rates", {})
    
    for model_name in baseline_win_rates:
        if model_name in enhanced_win_rates:
            baseline_wr = baseline_win_rates[model_name]
            enhanced_wr = enhanced_win_rates[model_name]
            wr_improvement = (enhanced_wr - baseline_wr) / baseline_wr if baseline_wr > 0 else 0
            
            analysis["retrieval_utilization"][model_name] = {
                "baseline_win_rate": baseline_wr,
                "enhanced_win_rate": enhanced_wr,
                "win_rate_improvement_percent": wr_improvement * 100
            }
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="Test enhanced retrieval with LLM Twin")
    parser.add_argument("--mode", choices=["retrieval", "model_comparison", "both"], 
                       default="both", help="Evaluation mode")
    parser.add_argument("--models", nargs="+", help="Model paths for comparison (name:path pairs)")
    parser.add_argument("--output", type=str, default="evaluation_results/enhanced_retrieval", 
                       help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    print("Enhanced Retrieval Testing for LLM Twin")
    print("=" * 50)
    
    # Setup retrieval system
    vector_store, embedder = setup_enhanced_retrieval_system()
    
    if args.mode in ["retrieval", "both"]:
        # Run retrieval evaluation
        retrieval_results = run_retrieval_evaluation(vector_store, embedder, args.output)
        
        print("\n🔍 RETRIEVAL EVALUATION SUMMARY:")
        print(f"NDCG@5 improvement: {retrieval_results['average_improvements'].get('ndcg_at_5', 0)*100:+.1f}%")
        print(f"Relevance score improvement: {retrieval_results['average_improvements'].get('relevance_score', 0)*100:+.1f}%")
    
    if args.mode in ["model_comparison", "both"]:
        if not args.models:
            print("⚠ No models specified for comparison. Using example models...")
            # Use example model paths if none provided
            model_configs = [
                {"name": "baseline_llama", "path": "dpo_llm_twin_merged"},
                {"name": "enhanced_llama", "path": "dpo_llm_twin_improved_merged"}
            ]
        else:
            # Parse model configurations
            model_configs = []
            for model_config in args.models:
                if ":" in model_config:
                    name, path = model_config.split(":", 1)
                else:
                    name = model_config.split("/")[-1]
                    path = model_config
                model_configs.append({"name": name, "path": path})
        
        # Run enhanced model comparison
        comparison_results = integrate_with_model_comparison(model_configs, args.output)
        
        print("\n🤖 MODEL COMPARISON SUMMARY:")
        improvements = comparison_results["improvement_analysis"]
        
        for model_name, metrics in improvements["overall_score_improvements"].items():
            print(f"{model_name}: {metrics['improvement_percent']:+.1f}% overall score improvement")
        
        for model_name, metrics in improvements["retrieval_utilization"].items():
            print(f"{model_name}: {metrics['win_rate_improvement_percent']:+.1f}% win rate improvement")
    
    print(f"\n✓ All evaluations complete! Results saved to {args.output}")
    print("\nNext steps:")
    print("1. Review the detailed results in the output directory")
    print("2. Check retrieval_evaluation_results.json for retrieval metrics")
    print("3. Check retrieval_augmentation_comparison.json for model comparison")
    print("4. Use the insights to fine-tune the retrieval parameters")


if __name__ == "__main__":
    main()
