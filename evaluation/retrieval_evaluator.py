"""
Enhanced Evaluation for Retrieval-Augmented LLM Twin
Evaluates the impact of query expansion and reranking on model performance
"""

import json
import time
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from loguru import logger

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.ingestion.enhanced_retrieval import EnhancedRetrievalPipeline, RetrievalConfig
from src.ingestion.hybrid_vector_store import HybridVectorStore, HybridStoreConfig
from src.ingestion.embedder import EmbeddingGenerator


@dataclass
class RetrievalEvaluationConfig:
    """Configuration for retrieval evaluation"""
    test_queries_path: str = "data/evaluation/test_queries.json"
    baseline_results_path: str = "data/evaluation/baseline_results.json"
    enhanced_results_path: str = "data/evaluation/enhanced_results.json"
    output_dir: str = "evaluation_results/retrieval_comparison"
    
    # Evaluation metrics
    metrics_to_compute: List[str] = None
    
    def __post_init__(self):
        if self.metrics_to_compute is None:
            self.metrics_to_compute = [
                "precision_at_k", "recall_at_k", "ndcg_at_k", 
                "relevance_score", "diversity_score", "coverage_score"
            ]


class RetrievalEvaluator:
    """Evaluates retrieval performance with and without enhancements"""
    
    def __init__(self, 
                 vector_store: HybridVectorStore,
                 embedder: EmbeddingGenerator,
                 config: RetrievalEvaluationConfig = None):
        self.vector_store = vector_store
        self.embedder = embedder
        self.config = config or RetrievalEvaluationConfig()
        
        # Initialize enhanced pipeline
        enhanced_config = RetrievalConfig(
            expansion_strategies=["synonym", "paraphrase", "domain_specific"],
            max_expanded_queries=5,
            top_k_per_query=10,
            final_top_k=5,
            reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.enhanced_pipeline = EnhancedRetrievalPipeline(
            vector_store, embedder, enhanced_config
        )
        
        logger.info("Retrieval evaluator initialized")
    
    def load_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries with ground truth relevance judgments"""
        try:
            with open(self.config.test_queries_path, 'r') as f:
                queries = json.load(f)
            logger.info(f"Loaded {len(queries)} test queries")
            return queries
        except FileNotFoundError:
            # Create sample test queries if file doesn't exist
            return self._create_sample_queries()
    
    def _create_sample_queries(self) -> List[Dict[str, Any]]:
        """Create sample test queries for evaluation"""
        sample_queries = [
            {
                "query": "How to implement machine learning algorithms in Python",
                "expected_document_types": ["article", "repository"],
                "expected_topics": ["machine learning", "python", "programming"],
                "ground_truth_relevance": [1, 0.8, 0.6, 0.4, 0.2]  # Relevance scores for top 5 results
            },
            {
                "query": "Benefits of renewable energy systems",
                "expected_document_types": ["article"],
                "expected_topics": ["renewable energy", "sustainability", "environment"],
                "ground_truth_relevance": [1, 0.9, 0.7, 0.5, 0.3]
            },
            {
                "query": "Code optimization techniques for better performance",
                "expected_document_types": ["repository", "article"],
                "expected_topics": ["optimization", "performance", "programming"],
                "ground_truth_relevance": [1, 0.8, 0.6, 0.4, 0.2]
            },
            {
                "query": "Latest developments in artificial intelligence research",
                "expected_document_types": ["article"],
                "expected_topics": ["AI", "research", "technology"],
                "ground_truth_relevance": [1, 0.9, 0.8, 0.6, 0.4]
            },
            {
                "query": "Best practices for software development documentation",
                "expected_document_types": ["article", "repository"],
                "expected_topics": ["documentation", "software development", "best practices"],
                "ground_truth_relevance": [1, 0.8, 0.7, 0.5, 0.3]
            }
        ]
        
        # Save sample queries for future use
        Path(self.config.test_queries_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.config.test_queries_path, 'w') as f:
            json.dump(sample_queries, f, indent=2)
        
        logger.info(f"Created {len(sample_queries)} sample test queries")
        return sample_queries
    
    def run_baseline_retrieval(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Run baseline retrieval without enhancements"""
        # Generate embedding for query
        query_embedding = self.embedder.generate_embeddings([query])[0]
        
        # Simple vector search
        results = self.vector_store.search_similar(
            query_vector=query_embedding,
            limit=top_k
        )
        
        return results
    
    def run_enhanced_retrieval(self, query: str, domain_context: Optional[Dict] = None) -> Dict[str, Any]:
        """Run enhanced retrieval with query expansion and reranking"""
        return self.enhanced_pipeline.retrieve(query, domain_context)
    
    def calculate_precision_at_k(self, results: List[Dict[str, Any]], ground_truth: List[float], k: int) -> float:
        """Calculate Precision@K"""
        if not results or not ground_truth:
            return 0.0
        
        # Consider results relevant if they have a relevance score >= 0.5
        relevant_results = sum(1 for i in range(min(k, len(results))) 
                             if i < len(ground_truth) and ground_truth[i] >= 0.5)
        
        return relevant_results / min(k, len(results))
    
    def calculate_recall_at_k(self, results: List[Dict[str, Any]], ground_truth: List[float], k: int) -> float:
        """Calculate Recall@K"""
        if not results or not ground_truth:
            return 0.0
        
        # Total relevant documents in ground truth
        total_relevant = sum(1 for score in ground_truth if score >= 0.5)
        if total_relevant == 0:
            return 0.0
        
        # Relevant documents found in top K
        relevant_found = sum(1 for i in range(min(k, len(results))) 
                           if i < len(ground_truth) and ground_truth[i] >= 0.5)
        
        return relevant_found / total_relevant
    
    def calculate_ndcg_at_k(self, results: List[Dict[str, Any]], ground_truth: List[float], k: int) -> float:
        """Calculate Normalized Discounted Cumulative Gain@K"""
        if not results or not ground_truth:
            return 0.0
        
        def dcg_at_k(relevance_scores: List[float], k: int) -> float:
            """Calculate DCG@K"""
            dcg = 0.0
            for i in range(min(k, len(relevance_scores))):
                if i == 0:
                    dcg += relevance_scores[i]
                else:
                    dcg += relevance_scores[i] / np.log2(i + 1)
            return dcg
        
        # Actual DCG
        actual_relevance = ground_truth[:min(k, len(results))]
        actual_dcg = dcg_at_k(actual_relevance, k)
        
        # Ideal DCG (sorted by relevance)
        ideal_relevance = sorted(ground_truth, reverse=True)[:k]
        ideal_dcg = dcg_at_k(ideal_relevance, k)
        
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0
    
    def calculate_relevance_score(self, results: List[Dict[str, Any]], ground_truth: List[float]) -> float:
        """Calculate average relevance score"""
        if not results or not ground_truth:
            return 0.0
        
        actual_relevance = ground_truth[:len(results)]
        return np.mean(actual_relevance) if actual_relevance else 0.0
    
    def calculate_diversity_score(self, results: List[Dict[str, Any]]) -> float:
        """Calculate diversity of document types in results"""
        if not results:
            return 0.0
        
        doc_types = set()
        for result in results:
            doc_type = result.get("document_type", "unknown")
            doc_types.add(doc_type)
        
        # Normalize by expected diversity (assuming 3 types is ideal)
        return min(len(doc_types) / 3, 1.0)
    
    def calculate_coverage_score(self, results: List[Dict[str, Any]], expected_topics: List[str]) -> float:
        """Calculate topic coverage in results"""
        if not results or not expected_topics:
            return 0.0
        
        covered_topics = set()
        for result in results:
            text = result.get("text", "").lower()
            for topic in expected_topics:
                if topic.lower() in text:
                    covered_topics.add(topic)
        
        return len(covered_topics) / len(expected_topics)
    
    def evaluate_single_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single query with both baseline and enhanced retrieval"""
        query = query_data["query"]
        expected_topics = query_data.get("expected_topics", [])
        ground_truth = query_data.get("ground_truth_relevance", [])
        
        logger.info(f"Evaluating query: '{query}'")
        
        # Run baseline retrieval
        start_time = time.time()
        baseline_results = self.run_baseline_retrieval(query)
        baseline_time = time.time() - start_time
        
        # Run enhanced retrieval
        start_time = time.time()
        enhanced_result = self.run_enhanced_retrieval(query)
        enhanced_results = enhanced_result.get("final_results", [])
        enhanced_time = time.time() - start_time
        
        # Calculate metrics for baseline
        baseline_metrics = {}
        for k in [1, 3, 5]:
            if k <= len(baseline_results):
                baseline_metrics[f"precision_at_{k}"] = self.calculate_precision_at_k(
                    baseline_results, ground_truth, k
                )
                baseline_metrics[f"recall_at_{k}"] = self.calculate_recall_at_k(
                    baseline_results, ground_truth, k
                )
                baseline_metrics[f"ndcg_at_{k}"] = self.calculate_ndcg_at_k(
                    baseline_results, ground_truth, k
                )
        
        baseline_metrics["relevance_score"] = self.calculate_relevance_score(
            baseline_results, ground_truth
        )
        baseline_metrics["diversity_score"] = self.calculate_diversity_score(baseline_results)
        baseline_metrics["coverage_score"] = self.calculate_coverage_score(
            baseline_results, expected_topics
        )
        baseline_metrics["retrieval_time"] = baseline_time
        
        # Calculate metrics for enhanced
        enhanced_metrics = {}
        for k in [1, 3, 5]:
            if k <= len(enhanced_results):
                enhanced_metrics[f"precision_at_{k}"] = self.calculate_precision_at_k(
                    enhanced_results, ground_truth, k
                )
                enhanced_metrics[f"recall_at_{k}"] = self.calculate_recall_at_k(
                    enhanced_results, ground_truth, k
                )
                enhanced_metrics[f"ndcg_at_{k}"] = self.calculate_ndcg_at_k(
                    enhanced_results, ground_truth, k
                )
        
        enhanced_metrics["relevance_score"] = self.calculate_relevance_score(
            enhanced_results, ground_truth
        )
        enhanced_metrics["diversity_score"] = self.calculate_diversity_score(enhanced_results)
        enhanced_metrics["coverage_score"] = self.calculate_coverage_score(
            enhanced_results, expected_topics
        )
        enhanced_metrics["retrieval_time"] = enhanced_time
        
        # Calculate improvements
        improvements = {}
        for metric in baseline_metrics:
            if metric in enhanced_metrics:
                baseline_val = baseline_metrics[metric]
                enhanced_val = enhanced_metrics[metric]
                if baseline_val > 0:
                    improvements[metric] = (enhanced_val - baseline_val) / baseline_val
                else:
                    improvements[metric] = enhanced_val
        
        return {
            "query": query,
            "baseline_results": baseline_results,
            "enhanced_results": enhanced_results,
            "baseline_metrics": baseline_metrics,
            "enhanced_metrics": enhanced_metrics,
            "improvements": improvements,
            "enhancement_metadata": enhanced_result.get("retrieval_metadata", {})
        }
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation on all test queries"""
        test_queries = self.load_test_queries()
        
        logger.info(f"Starting evaluation with {len(test_queries)} queries")
        
        all_results = []
        baseline_metrics_sum = {}
        enhanced_metrics_sum = {}
        improvements_sum = {}
        
        for query_data in test_queries:
            result = self.evaluate_single_query(query_data)
            all_results.append(result)
            
            # Accumulate metrics for averaging
            for metric, value in result["baseline_metrics"].items():
                baseline_metrics_sum[metric] = baseline_metrics_sum.get(metric, 0) + value
            
            for metric, value in result["enhanced_metrics"].items():
                enhanced_metrics_sum[metric] = enhanced_metrics_sum.get(metric, 0) + value
            
            for metric, value in result["improvements"].items():
                improvements_sum[metric] = improvements_sum.get(metric, 0) + value
        
        # Calculate averages
        num_queries = len(test_queries)
        avg_baseline_metrics = {k: v / num_queries for k, v in baseline_metrics_sum.items()}
        avg_enhanced_metrics = {k: v / num_queries for k, v in enhanced_metrics_sum.items()}
        avg_improvements = {k: v / num_queries for k, v in improvements_sum.items()}
        
        # Prepare final results
        evaluation_results = {
            "evaluation_config": {
                "num_queries": num_queries,
                "metrics_computed": self.config.metrics_to_compute,
                "enhancement_config": {
                    "expansion_strategies": ["synonym", "paraphrase", "domain_specific"],
                    "max_expanded_queries": 5,
                    "reranker_model": "cross-encoder/ms-marco-MiniLM-L-6-v2"
                }
            },
            "average_baseline_metrics": avg_baseline_metrics,
            "average_enhanced_metrics": avg_enhanced_metrics,
            "average_improvements": avg_improvements,
            "detailed_results": all_results,
            "summary": self._generate_summary(avg_baseline_metrics, avg_enhanced_metrics, avg_improvements)
        }
        
        # Save results
        self._save_results(evaluation_results)
        
        # Print summary
        self._print_summary(evaluation_results)
        
        return evaluation_results
    
    def _generate_summary(self, baseline_metrics: Dict, enhanced_metrics: Dict, improvements: Dict) -> Dict:
        """Generate a summary of evaluation results"""
        summary = {
            "key_findings": [],
            "significant_improvements": [],
            "performance_impact": {}
        }
        
        # Identify significant improvements (>10%)
        for metric, improvement in improvements.items():
            if improvement > 0.1:  # 10% improvement
                summary["significant_improvements"].append({
                    "metric": metric,
                    "improvement_percent": improvement * 100,
                    "baseline": baseline_metrics.get(metric, 0),
                    "enhanced": enhanced_metrics.get(metric, 0)
                })
        
        # Performance impact
        baseline_time = baseline_metrics.get("retrieval_time", 0)
        enhanced_time = enhanced_metrics.get("retrieval_time", 0)
        if baseline_time > 0:
            time_overhead = (enhanced_time - baseline_time) / baseline_time
            summary["performance_impact"] = {
                "baseline_time_ms": baseline_time * 1000,
                "enhanced_time_ms": enhanced_time * 1000,
                "overhead_percent": time_overhead * 100
            }
        
        # Key findings
        if summary["significant_improvements"]:
            summary["key_findings"].append(
                f"Enhanced retrieval shows significant improvements in {len(summary['significant_improvements'])} metrics"
            )
        
        ndcg_improvement = improvements.get("ndcg_at_5", 0)
        if ndcg_improvement > 0:
            summary["key_findings"].append(
                f"NDCG@5 improved by {ndcg_improvement*100:.1f}% indicating better ranking quality"
            )
        
        return summary
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results to files"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save complete results
        with open(output_path / "retrieval_evaluation_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary separately
        with open(output_path / "evaluation_summary.json", 'w') as f:
            json.dump(results["summary"], f, indent=2, default=str)
        
        logger.info(f"Evaluation results saved to {output_path}")
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print evaluation summary to console"""
        print("\n" + "="*80)
        print("ENHANCED RETRIEVAL EVALUATION RESULTS")
        print("="*80)
        
        print(f"\nEvaluated {results['evaluation_config']['num_queries']} queries")
        print(f"Enhancement: Query expansion + reranking")
        
        print("\nAVERAGE METRICS:")
        print("-" * 40)
        
        baseline = results["average_baseline_metrics"]
        enhanced = results["average_enhanced_metrics"]
        improvements = results["average_improvements"]
        
        key_metrics = ["precision_at_5", "recall_at_5", "ndcg_at_5", "relevance_score", "diversity_score"]
        
        for metric in key_metrics:
            if metric in baseline and metric in enhanced:
                b_val = baseline[metric]
                e_val = enhanced[metric]
                imp = improvements.get(metric, 0) * 100
                
                print(f"{metric:20}: {b_val:.3f} → {e_val:.3f} ({imp:+.1f}%)")
        
        print("\nPERFORMANCE IMPACT:")
        print("-" * 40)
        perf_impact = results["summary"]["performance_impact"]
        if perf_impact:
            baseline_ms = perf_impact["baseline_time_ms"]
            enhanced_ms = perf_impact["enhanced_time_ms"]
            overhead = perf_impact["overhead_percent"]
            
            print(f"Retrieval time:       {baseline_ms:.1f}ms → {enhanced_ms:.1f}ms ({overhead:+.1f}%)")
        
        print("\nKEY FINDINGS:")
        print("-" * 40)
        for finding in results["summary"]["key_findings"]:
            print(f"• {finding}")
        
        if results["summary"]["significant_improvements"]:
            print("\nSIGNIFICANT IMPROVEMENTS (>10%):")
            for imp in results["summary"]["significant_improvements"]:
                print(f"• {imp['metric']}: {imp['improvement_percent']:+.1f}%")
        
        print("\n" + "="*80)
