#!/usr/bin/env python3
"""
Comprehensive RAG Evaluation Test using RAGAS and ARES
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ragas_evaluator import RAGASEvaluator
from ares_evaluator import ARESEvaluator

def main():
    print("="*70)
    print("Comprehensive RAG Evaluation Test (RAGAS + ARES)")
    print("="*70)
    
    model_path = "/Users/ronel/Downloads/llm twin/dpo_llm_twin"
    
    # Test RAGAS evaluation
    print("\n1. Testing RAGAS Evaluation...")
    ragas_eval = RAGASEvaluator(model_path)
    ragas_dataset = ragas_eval.load_rag_dataset()
    
    # Test with small sample
    ragas_results = ragas_eval.evaluate_dataset(ragas_dataset, num_samples=5, output_dir="test_results/ragas")
    
    print(f"\nRAGAS Results:")
    print(f"Overall Score: {ragas_results['overall_score']:.3f}/1.0")
    print(f"Faithfulness: {ragas_results['avg_faithfulness']:.3f}/1.0")
    print(f"Answer Relevance: {ragas_results['avg_relevance']:.3f}/1.0")
    print(f"Context Recall: {ragas_results['avg_recall']:.3f}/1.0")
    print(f"Context Precision: {ragas_results['avg_precision']:.3f}/1.0")
    
    # Test ARES evaluation
    print("\n2. Testing ARES Evaluation...")
    ares_eval = ARESEvaluator(model_path)
    ares_data = ares_eval.generate_synthetic_rag_data()
    
    # Test with small sample
    ares_results = ares_eval.evaluate_dataset(ares_data, num_samples=5, output_dir="test_results/ares")
    
    print(f"\nARES Results:")
    print(f"Overall Score: {ares_results['overall_score']:.3f}/1.0")
    print(f"Context Utilization: {ares_results['avg_context_utilization']:.3f}/1.0")
    print(f"Answer Coherence: {ares_results['avg_answer_coherence']:.3f}/1.0")
    print(f"Question Alignment: {ares_results['avg_question_alignment']:.3f}/1.0")
    print(f"Answer Similarity: {ares_results['avg_answer_similarity']:.3f}/1.0")
    
    # Combined analysis
    print("\n" + "="*70)
    print("Combined RAG Performance Analysis")
    print("="*70)
    
    # RAGAS assessment
    print(f"\nRAGAS Framework Assessment:")
    if ragas_results['overall_score'] > 0.8:
        print("✅ Excellent RAG performance - High quality retrieval and generation")
    elif ragas_results['overall_score'] > 0.6:
        print("📈 Good RAG performance - Solid retrieval and generation capabilities")
    elif ragas_results['overall_score'] > 0.4:
        print("⚠️  Fair RAG performance - Some improvement needed")
    else:
        print("❌ Poor RAG performance - Significant issues identified")
    
    # ARES assessment
    print(f"\nARES Framework Assessment:")
    if ares_results['overall_score'] > 0.8:
        print("✅ Excellent synthetic RAG performance - Strong context utilization")
    elif ares_results['overall_score'] > 0.6:
        print("📈 Good synthetic RAG performance - Decent context usage")
    elif ares_results['overall_score'] > 0.4:
        print("⚠️  Fair synthetic RAG performance - Context utilization needs work")
    else:
        print("❌ Poor synthetic RAG performance - Weak context utilization")
    
    # Component breakdown
    print(f"\nDetailed Component Analysis:")
    print(f"RAGAS Components:")
    print(f"  • Faithfulness (Context Adherence): {ragas_results['avg_faithfulness']:.3f}")
    print(f"  • Answer Relevance (Question Matching): {ragas_results['avg_relevance']:.3f}")
    print(f"  • Context Recall (Information Coverage): {ragas_results['avg_recall']:.3f}")
    print(f"  • Context Precision (Retrieval Quality): {ragas_results['avg_precision']:.3f}")
    
    print(f"\nARES Components:")
    print(f"  • Context Utilization: {ares_results['avg_context_utilization']:.3f}")
    print(f"  • Answer Coherence: {ares_results['avg_answer_coherence']:.3f}")
    print(f"  • Question Alignment: {ares_results['avg_question_alignment']:.3f}")
    print(f"  • Answer Similarity: {ares_results['avg_answer_similarity']:.3f}")
    
    # Overall RAG capability assessment
    combined_score = (ragas_results['overall_score'] + ares_results['overall_score']) / 2
    
    print(f"\nOverall RAG Capability Score: {combined_score:.3f}/1.0")
    
    if combined_score > 0.8:
        print("🏆 EXCELLENT: Your LLM twin demonstrates outstanding RAG capabilities!")
        print("   Strong retrieval utilization and faithful answer generation")
    elif combined_score > 0.6:
        print("🎯 GOOD: Your LLM twin shows solid RAG performance")
        print("   Good balance of retrieval usage and generation quality")
    elif combined_score > 0.4:
        print("📊 FAIR: Your LLM twin has basic RAG capabilities")
        print("   Some areas need improvement for production use")
    else:
        print("⚠️  NEEDS WORK: RAG capabilities require significant improvement")
        print("   Focus on better context utilization and answer generation")
    
    # Recommendations
    print(f"\n📋 Recommendations:")
    
    if ragas_results['avg_faithfulness'] < 0.7:
        print("• Improve faithfulness: Better adherence to provided context")
    
    if ragas_results['avg_relevance'] < 0.7:
        print("• Improve relevance: Better question-answer alignment")
    
    if ares_results['avg_context_utilization'] < 0.7:
        print("• Improve utilization: Better use of retrieved information")
    
    if ares_results['avg_answer_coherence'] < 0.7:
        print("• Improve coherence: Better structured and fluent answers")
    
    # Category performance highlights
    print(f"\n📈 Category Performance Highlights:")
    
    # RAGAS categories
    best_ragas_category = max(ragas_results['category_averages'].items(), key=lambda x: x[1]['overall'])
    print(f"RAGAS Best Category: {best_ragas_category[0]} ({best_ragas_category[1]['overall']:.3f})")
    
    # ARES topics
    best_ares_topic = max(ares_results['topic_averages'].items(), key=lambda x: x[1]['overall'])
    print(f"ARES Best Topic: {best_ares_topic[0]} ({best_ares_topic[1]['overall']:.3f})")
    
    print("="*70)

if __name__ == "__main__":
    main()
