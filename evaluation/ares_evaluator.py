"""
ARES (Automated RAG Evaluation System) Evaluation Script for LLM Twin
Advanced RAG evaluation using synthetic data generation and automated assessment
"""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import re
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

class ARESEvaluator:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        print(f"Loading model from {self.model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model = self.model.to(self.device)
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model loaded on device: {self.device}")
    
    def generate_synthetic_rag_data(self) -> Dict:
        """Generate synthetic RAG evaluation data using the model itself"""
        print("Generating synthetic RAG evaluation data...")
        
        # Base documents for synthetic data generation
        base_documents = [
            {
                "content": "Artificial Intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence. These include learning, reasoning, problem-solving, perception, and language understanding. AI systems can be classified into narrow AI (designed for specific tasks) and general AI (designed for any intellectual task). Modern AI approaches include machine learning, deep learning, neural networks, and natural language processing.",
                "topic": "artificial_intelligence"
            },
            {
                "content": "Climate change refers to significant changes in global temperatures and weather patterns over time. While climate change is natural, human activities have been the main driver since the 1950s. The primary cause is the emission of greenhouse gases like carbon dioxide, methane, and nitrous oxide from burning fossil fuels, deforestation, and industrial processes. Effects include rising temperatures, sea-level rise, extreme weather events, and ecosystem disruption. Mitigation strategies include renewable energy adoption, energy efficiency, and carbon capture technologies.",
                "topic": "climate_change"
            },
            {
                "content": "Machine Learning (ML) is a subset of AI that enables systems to learn and improve from experience without being explicitly programmed. ML algorithms build mathematical models based on sample data, known as training data, to make predictions or decisions. Types include supervised learning (labeled data), unsupervised learning (unlabeled data), and reinforcement learning (reward-based learning). Common applications include image recognition, natural language processing, recommendation systems, and autonomous vehicles. Key algorithms include neural networks, decision trees, support vector machines, and clustering algorithms.",
                "topic": "machine_learning"
            },
            {
                "content": "Renewable energy sources are naturally replenished and environmentally friendly alternatives to fossil fuels. Major types include solar energy (photovoltaic cells and thermal systems), wind energy (turbines converting wind to electricity), hydroelectric power (water flow driving turbines), geothermal energy (heat from Earth's core), and biomass (organic matter conversion). Benefits include reduced greenhouse gas emissions, energy independence, and sustainability. Challenges include intermittency, storage requirements, and initial infrastructure costs. Global renewable capacity has been growing rapidly, with solar and wind leading the expansion.",
                "topic": "renewable_energy"
            },
            {
                "content": "The Internet of Things (IoT) connects everyday physical objects to the internet, enabling data collection and exchange. IoT devices include smart home appliances, wearable technology, industrial sensors, and connected vehicles. These devices use sensors, processors, and communication hardware to collect and exchange data. Applications include smart homes, industrial automation, healthcare monitoring, and smart cities. Key technologies include wireless communication protocols, edge computing, cloud platforms, and security frameworks. Challenges include privacy concerns, security vulnerabilities, interoperability standards, and data management.",
                "topic": "internet_of_things"
            }
        ]
        
        # Generate synthetic questions and answers for each document
        synthetic_data = []
        
        for doc in base_documents:
            # Generate questions for each document
            questions = [
                f"What are the main components of {doc['topic'].replace('_', ' ')}?",
                f"How does {doc['topic'].replace('_', ' ')} work in practice?",
                f"What are the key applications of {doc['topic'].replace('_', ' ')}?",
                f"What are the advantages and disadvantages of {doc['topic'].replace('_', ' ')}?",
                f"How has {doc['topic'].replace('_', ' ')} evolved over time?"
            ]
            
            for question in questions:
                # Generate synthetic answer using the model
                answer = self.generate_synthetic_answer(question, doc["content"])
                
                synthetic_data.append({
                    "question": question,
                    "context": doc["content"],
                    "synthetic_answer": answer,
                    "topic": doc["topic"]
                })
        
        return synthetic_data
    
    def generate_synthetic_answer(self, question: str, context: str) -> str:
        """Generate synthetic answer using the model"""
        prompt = f"""Based on the following context, please provide a comprehensive answer to the question.

Context:
{context}

Question: {question}

Answer:"""
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Create attention mask
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=120,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        answer = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return answer.strip()
    
    def evaluate_context_utilization(self, question: str, context: str, answer: str) -> Dict:
        """Evaluate how well the answer utilizes the provided context"""
        utilization_scores = {}
        
        context_words = set(context.lower().split())
        answer_words = set(answer.lower().split())
        
        # Word overlap between context and answer
        overlap = len(context_words.intersection(answer_words))
        utilization_scores["word_utilization"] = overlap / max(len(answer_words), 1)
        
        # Key concept utilization
        context_sentences = context.split('.')
        utilized_sentences = 0
        for sentence in context_sentences:
            sentence_words = set(sentence.lower().split())
            if len(sentence_words.intersection(answer_words)) > 2:  # At least 3 overlapping words
                utilized_sentences += 1
        
        utilization_scores["sentence_utilization"] = utilized_sentences / max(len(context_sentences), 1)
        
        # Information density (how much context information is used)
        context_key_terms = [word for word in context.lower().split() if len(word) > 5]
        used_key_terms = sum(1 for term in context_key_terms if term in answer.lower())
        utilization_scores["key_term_utilization"] = used_key_terms / max(len(context_key_terms), 1)
        
        # Overall utilization
        utilization_scores["overall"] = (utilization_scores["word_utilization"] * 0.4 + 
                                      utilization_scores["sentence_utilization"] * 0.3 + 
                                      utilization_scores["key_term_utilization"] * 0.3)
        
        return utilization_scores
    
    def evaluate_answer_coherence(self, answer: str) -> Dict:
        """Evaluate the coherence and structure of the answer"""
        coherence_scores = {}
        
        # Sentence structure
        sentences = [s.strip() for s in answer.split('.') if s.strip()]
        coherence_scores["sentence_count"] = len(sentences)
        
        # Average sentence length
        if sentences:
            avg_length = sum(len(s.split()) for s in sentences) / len(sentences)
            coherence_scores["avg_sentence_length"] = min(avg_length / 15, 1.0)  # Normalize to 0-1
        else:
            coherence_scores["avg_sentence_length"] = 0.0
        
        # Logical flow indicators
        flow_indicators = ["therefore", "however", "additionally", "furthermore", "consequently", "moreover"]
        flow_count = sum(1 for indicator in flow_indicators if indicator in answer.lower())
        coherence_scores["logical_flow"] = min(flow_count / 3, 1.0)  # Normalize to 0-1
        
        # Repetition (lower is better)
        words = answer.lower().split()
        unique_words = len(set(words))
        repetition_score = unique_words / max(len(words), 1)
        coherence_scores["low_repetition"] = repetition_score
        
        # Overall coherence
        coherence_scores["overall"] = (coherence_scores["avg_sentence_length"] * 0.3 + 
                                      coherence_scores["logical_flow"] * 0.3 + 
                                      coherence_scores["low_repetition"] * 0.4)
        
        return coherence_scores
    
    def evaluate_question_answer_alignment(self, question: str, answer: str) -> Dict:
        """Evaluate how well the answer addresses the question"""
        alignment_scores = {}
        
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Word overlap
        overlap = len(question_words.intersection(answer_words))
        alignment_scores["word_overlap"] = overlap / max(len(question_words), 1)
        
        # Question type matching
        question_types = {
            "what": ["is", "are", "refers to", "means", "definition"],
            "how": ["process", "method", "steps", "way", "works"],
            "why": ["because", "reason", "cause", "due to"],
            "when": ["time", "year", "date", "period", "era"],
            "where": ["location", "place", "area", "region"]
        }
        
        matched_type = False
        for q_type, indicators in question_types.items():
            if q_type in question.lower():
                if isinstance(indicators, list):
                    matched_type = any(indicator in answer.lower() for indicator in indicators)
                else:
                    matched_type = indicators in answer.lower()
                break
        
        alignment_scores["type_match"] = 1.0 if matched_type else 0.5
        
        # Completeness (does answer seem complete)
        answer_length = len(answer.split())
        alignment_scores["completeness"] = min(answer_length / 25, 1.0)  # Expect at least 25 words
        
        # Overall alignment
        alignment_scores["overall"] = (alignment_scores["word_overlap"] * 0.4 + 
                                      alignment_scores["type_match"] * 0.3 + 
                                      alignment_scores["completeness"] * 0.3)
        
        return alignment_scores
    
    def evaluate_synthetic_rag(self, question: str, context: str, synthetic_answer: str, topic: str) -> Dict:
        """Evaluate synthetic RAG example using ARES methodology"""
        
        # Generate model's answer for comparison
        model_answer = self.generate_synthetic_answer(question, context)
        
        # Evaluate different ARES metrics
        context_utilization = self.evaluate_context_utilization(question, context, model_answer)
        answer_coherence = self.evaluate_answer_coherence(model_answer)
        question_alignment = self.evaluate_question_answer_alignment(question, model_answer)
        
        # Compare with synthetic answer (ground truth)
        synthetic_coherence = self.evaluate_answer_coherence(synthetic_answer)
        
        # Answer similarity with synthetic answer
        similarity = self.calculate_answer_similarity(model_answer, synthetic_answer)
        
        # Calculate overall ARES score
        overall_score = (context_utilization["overall"] * 0.3 + 
                        answer_coherence["overall"] * 0.3 + 
                        question_alignment["overall"] * 0.2 + 
                        similarity * 0.2)
        
        return {
            "question": question,
            "context": context,
            "synthetic_answer": synthetic_answer,
            "model_answer": model_answer,
            "topic": topic,
            "context_utilization": context_utilization,
            "answer_coherence": answer_coherence,
            "question_alignment": question_alignment,
            "synthetic_coherence": synthetic_coherence,
            "answer_similarity": similarity,
            "overall_score": overall_score
        }
    
    def calculate_answer_similarity(self, answer1: str, answer2: str) -> float:
        """Calculate similarity between two answers"""
        words1 = set(answer1.lower().split())
        words2 = set(answer2.lower().split())
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def evaluate_dataset(self, synthetic_data, num_samples: int = None, output_dir: str = "eval_results") -> Dict:
        """Evaluate on the synthetic dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if num_samples:
            synthetic_data = synthetic_data[:min(num_samples, len(synthetic_data))]
        
        results = []
        topic_scores = {}
        
        print(f"Evaluating on {len(synthetic_data)} ARES RAG tasks...")
        
        for i, example in enumerate(tqdm(synthetic_data)):
            try:
                result = self.evaluate_synthetic_rag(
                    example["question"],
                    example["context"],
                    example["synthetic_answer"],
                    example["topic"]
                )
                
                result["example_id"] = i
                results.append(result)
                
                # Track topic scores
                topic = example["topic"]
                if topic not in topic_scores:
                    topic_scores[topic] = {
                        "total": 0,
                        "score_sum": 0,
                        "utilization_sum": 0,
                        "coherence_sum": 0,
                        "alignment_sum": 0,
                        "similarity_sum": 0
                    }
                topic_scores[topic]["total"] += 1
                topic_scores[topic]["score_sum"] += result["overall_score"]
                topic_scores[topic]["utilization_sum"] += result["context_utilization"]["overall"]
                topic_scores[topic]["coherence_sum"] += result["answer_coherence"]["overall"]
                topic_scores[topic]["alignment_sum"] += result["question_alignment"]["overall"]
                topic_scores[topic]["similarity_sum"] += result["answer_similarity"]
                
                # Save intermediate results every 3 examples
                if (i + 1) % 3 == 0:
                    self.save_results(results[:i+1], output_path / f"intermediate_ares_{i+1}.json")
                    
            except Exception as e:
                print(f"Error on example {i}: {e}")
                continue
        
        # Calculate final metrics
        total_examples = len(results)
        overall_score = sum(r["overall_score"] for r in results) / total_examples if results else 0
        
        # Calculate component averages
        avg_utilization = sum(r["context_utilization"]["overall"] for r in results) / total_examples
        avg_coherence = sum(r["answer_coherence"]["overall"] for r in results) / total_examples
        avg_alignment = sum(r["question_alignment"]["overall"] for r in results) / total_examples
        avg_similarity = sum(r["answer_similarity"] for r in results) / total_examples
        
        # Calculate topic-wise averages
        topic_averages = {}
        for topic, stats in topic_scores.items():
            topic_averages[topic] = {
                "overall": stats["score_sum"] / stats["total"],
                "utilization": stats["utilization_sum"] / stats["total"],
                "coherence": stats["coherence_sum"] / stats["total"],
                "alignment": stats["alignment_sum"] / stats["total"],
                "similarity": stats["similarity_sum"] / stats["total"]
            }
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_examples": total_examples,
            "overall_score": overall_score,
            "avg_context_utilization": avg_utilization,
            "avg_answer_coherence": avg_coherence,
            "avg_question_alignment": avg_alignment,
            "avg_answer_similarity": avg_similarity,
            "topic_averages": topic_averages,
            "topic_stats": topic_scores,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "ares_results.json")
        
        # Print summary
        self.print_summary(final_results)
        
        return final_results
    
    def save_results(self, results: Dict, filepath: Path):
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def print_summary(self, results: Dict):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("ARES Evaluation Results")
        print("="*60)
        print(f"Model: {results['model_path']}")
        print(f"Total Examples: {results['total_examples']}")
        print(f"Overall Score: {results['overall_score']:.3f}/1.0")
        print("\nComponent Scores:")
        print(f"  Context Utilization: {results['avg_context_utilization']:.3f}/1.0")
        print(f"  Answer Coherence: {results['avg_answer_coherence']:.3f}/1.0")
        print(f"  Question Alignment: {results['avg_question_alignment']:.3f}/1.0")
        print(f"  Answer Similarity: {results['avg_answer_similarity']:.3f}/1.0")
        
        print("\nTopic-wise Performance:")
        for topic, scores in sorted(results['topic_averages'].items(), 
                                  key=lambda x: x[1]['overall'], reverse=True):
            total = results['topic_stats'][topic]['total']
            print(f"  {topic}: {scores['overall']:.3f} ({total} examples)")
            print(f"    Utilization: {scores['utilization']:.3f}")
            print(f"    Coherence: {scores['coherence']:.3f}")
            print(f"    Alignment: {scores['alignment']:.3f}")
            print(f"    Similarity: {scores['similarity']:.3f}")
        
        # Show examples
        sorted_results = sorted(results["detailed_results"], key=lambda x: x["overall_score"], reverse=True)
        best_result = sorted_results[0]
        worst_result = sorted_results[-1]
        
        print(f"\nBest Performing ARES Example:")
        print(f"  Topic: {best_result['topic']}")
        print(f"  Overall Score: {best_result['overall_score']:.3f}")
        print(f"  Question: {best_result['question']}")
        print(f"  Model Answer: {best_result['model_answer'][:80]}...")
        
        print(f"\nLowest Performing ARES Example:")
        print(f"  Topic: {worst_result['topic']}")
        print(f"  Overall Score: {worst_result['overall_score']:.3f}")
        print(f"  Question: {worst_result['question']}")
        print(f"  Model Answer: {worst_result['model_answer'][:80]}...")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on ARES metrics")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = ARESEvaluator(args.model, args.device)
    synthetic_data = evaluator.generate_synthetic_rag_data()
    results = evaluator.evaluate_dataset(synthetic_data, args.samples, args.output)

if __name__ == "__main__":
    main()
