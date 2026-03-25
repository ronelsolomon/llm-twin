"""
RAGAS (Retrieval-Augmented Generation Assessment) Evaluation Script for LLM Twin
Evaluates RAG systems using metrics like faithfulness, answer relevance, context recall
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

class RAGASEvaluator:
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
    
    def load_rag_dataset(self) -> Dict:
        """Load RAG evaluation dataset with questions, contexts, and ground truth"""
        print("Loading RAG evaluation dataset...")
        
        # Create synthetic RAG dataset
        dataset = [
            {
                "question": "What are the main causes of climate change?",
                "contexts": [
                    "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is natural, human activities have been the main driver since the 1950s. The burning of fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around Earth, trapping heat and raising temperatures.",
                    "The primary greenhouse gases include carbon dioxide (CO2), methane (CH4), and nitrous oxide (N2O). CO2 is responsible for about 76% of all greenhouse gas emissions. These gases are released through burning fossil fuels, industrial processes, and deforestation."
                ],
                "ground_truth": "The main causes of climate change are greenhouse gas emissions from burning fossil fuels, industrial processes, and deforestation, with CO2 being the primary greenhouse gas.",
                "category": "environment"
            },
            {
                "question": "How does machine learning work?",
                "contexts": [
                    "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
                    "The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future based on the examples that we provide."
                ],
                "ground_truth": "Machine learning works by analyzing data to find patterns and make decisions without explicit programming, using observations and examples to improve performance over time.",
                "category": "technology"
            },
            {
                "question": "What are the health benefits of regular exercise?",
                "contexts": [
                    "Exercise provides numerous health benefits, including improved cardiovascular health, stronger muscles and bones, better mental health, and increased longevity. Regular physical activity can help prevent chronic diseases like diabetes, heart disease, and certain types of cancer.",
                    "Health experts recommend at least 150 minutes of moderate exercise per week for optimal health benefits. This can include activities like walking, swimming, cycling, or strength training."
                ],
                "ground_truth": "Regular exercise improves cardiovascular health, strengthens muscles and bones, enhances mental health, prevents chronic diseases, and increases longevity, with experts recommending 150 minutes weekly.",
                "category": "health"
            },
            {
                "question": "How does the Internet of Things (IoT) work?",
                "contexts": [
                    "The Internet of Things (IoT) refers to the network of physical devices, vehicles, home appliances, and other items embedded with sensors, software, and other technologies that enable these objects to connect and exchange data over the internet.",
                    "IoT devices collect data through sensors, process it locally or in the cloud, and can communicate with other devices. This creates smart environments where devices can make automated decisions based on real-time data."
                ],
                "ground_truth": "IoT works by embedding sensors and software in physical devices to collect data, enabling internet connectivity and data exchange between devices for automated decision-making.",
                "category": "technology"
            },
            {
                "question": "What are the main components of a healthy diet?",
                "contexts": [
                    "A healthy diet includes a variety of foods from all food groups: fruits, vegetables, grains, protein foods, and dairy or fortified soy alternatives. Limiting foods high in saturated fats, added sugars, and sodium is also important.",
                    "Nutrition experts recommend eating a rainbow of fruits and vegetables, whole grains, lean proteins, and healthy fats. Portion control and balance are key principles of healthy eating."
                ],
                "ground_truth": "A healthy diet consists of fruits, vegetables, whole grains, lean proteins, and healthy fats, while limiting saturated fats, added sugars, and sodium with proper portion control.",
                "category": "health"
            },
            {
                "question": "How do renewable energy sources work?",
                "contexts": [
                    "Renewable energy sources are naturally replenished on a human timescale. The main types include solar, wind, hydroelectric, geothermal, and biomass. These sources are becoming increasingly important as the world seeks to reduce carbon emissions.",
                    "Solar energy works by converting sunlight into electricity through photovoltaic cells. Wind energy uses turbines to convert wind kinetic energy into electrical power. Both technologies require specific environmental conditions to be effective."
                ],
                "ground_truth": "Renewable energy sources work by converting natural processes like sunlight, wind, and water flow into electricity through technologies such as solar panels, wind turbines, and hydroelectric systems.",
                "category": "environment"
            },
            {
                "question": "What are the key principles of effective communication?",
                "contexts": [
                    "Effective communication involves clear expression, active listening, and appropriate non-verbal cues. It requires understanding the audience, choosing the right medium, and ensuring the message is received and understood as intended.",
                    "Key communication skills include clarity, conciseness, empathy, and feedback. Good communicators adapt their style to different situations and audiences, and they listen as much as they speak."
                ],
                "ground_truth": "Effective communication requires clear expression, active listening, audience awareness, appropriate medium selection, and skills like clarity, conciseness, empathy, and feedback.",
                "category": "communication"
            },
            {
                "question": "How do vaccines work in the human body?",
                "contexts": [
                    "Vaccines work by introducing a harmless piece of a disease-causing organism to stimulate the immune system. This triggers the production of antibodies, providing immunity without causing the actual disease.",
                    "When vaccinated, the body's immune system recognizes the vaccine as foreign and builds antibodies against it. This creates immunological memory, allowing the body to fight off future infections more effectively."
                ],
                "ground_truth": "Vaccines work by introducing harmless disease components to stimulate antibody production and create immunological memory, enabling the body to fight future infections without causing the actual disease.",
                "category": "health"
            }
        ]
        
        return dataset
    
    def generate_rag_response(self, question: str, contexts: List[str]) -> str:
        """Generate response using RAG-style context"""
        # Combine contexts into a single prompt
        context_text = "\n\n".join([f"Context {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        prompt = f"""Based on the following contexts, please answer the question. Use only the information provided in the contexts.

{context_text}

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
                max_new_tokens=150,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    
    def evaluate_faithfulness(self, response: str, contexts: List[str]) -> Dict:
        """Evaluate if response is faithful to provided contexts"""
        faithfulness_scores = {}
        
        response_sentences = [s.strip() for s in response.split('.') if s.strip()]
        context_text = " ".join(contexts).lower()
        
        faithful_sentences = 0
        total_sentences = len(response_sentences)
        
        for sentence in response_sentences:
            sentence_lower = sentence.lower()
            
            # Check if key claims in sentence are supported by context
            sentence_words = set(sentence_lower.split())
            context_words = set(context_text.split())
            
            # Calculate overlap
            overlap = len(sentence_words.intersection(context_words))
            overlap_ratio = overlap / max(len(sentence_words), 1)
            
            # Consider sentence faithful if significant overlap
            if overlap_ratio > 0.3:  # 30% overlap threshold
                faithful_sentences += 1
        
        faithfulness_scores["faithfulness"] = faithful_sentences / max(total_sentences, 1)
        
        # Check for contradictions
        contradiction_indicators = ["however", "although", "despite", "but"]
        has_contradiction = any(indicator in response.lower() for indicator in contradiction_indicators)
        faithfulness_scores["no_contradiction"] = 1.0 if not has_contradiction else 0.7
        
        # Overall faithfulness
        faithfulness_scores["overall"] = (faithfulness_scores["faithfulness"] * 0.7 + 
                                         faithfulness_scores["no_contradiction"] * 0.3)
        
        return faithfulness_scores
    
    def evaluate_answer_relevance(self, question: str, response: str) -> Dict:
        """Evaluate if answer is relevant to the question"""
        relevance_scores = {}
        
        question_words = set(question.lower().split())
        response_words = set(response.lower().split())
        
        # Word overlap
        overlap = len(question_words.intersection(response_words))
        overlap_ratio = overlap / max(len(question_words), 1)
        relevance_scores["word_overlap"] = overlap_ratio
        
        # Check if response addresses question type
        question_indicators = {
            "what": ["is", "are", "refers to", "means"],
            "how": ["works", "process", "steps", "method"],
            "why": "because",
            "when": ["time", "year", "date", "period"],
            "where": ["location", "place", "area"]
        }
        
        question_type = None
        for q_type, indicators in question_indicators.items():
            if q_type in question.lower():
                question_type = q_type
                break
        
        if question_type and isinstance(indicators, list):
            has_indicator = any(indicator in response.lower() for indicator in indicators)
            relevance_scores["question_type_match"] = 1.0 if has_indicator else 0.5
        else:
            relevance_scores["question_type_match"] = 0.7  # Neutral
        
        # Response completeness
        response_length = len(response.split())
        if response_length > 10:
            relevance_scores["completeness"] = min(response_length / 30, 1.0)
        else:
            relevance_scores["completeness"] = response_length / 10
        
        # Overall relevance
        relevance_scores["overall"] = (relevance_scores["word_overlap"] * 0.4 + 
                                    relevance_scores["question_type_match"] * 0.3 + 
                                    relevance_scores["completeness"] * 0.3)
        
        return relevance_scores
    
    def evaluate_context_recall(self, ground_truth: str, contexts: List[str]) -> Dict:
        """Evaluate if contexts contain information needed for ground truth"""
        recall_scores = {}
        
        gt_words = set(ground_truth.lower().split())
        context_text = " ".join(contexts).lower()
        context_words = set(context_text.split())
        
        # Word overlap between ground truth and contexts
        overlap = len(gt_words.intersection(context_words))
        recall_scores["word_recall"] = overlap / max(len(gt_words), 1)
        
        # Check for key concepts
        gt_concepts = [word for word in ground_truth.lower().split() if len(word) > 4]
        found_concepts = sum(1 for concept in gt_concepts if concept in context_text)
        recall_scores["concept_recall"] = found_concepts / max(len(gt_concepts), 1)
        
        # Overall recall
        recall_scores["overall"] = (recall_scores["word_recall"] * 0.6 + 
                                   recall_scores["concept_recall"] * 0.4)
        
        return recall_scores
    
    def evaluate_context_precision(self, question: str, contexts: List[str]) -> Dict:
        """Evaluate if contexts are relevant to the question"""
        precision_scores = {}
        
        question_words = set(question.lower().split())
        relevant_contexts = 0
        
        for context in contexts:
            context_words = set(context.lower().split())
            overlap = len(question_words.intersection(context_words))
            overlap_ratio = overlap / max(len(question_words), 1)
            
            # Consider context relevant if significant overlap
            if overlap_ratio > 0.2:  # 20% overlap threshold
                relevant_contexts += 1
        
        precision_scores["context_precision"] = relevant_contexts / len(contexts)
        
        # Check context quality (length and information density)
        avg_context_length = sum(len(ctx.split()) for ctx in contexts) / len(contexts)
        precision_scores["context_quality"] = min(avg_context_length / 50, 1.0)  # Normalize to 0-1
        
        # Overall precision
        precision_scores["overall"] = (precision_scores["context_precision"] * 0.7 + 
                                      precision_scores["context_quality"] * 0.3)
        
        return precision_scores
    
    def evaluate_single_rag(self, question: str, contexts: List[str], ground_truth: str, category: str) -> Dict:
        """Evaluate model on a single RAG task"""
        
        # Generate response
        response = self.generate_rag_response(question, contexts)
        
        # Evaluate different RAGAS metrics
        faithfulness = self.evaluate_faithfulness(response, contexts)
        answer_relevance = self.evaluate_answer_relevance(question, response)
        context_recall = self.evaluate_context_recall(ground_truth, contexts)
        context_precision = self.evaluate_context_precision(question, contexts)
        
        # Calculate overall RAGAS score
        overall_score = (faithfulness["overall"] * 0.3 + 
                        answer_relevance["overall"] * 0.3 + 
                        context_recall["overall"] * 0.2 + 
                        context_precision["overall"] * 0.2)
        
        return {
            "question": question,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "generated_response": response,
            "category": category,
            "faithfulness": faithfulness,
            "answer_relevance": answer_relevance,
            "context_recall": context_recall,
            "context_precision": context_precision,
            "overall_score": overall_score
        }
    
    def evaluate_dataset(self, dataset, num_samples: int = None, output_dir: str = "eval_results") -> Dict:
        """Evaluate on the entire dataset or sample"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if num_samples:
            dataset = dataset[:min(num_samples, len(dataset))]
        
        results = []
        category_scores = {}
        
        print(f"Evaluating on {len(dataset)} RAG tasks...")
        
        for i, example in enumerate(tqdm(dataset)):
            try:
                result = self.evaluate_single_rag(
                    example["question"],
                    example["contexts"],
                    example["ground_truth"],
                    example["category"]
                )
                
                result["example_id"] = i
                results.append(result)
                
                # Track category scores
                category = example["category"]
                if category not in category_scores:
                    category_scores[category] = {
                        "total": 0, 
                        "score_sum": 0,
                        "faithfulness_sum": 0,
                        "relevance_sum": 0,
                        "recall_sum": 0,
                        "precision_sum": 0
                    }
                category_scores[category]["total"] += 1
                category_scores[category]["score_sum"] += result["overall_score"]
                category_scores[category]["faithfulness_sum"] += result["faithfulness"]["overall"]
                category_scores[category]["relevance_sum"] += result["answer_relevance"]["overall"]
                category_scores[category]["recall_sum"] += result["context_recall"]["overall"]
                category_scores[category]["precision_sum"] += result["context_precision"]["overall"]
                
                # Save intermediate results every 3 examples
                if (i + 1) % 3 == 0:
                    self.save_results(results[:i+1], output_path / f"intermediate_ragas_{i+1}.json")
                    
            except Exception as e:
                print(f"Error on example {i}: {e}")
                continue
        
        # Calculate final metrics
        total_examples = len(results)
        overall_score = sum(r["overall_score"] for r in results) / total_examples if results else 0
        
        # Calculate component averages
        avg_faithfulness = sum(r["faithfulness"]["overall"] for r in results) / total_examples
        avg_relevance = sum(r["answer_relevance"]["overall"] for r in results) / total_examples
        avg_recall = sum(r["context_recall"]["overall"] for r in results) / total_examples
        avg_precision = sum(r["context_precision"]["overall"] for r in results) / total_examples
        
        # Calculate category-wise averages
        category_averages = {}
        for category, stats in category_scores.items():
            category_averages[category] = {
                "overall": stats["score_sum"] / stats["total"],
                "faithfulness": stats["faithfulness_sum"] / stats["total"],
                "relevance": stats["relevance_sum"] / stats["total"],
                "recall": stats["recall_sum"] / stats["total"],
                "precision": stats["precision_sum"] / stats["total"]
            }
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_examples": total_examples,
            "overall_score": overall_score,
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
            "avg_recall": avg_recall,
            "avg_precision": avg_precision,
            "category_averages": category_averages,
            "category_stats": category_scores,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "ragas_results.json")
        
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
        print("RAGAS Evaluation Results")
        print("="*60)
        print(f"Model: {results['model_path']}")
        print(f"Total Examples: {results['total_examples']}")
        print(f"Overall Score: {results['overall_score']:.3f}/1.0")
        print("\nComponent Scores:")
        print(f"  Faithfulness: {results['avg_faithfulness']:.3f}/1.0")
        print(f"  Answer Relevance: {results['avg_relevance']:.3f}/1.0")
        print(f"  Context Recall: {results['avg_recall']:.3f}/1.0")
        print(f"  Context Precision: {results['avg_precision']:.3f}/1.0")
        
        print("\nCategory-wise Performance:")
        for category, scores in sorted(results['category_averages'].items(), 
                                      key=lambda x: x[1]['overall'], reverse=True):
            total = results['category_stats'][category]['total']
            print(f"  {category}: {scores['overall']:.3f} ({total} examples)")
            print(f"    Faithfulness: {scores['faithfulness']:.3f}")
            print(f"    Relevance: {scores['relevance']:.3f}")
            print(f"    Recall: {scores['recall']:.3f}")
            print(f"    Precision: {scores['precision']:.3f}")
        
        # Show examples
        sorted_results = sorted(results["detailed_results"], key=lambda x: x["overall_score"], reverse=True)
        best_result = sorted_results[0]
        worst_result = sorted_results[-1]
        
        print(f"\nBest Performing RAG Example:")
        print(f"  Category: {best_result['category']}")
        print(f"  Overall Score: {best_result['overall_score']:.3f}")
        print(f"  Question: {best_result['question']}")
        print(f"  Response: {best_result['generated_response'][:80]}...")
        
        print(f"\nLowest Performing RAG Example:")
        print(f"  Category: {worst_result['category']}")
        print(f"  Overall Score: {worst_result['overall_score']:.3f}")
        print(f"  Question: {worst_result['question']}")
        print(f"  Response: {worst_result['generated_response'][:80]}...")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on RAGAS metrics")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = RAGASEvaluator(args.model, args.device)
    dataset = evaluator.load_rag_dataset()
    results = evaluator.evaluate_dataset(dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
