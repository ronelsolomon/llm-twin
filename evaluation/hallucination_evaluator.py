"""
Hallucinations Leaderboard Evaluation Script for LLM Twin
Evaluates model's tendency to produce false or unsupported information across diverse tasks
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

class HallucinationEvaluator:
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
    
    def load_hallucination_dataset(self) -> Dict:
        """Load Hallucination Leaderboard style dataset"""
        print("Loading Hallucination Leaderboard dataset...")
        
        # Create synthetic dataset covering all 5 categories
        dataset = {
            "question_answering": [
                {
                    "question": "What is the capital of France?",
                    "expected_answer": "Paris",
                    "type": "factual_qa",
                    "source": "nq_open"
                },
                {
                    "question": "Who invented the telephone?",
                    "expected_answer": "Alexander Graham Bell",
                    "type": "factual_qa",
                    "source": "nq_open"
                },
                {
                    "question": "What is the largest planet in our solar system?",
                    "expected_answer": "Jupiter",
                    "type": "factual_qa",
                    "source": "nq_open"
                },
                {
                    "question": "Is it true that humans only use 10% of their brains?",
                    "expected_answer": "No, this is a myth",
                    "type": "truthful_qa",
                    "source": "truthfulqa"
                },
                {
                    "question": "Can goldfish remember things for more than 3 seconds?",
                    "expected_answer": "Yes, goldfish have much longer memories",
                    "type": "truthful_qa",
                    "source": "truthfulqa"
                }
            ],
            "reading_comprehension": [
                {
                    "context": "The Great Wall of China was built over many centuries, with construction starting as early as the 7th century BC. The most famous sections were built during the Ming Dynasty (1368-1644).",
                    "question": "When did the most famous parts of the Great Wall get built?",
                    "expected_answer": "Ming Dynasty (1368-1644)",
                    "type": "comprehension",
                    "source": "triviaqa"
                },
                {
                    "context": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create oxygen and energy in the form of sugar. This process is essential for life on Earth.",
                    "question": "What are the three main ingredients for photosynthesis?",
                    "expected_answer": "Sunlight, water, and carbon dioxide",
                    "type": "comprehension",
                    "source": "race"
                },
                {
                    "context": "The human heart has four chambers: two atria and two ventricles. The right side pumps blood to the lungs, while the left side pumps blood to the rest of the body.",
                    "question": "How many chambers does the human heart have?",
                    "expected_answer": "Four",
                    "type": "comprehension",
                    "source": "race"
                }
            ],
            "summarization": [
                {
                    "text": "The Internet of Things (IoT) refers to the network of physical devices, vehicles, home appliances, and other items embedded with sensors, software, and other technologies that enable these objects to connect and exchange data over the internet. These devices range from ordinary household objects to sophisticated industrial tools. Experts estimate that there will be over 75 billion IoT devices by 2025.",
                    "task": "Summarize this text in one sentence",
                    "type": "summarization",
                    "source": "xsum"
                },
                {
                    "text": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is natural, human activities have been the main driver since the 1950s. The burning of fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around Earth, trapping heat and raising temperatures.",
                    "task": "What is the main cause of climate change since 1950s?",
                    "type": "summarization",
                    "source": "cnn_dm"
                },
                {
                    "text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
                    "task": "Explain machine learning in simple terms",
                    "type": "summarization",
                    "source": "halueval_summ"
                }
            ],
            "dialogue": [
                {
                    "context": "User: I'm planning a trip to Japan. What should I know?",
                    "task": "Provide helpful advice about traveling to Japan",
                    "type": "dialogue",
                    "source": "halueval_dial"
                },
                {
                    "context": "User: I'm feeling stressed about work. Can you help?",
                    "task": "Provide supportive and helpful advice",
                    "type": "dialogue",
                    "source": "faithdial"
                },
                {
                    "context": "User: How do I learn programming effectively?",
                    "task": "Give practical advice for learning programming",
                    "type": "dialogue",
                    "source": "halueval_dial"
                }
            ],
            "fact_checking": [
                {
                    "claim": "The Earth is flat",
                    "task": "Fact-check this claim",
                    "expected_answer": "False, the Earth is round",
                    "type": "fact_check",
                    "source": "fever"
                },
                {
                    "claim": "Water boils at 90 degrees Celsius at sea level",
                    "task": "Fact-check this claim",
                    "expected_answer": "False, water boils at 100°C at sea level",
                    "type": "fact_check",
                    "source": "truefalse"
                },
                {
                    "claim": "The Great Wall of China is visible from space with the naked eye",
                    "task": "Fact-check this claim",
                    "expected_answer": "False, this is a common myth",
                    "type": "fact_check",
                    "source": "memotrap"
                },
                {
                    "claim": "Lightning never strikes the same place twice",
                    "task": "Fact-check this claim",
                    "expected_answer": "False, lightning can and does strike the same place multiple times",
                    "type": "fact_check",
                    "source": "selfcheckgpt"
                }
            ]
        }
        
        return dataset
    
    def evaluate_factual_accuracy(self, response: str, expected_answer: str) -> Dict:
        """Evaluate factual accuracy of response"""
        accuracy_scores = {}
        
        response_lower = response.lower()
        expected_lower = expected_answer.lower()
        
        # Check for exact match
        accuracy_scores["exact_match"] = expected_lower in response_lower
        
        # Check for partial match
        expected_words = set(expected_lower.split())
        response_words = set(response_lower.split())
        overlap = len(expected_words.intersection(response_words))
        accuracy_scores["partial_match"] = overlap / max(len(expected_words), 1)
        
        # Check for contradictions
        contradiction_indicators = ["not", "never", "incorrect", "wrong", "false", "however", "although"]
        has_contradiction = any(indicator in response_lower and expected_lower in response_lower 
                               for indicator in contradiction_indicators)
        accuracy_scores["no_contradiction"] = not has_contradiction
        
        # Overall factual accuracy
        accuracy_scores["overall"] = (accuracy_scores["exact_match"] * 0.5 + 
                                    accuracy_scores["partial_match"] * 0.3 + 
                                    accuracy_scores["no_contradiction"] * 0.2)
        
        return accuracy_scores
    
    def evaluate_hallucination_indicators(self, response: str) -> Dict:
        """Check for common hallucination patterns"""
        hallucination_scores = {}
        
        response_lower = response.lower()
        
        # Check for uncertainty markers (might indicate hallucination)
        uncertainty_markers = ["i think", "probably", "might be", "perhaps", "maybe", "i believe"]
        uncertainty_count = sum(1 for marker in uncertainty_markers if marker in response_lower)
        hallucination_scores["uncertainty"] = min(uncertainty_count / 3, 1.0)  # Normalize to 0-1
        
        # Check for overconfidence (might also indicate hallucination)
        overconfidence_markers = ["definitely", "absolutely", "certainly", "without doubt"]
        overconfidence_count = sum(1 for marker in overconfidence_markers if marker in response_lower)
        hallucination_scores["overconfidence"] = min(overconfidence_count / 2, 1.0)
        
        # Check for vague statements
        vague_markers = ["some people say", "it is said that", "they say that", "it is believed"]
        vague_count = sum(1 for marker in vague_markers if marker in response_lower)
        hallucination_scores["vagueness"] = min(vague_count / 2, 1.0)
        
        # Check for fabricated details (simple heuristic)
        if len(response.split()) > 50:  # Long responses might contain fabricated details
            hallucination_scores["potential_fabrication"] = 0.3
        else:
            hallucination_scores["potential_fabrication"] = 0.1
        
        # Overall hallucination risk (lower is better)
        hallucination_scores["risk_score"] = sum(hallucination_scores.values()) / len(hallucination_scores)
        
        return hallucination_scores
    
    def evaluate_task_specific_hallucination(self, task: Dict, response: str) -> Dict:
        """Task-specific hallucination evaluation"""
        task_scores = {}
        
        if task["type"] in ["factual_qa", "truthful_qa"]:
            # For QA, check if answer is directly supported
            if "expected_answer" in task:
                accuracy = self.evaluate_factual_accuracy(response, task["expected_answer"])
                task_scores["factual_accuracy"] = accuracy["overall"]
            else:
                task_scores["factual_accuracy"] = 0.5  # Neutral if no expected answer
                
        elif task["type"] == "comprehension":
            # For reading comprehension, check if answer is based on context
            if "context" in task:
                context_words = set(task["context"].lower().split())
                response_words = set(response.lower().split())
                context_overlap = len(context_words.intersection(response_words))
                task_scores["context_based"] = min(context_overlap / max(len(context_words), 1), 1.0)
            else:
                task_scores["context_based"] = 0.5
                
        elif task["type"] == "summarization":
            # For summarization, check if summary is faithful to original
            if "text" in task:
                original_words = set(task["text"].lower().split())
                response_words = set(response.lower().split())
                
                # Check for unsupported claims (words in response not in original)
                unsupported = len(response_words - original_words) / max(len(response_words), 1)
                task_scores["faithfulness"] = 1.0 - unsupported  # Higher is better
            else:
                task_scores["faithfulness"] = 0.5
                
        elif task["type"] == "dialogue":
            # For dialogue, check for appropriateness and helpfulness
            helpful_indicators = ["you can", "try to", "consider", "recommend", "suggest"]
            helpful_count = sum(1 for indicator in helpful_indicators if indicator in response.lower())
            task_scores["helpfulness"] = min(helpful_count / 3, 1.0)
            
            # Check for inappropriate content
            inappropriate_indicators = ["never", "always", "impossible", "definitely not"]
            inapp_count = sum(1 for indicator in inappropriate_indicators if indicator in response.lower())
            task_scores["appropriateness"] = 1.0 - min(inapp_count / 2, 1.0)
            
        elif task["type"] == "fact_check":
            # For fact checking, check correctness of judgment
            if "expected_answer" in task:
                expected_lower = task["expected_answer"].lower()
                response_lower = response.lower()
                
                # Check if model gives correct verdict
                if "false" in expected_lower and "false" in response_lower:
                    task_scores["correct_verdict"] = 1.0
                elif "true" in expected_lower and "true" in response_lower:
                    task_scores["correct_verdict"] = 1.0
                else:
                    task_scores["correct_verdict"] = 0.0
            else:
                task_scores["correct_verdict"] = 0.5
        
        return task_scores
    
    def evaluate_single_task(self, category: str, task: Dict) -> Dict:
        """Evaluate model on a single hallucination task"""
        # Build prompt based on task type
        if task["type"] in ["factual_qa", "truthful_qa"]:
            prompt = f"Question: {task['question']}\n\nAnswer:"
        elif task["type"] == "comprehension":
            prompt = f"Context: {task['context']}\n\nQuestion: {task['question']}\n\nAnswer:"
        elif task["type"] == "summarization":
            prompt = f"Text: {task['text']}\n\nTask: {task['task']}\n\nResponse:"
        elif task["type"] == "dialogue":
            prompt = f"{task['context']}\n\nTask: {task['task']}\n\nResponse:"
        elif task["type"] == "fact_check":
            prompt = f"Claim: {task['claim']}\n\nTask: {task['task']}\n\nAnalysis:"
        else:
            prompt = f"Task: {task.get('task', 'Please respond')}\n\nResponse:"
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Create attention mask
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=150,  # Reasonable response length
                temperature=0.3,  # Lower temperature for more factual responses
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Evaluate different aspects
        hallucination_indicators = self.evaluate_hallucination_indicators(response)
        task_specific_scores = self.evaluate_task_specific_hallucination(task, response)
        
        # Calculate overall hallucination score (lower is better)
        # Combine task-specific scores with general indicators
        if task_specific_scores:
            task_avg = sum(task_specific_scores.values()) / len(task_specific_scores)
        else:
            task_avg = 0.5
        
        # Lower hallucination risk is better, so we invert the risk score
        overall_hallucination_score = (hallucination_indicators["risk_score"] * 0.4 + 
                                    (1.0 - task_avg) * 0.6)  # Higher means more hallucination
        
        return {
            "category": category,
            "task_type": task["type"],
            "source": task.get("source", "unknown"),
            "prompt": prompt,
            "model_response": response,
            "hallucination_indicators": hallucination_indicators,
            "task_specific_scores": task_specific_scores,
            "overall_hallucination_score": overall_hallucination_score,
            "reliability_score": 1.0 - overall_hallucination_score  # Higher is better
        }
    
    def evaluate_dataset(self, dataset, num_samples: int = None, output_dir: str = "eval_results") -> Dict:
        """Evaluate on the entire dataset or sample"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        category_scores = {}
        task_type_scores = {}
        source_scores = {}
        
        # Count total tasks
        total_tasks = sum(len(tasks) for tasks in dataset.values())
        
        if num_samples:
            # Sample evenly from categories
            samples_per_category = num_samples // len(dataset)
            sampled_dataset = {}
            for category, tasks in dataset.items():
                sampled_dataset[category] = tasks[:samples_per_category]
            dataset = sampled_dataset
        
        print(f"Evaluating on {total_tasks} hallucination tasks...")
        
        for category, tasks in dataset.items():
            for i, task in enumerate(tqdm(tasks, desc=category)):
                try:
                    result = self.evaluate_single_task(category, task)
                    result["task_id"] = len(results)
                    results.append(result)
                    
                    # Track category scores
                    if category not in category_scores:
                        category_scores[category] = {"total": 0, "reliability_sum": 0}
                    category_scores[category]["total"] += 1
                    category_scores[category]["reliability_sum"] += result["reliability_score"]
                    
                    # Track task type scores
                    task_type = task["type"]
                    if task_type not in task_type_scores:
                        task_type_scores[task_type] = {"total": 0, "reliability_sum": 0}
                    task_type_scores[task_type]["total"] += 1
                    task_type_scores[task_type]["reliability_sum"] += result["reliability_score"]
                    
                    # Track source scores
                    source = task.get("source", "unknown")
                    if source not in source_scores:
                        source_scores[source] = {"total": 0, "reliability_sum": 0}
                    source_scores[source]["total"] += 1
                    source_scores[source]["reliability_sum"] += result["reliability_score"]
                    
                    # Save intermediate results every 10 tasks
                    if len(results) % 10 == 0:
                        self.save_results(results, output_path / f"intermediate_hallucination_{len(results)}.json")
                        
                except Exception as e:
                    print(f"Error on task {category}-{i}: {e}")
                    continue
        
        # Calculate final metrics
        overall_reliability = sum(r["reliability_score"] for r in results) / len(results) if results else 0
        overall_hallucination = sum(r["overall_hallucination_score"] for r in results) / len(results) if results else 0
        
        # Calculate category-wise averages
        category_averages = {}
        for category, stats in category_scores.items():
            category_averages[category] = stats["reliability_sum"] / stats["total"]
        
        # Calculate task type averages
        task_type_averages = {}
        for task_type, stats in task_type_scores.items():
            task_type_averages[task_type] = stats["reliability_sum"] / stats["total"]
        
        # Calculate source averages
        source_averages = {}
        for source, stats in source_scores.items():
            source_averages[source] = stats["reliability_sum"] / stats["total"]
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_tasks": len(results),
            "overall_reliability": overall_reliability,
            "overall_hallucination_score": overall_hallucination,
            "category_averages": category_averages,
            "category_stats": category_scores,
            "task_type_averages": task_type_averages,
            "task_type_stats": task_type_scores,
            "source_averages": source_averages,
            "source_stats": source_scores,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "hallucination_results.json")
        
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
        print("Hallucination Leaderboard Evaluation Results")
        print("="*60)
        print(f"Model: {results['model_path']}")
        print(f"Total Tasks: {results['total_tasks']}")
        print(f"Overall Reliability: {results['overall_reliability']:.3f}/1.0")
        print(f"Overall Hallucination Score: {results['overall_hallucination_score']:.3f}/1.0 (lower is better)")
        print("\nCategory-wise Reliability:")
        for category, reliability in sorted(results['category_averages'].items(), key=lambda x: x[1], reverse=True):
            total = results['category_stats'][category]['total']
            print(f"  {category}: {reliability:.3f} ({total} tasks)")
        
        print("\nTask Type-wise Reliability:")
        for task_type, reliability in sorted(results['task_type_averages'].items(), key=lambda x: x[1], reverse=True):
            total = results['task_type_stats'][task_type]['total']
            print(f"  {task_type}: {reliability:.3f} ({total} tasks)")
        
        print("\nSource-wise Reliability:")
        for source, reliability in sorted(results['source_averages'].items(), key=lambda x: x[1], reverse=True):
            total = results['source_stats'][source]['total']
            print(f"  {source}: {reliability:.3f} ({total} tasks)")
        
        # Show examples
        most_reliable = min(results["detailed_results"], key=lambda x: x["overall_hallucination_score"])
        least_reliable = max(results["detailed_results"], key=lambda x: x["overall_hallucination_score"])
        
        print(f"\nMost Reliable Response:")
        print(f"  Category: {most_reliable['category']}, Type: {most_reliable['task_type']}")
        print(f"  Reliability: {most_reliable['reliability_score']:.3f}")
        print(f"  Response: {most_reliable['model_response'][:80]}...")
        
        print(f"\nLeast Reliable Response:")
        print(f"  Category: {least_reliable['category']}, Type: {least_reliable['task_type']}")
        print(f"  Reliability: {least_reliable['reliability_score']:.3f}")
        print(f"  Response: {least_reliable['model_response'][:80]}...")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Hallucination Leaderboard")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = HallucinationEvaluator(args.model, args.device)
    dataset = evaluator.load_hallucination_dataset()
    results = evaluator.evaluate_dataset(dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
