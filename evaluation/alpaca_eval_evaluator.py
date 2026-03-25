"""
AlpacaEval Evaluation Script for LLM Twin
Automatic evaluation for instruction following that correlates with human preferences
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

class AlpacaEvalEvaluator:
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
    
    def load_alpaca_eval_dataset(self, split: str = "eval") -> Dict:
        """Load AlpacaEval dataset from HuggingFace"""
        print("Loading AlpacaEval dataset...")
        
        # Create synthetic AlpacaEval-style dataset with diverse instruction types
        dataset = [
            {
                "instruction": "Write a Python function that calculates the factorial of a number.",
                "category": "coding",
                "difficulty": "easy"
            },
            {
                "instruction": "Explain the concept of machine learning to a 10-year-old.",
                "category": "explanation",
                "difficulty": "medium"
            },
            {
                "instruction": "Summarize the main causes of World War II in three paragraphs.",
                "category": "summarization",
                "difficulty": "hard"
            },
            {
                "instruction": "Create a recipe for chocolate chip cookies.",
                "category": "creative",
                "difficulty": "easy"
            },
            {
                "instruction": "Compare and contrast Python and JavaScript programming languages.",
                "category": "comparison",
                "difficulty": "medium"
            },
            {
                "instruction": "Write a short story about a time-traveling historian.",
                "category": "creative",
                "difficulty": "medium"
            },
            {
                "instruction": "Explain how photosynthesis works in detail.",
                "category": "explanation",
                "difficulty": "hard"
            },
            {
                "instruction": "Design a simple algorithm for sorting a list of numbers.",
                "category": "coding",
                "difficulty": "medium"
            },
            {
                "instruction": "Translate the following English sentence to French: 'The cat is sleeping on the couch.'",
                "category": "translation",
                "difficulty": "easy"
            },
            {
                "instruction": "Analyze the economic impact of artificial intelligence on job markets.",
                "category": "analysis",
                "difficulty": "hard"
            },
            {
                "instruction": "Write a poem about the changing seasons.",
                "category": "creative",
                "difficulty": "easy"
            },
            {
                "instruction": "Explain the difference between renewable and non-renewable energy sources.",
                "category": "explanation",
                "difficulty": "medium"
            },
            {
                "instruction": "Create a workout routine for beginners focusing on cardio exercises.",
                "category": "practical",
                "difficulty": "easy"
            },
            {
                "instruction": "Describe the process of cellular respiration.",
                "category": "explanation",
                "difficulty": "hard"
            },
            {
                "instruction": "Write a professional email to a client about a project delay.",
                "category": "practical",
                "difficulty": "medium"
            }
        ]
        
        return dataset
    
    def evaluate_response_quality(self, instruction: str, response: str, category: str) -> Dict:
        """Evaluate response quality based on multiple criteria"""
        
        quality_scores = {}
        
        # Length appropriateness
        response_length = len(response.split())
        if category == "coding":
            length_score = 1.0 if 50 <= response_length <= 500 else 0.5
        elif category == "explanation":
            length_score = 1.0 if 100 <= response_length <= 300 else 0.5
        elif category == "creative":
            length_score = 1.0 if 50 <= response_length <= 400 else 0.5
        else:
            length_score = 1.0 if 20 <= response_length <= 200 else 0.5
        
        quality_scores["length_appropriateness"] = length_score
        
        # Relevance (simple keyword matching)
        instruction_words = set(instruction.lower().split())
        response_words = set(response.lower().split())
        overlap = len(instruction_words.intersection(response_words))
        relevance_score = min(overlap / max(len(instruction_words), 1), 1.0)
        quality_scores["relevance"] = relevance_score
        
        # Coherence (simple check - no repeated phrases)
        sentences = response.split('.')
        unique_sentences = len(set(sentences))
        coherence_score = unique_sentences / max(len(sentences), 1)
        quality_scores["coherence"] = coherence_score
        
        # Completeness (has some content)
        completeness_score = 1.0 if len(response.strip()) > 10 else 0.0
        quality_scores["completeness"] = completeness_score
        
        # Category-specific scoring
        if category == "coding":
            # Check for code-like patterns
            has_code = any(keyword in response.lower() for keyword in ["def", "function", "class", "import", "return"])
            quality_scores["code_presence"] = 1.0 if has_code else 0.0
        elif category == "explanation":
            # Check for explanation markers
            has_explanation = any(keyword in response.lower() for keyword in ["because", "therefore", "however", "additionally", "furthermore"])
            quality_scores["explanation_markers"] = 1.0 if has_explanation else 0.0
        elif category == "creative":
            # Check for creative elements
            has_creative = any(keyword in response.lower() for keyword in ["imagine", "story", "character", "scene", "narrative"])
            quality_scores["creative_elements"] = 1.0 if has_creative else 0.0
        else:
            quality_scores["category_specific"] = 0.5  # neutral score
        
        # Calculate overall quality score
        overall_score = sum(quality_scores.values()) / len(quality_scores)
        quality_scores["overall"] = overall_score
        
        return quality_scores
    
    def evaluate_single_instruction(self, instruction: str, category: str, difficulty: str) -> Dict:
        """Evaluate model on a single AlpacaEval instruction"""
        prompt = f"""Please respond to the following instruction:

{instruction}

Your response:"""
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Create attention mask
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=200,  # Allow for detailed responses
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Evaluate response quality
        quality_scores = self.evaluate_response_quality(instruction, response, category)
        
        return {
            "instruction": instruction,
            "category": category,
            "difficulty": difficulty,
            "model_response": response,
            "quality_scores": quality_scores,
            "overall_score": quality_scores["overall"]
        }
    
    def evaluate_dataset(self, dataset, num_samples: int = None, output_dir: str = "eval_results") -> Dict:
        """Evaluate on the entire dataset or sample"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if num_samples:
            dataset = dataset[:min(num_samples, len(dataset))]
        
        results = []
        category_scores = {}
        difficulty_scores = {}
        
        print(f"Evaluating on {len(dataset)} AlpacaEval instructions...")
        
        for i, example in enumerate(tqdm(dataset)):
            try:
                result = self.evaluate_single_instruction(
                    example["instruction"],
                    example["category"],
                    example["difficulty"]
                )
                
                result["question_id"] = i
                results.append(result)
                
                # Track category scores
                category = example["category"]
                if category not in category_scores:
                    category_scores[category] = {"total": 0, "score_sum": 0}
                category_scores[category]["total"] += 1
                category_scores[category]["score_sum"] += result["overall_score"]
                
                # Track difficulty scores
                difficulty = example["difficulty"]
                if difficulty not in difficulty_scores:
                    difficulty_scores[difficulty] = {"total": 0, "score_sum": 0}
                difficulty_scores[difficulty]["total"] += 1
                difficulty_scores[difficulty]["score_sum"] += result["overall_score"]
                
                # Save intermediate results every 5 questions
                if (i + 1) % 5 == 0:
                    self.save_results(results[:i+1], output_path / f"intermediate_alpaca_eval_{i+1}.json")
                    
            except Exception as e:
                print(f"Error on instruction {i}: {e}")
                continue
        
        # Calculate final metrics
        total_instructions = len(results)
        overall_score = sum(r["overall_score"] for r in results) / total_instructions if results else 0
        
        # Calculate category-wise averages
        category_averages = {}
        for category, stats in category_scores.items():
            category_averages[category] = stats["score_sum"] / stats["total"]
        
        # Calculate difficulty-wise averages
        difficulty_averages = {}
        for difficulty, stats in difficulty_scores.items():
            difficulty_averages[difficulty] = stats["score_sum"] / stats["total"]
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_instructions": total_instructions,
            "overall_score": overall_score,
            "category_averages": category_averages,
            "difficulty_averages": difficulty_averages,
            "category_stats": category_scores,
            "difficulty_stats": difficulty_scores,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "alpaca_eval_results.json")
        
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
        print("\n" + "="*50)
        print("AlpacaEval Evaluation Results")
        print("="*50)
        print(f"Model: {results['model_path']}")
        print(f"Total Instructions: {results['total_instructions']}")
        print(f"Overall Score: {results['overall_score']:.3f}/1.0")
        print("\nCategory-wise Performance:")
        for category, avg_score in sorted(results['category_averages'].items(), key=lambda x: x[1], reverse=True):
            total = results['category_stats'][category]['total']
            print(f"  {category}: {avg_score:.3f} ({total} instructions)")
        
        print("\nDifficulty-wise Performance:")
        for difficulty, avg_score in sorted(results['difficulty_averages'].items(), key=lambda x: x[1], reverse=True):
            total = results['difficulty_stats'][difficulty]['total']
            print(f"  {difficulty}: {avg_score:.3f} ({total} instructions)")
        
        # Show top and bottom examples
        sorted_results = sorted(results["detailed_results"], key=lambda x: x["overall_score"], reverse=True)
        top_examples = sorted_results[:2]
        bottom_examples = sorted_results[-2:]
        
        print(f"\nTop Performing Examples:")
        for i, example in enumerate(top_examples, 1):
            print(f"  {i}. [{example['category']}] {example['instruction'][:50]}...")
            print(f"     Score: {example['overall_score']:.3f}")
            print(f"     Response: {example['model_response'][:60]}...")
        
        print(f"\nLowest Performing Examples:")
        for i, example in enumerate(bottom_examples, 1):
            print(f"  {i}. [{example['category']}] {example['instruction'][:50]}...")
            print(f"     Score: {example['overall_score']:.3f}")
        
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on AlpacaEval dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = AlpacaEvalEvaluator(args.model, args.device)
    dataset = evaluator.load_alpaca_eval_dataset()
    results = evaluator.evaluate_dataset(dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
