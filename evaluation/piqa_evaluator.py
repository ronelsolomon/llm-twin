"""
PIQA (Physical Interaction: Question Answering) Evaluation Script for LLM Twin
Measures physical common sense understanding through everyday physical interactions
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

class PIQAEvaluator:
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
    
    def load_piqa_dataset(self, split: str = "validation") -> Dict:
        """Load PIQA dataset from HuggingFace"""
        print("Loading PIQA dataset...")
        dataset = load_dataset("piqa", split=split)
        return dataset
    
    def format_question(self, goal: str, sol1: str, sol2: str) -> str:
        """Format PIQA question with physical interaction solutions"""
        prompt = f"""Goal: {goal}

Which solution is physically correct?

Solution 1: {sol1}
Solution 2: {sol2}

The physically correct solution is:"""
        return prompt
    
    def extract_answer(self, response: str) -> str:
        """Extract the answer number from model response"""
        # Look for patterns like "Solution 1", "Solution 2", "1", "2"
        patterns = [
            r"Solution (\d)",
            r"solution (\d)",
            r"Answer: (\d)",
            r"The physically correct solution is (\d)",
            r"Correct solution: (\d)",
            r"^(\d)",
            r"(\d)[.!?]"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1)
        
        # If no pattern matches, try to find 1 or 2
        numbers = re.findall(r"[12]", response)
        if numbers:
            return numbers[0]
        
        return "UNKNOWN"
    
    def evaluate_single_question(self, goal: str, sol1: str, sol2: str, correct_label: int) -> Dict:
        """Evaluate model on a single PIQA question"""
        prompt = self.format_question(goal, sol1, sol2)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Create attention mask
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=20,  # Short responses for MCQ
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Extract answer
        predicted_answer_str = self.extract_answer(response)
        predicted_answer = int(predicted_answer_str) if predicted_answer_str.isdigit() else -1
        
        # Check if correct (convert 0-based to 1-based)
        predicted_label = predicted_answer - 1 if predicted_answer > 0 else -1
        is_correct = predicted_label == correct_label
        
        return {
            "goal": goal,
            "sol1": sol1,
            "sol2": sol2,
            "correct_label": correct_label,
            "predicted_answer_str": predicted_answer_str,
            "predicted_label": predicted_label,
            "model_response": response,
            "is_correct": is_correct
        }
    
    def evaluate_dataset(self, dataset, num_samples: int = None, output_dir: str = "eval_results") -> Dict:
        """Evaluate on the entire dataset or sample"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if num_samples:
            dataset = dataset.select(range(min(num_samples, len(dataset))))
        
        results = []
        correct_count = 0
        category_scores = {}
        
        print(f"Evaluating on {len(dataset)} PIQA questions...")
        
        for i, example in enumerate(tqdm(dataset)):
            try:
                result = self.evaluate_single_question(
                    example["goal"],
                    example["sol1"],
                    example["sol2"],
                    example["label"]
                )
                
                # Extract category from goal (simple heuristic)
                goal_lower = result["goal"].lower()
                if any(word in goal_lower for word in ["cook", "food", "kitchen", "eat"]):
                    category = "cooking"
                elif any(word in goal_lower for word in ["clean", "wash", "dust"]):
                    category = "cleaning"
                elif any(word in goal_lower for word in ["tool", "fix", "repair"]):
                    category = "tools"
                elif any(word in goal_lower for word in ["play", "game", "toy"]):
                    category = "play"
                else:
                    category = "general"
                
                result["category"] = category
                result["question_id"] = i
                results.append(result)
                
                if result["is_correct"]:
                    correct_count += 1
                
                # Update category scores
                if category not in category_scores:
                    category_scores[category] = {"correct": 0, "total": 0}
                category_scores[category]["total"] += 1
                if result["is_correct"]:
                    category_scores[category]["correct"] += 1
                
                # Save intermediate results every 100 questions
                if (i + 1) % 100 == 0:
                    self.save_results(results[:i+1], output_path / f"intermediate_piqa_{i+1}.json")
                    
            except Exception as e:
                print(f"Error on question {i}: {e}")
                continue
        
        # Calculate final metrics
        total_accuracy = correct_count / len(results) if results else 0
        
        # Calculate category-wise accuracies
        category_accuracies = {}
        for category, scores in category_scores.items():
            category_accuracies[category] = scores["correct"] / scores["total"]
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_questions": len(results),
            "correct_answers": correct_count,
            "overall_accuracy": total_accuracy,
            "category_accuracies": category_accuracies,
            "category_scores": category_scores,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "piqa_results.json")
        
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
        print("PIQA Evaluation Results")
        print("="*50)
        print(f"Model: {results['model_path']}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
        print("\nCategory-wise Accuracies:")
        for category, accuracy in sorted(results['category_accuracies'].items(), key=lambda x: x[1], reverse=True):
            total = results['category_scores'][category]['total']
            correct = results['category_scores'][category]['correct']
            print(f"  {category}: {accuracy:.2%} ({correct}/{total})")
        
        # Show some examples
        correct_examples = [r for r in results["detailed_results"] if r["is_correct"]][:2]
        incorrect_examples = [r for r in results["detailed_results"] if not r["is_correct"]][:2]
        
        print(f"\nSample Correct Answers:")
        for i, example in enumerate(correct_examples, 1):
            print(f"  {i}. Goal: {example['goal'][:60]}...")
            print(f"     Correct: Solution {example['correct_label'] + 1}, Predicted: {example['predicted_label'] + 1 if example['predicted_label'] >= 0 else 'Unknown'}")
        
        print(f"\nSample Incorrect Answers:")
        for i, example in enumerate(incorrect_examples, 1):
            print(f"  {i}. Goal: {example['goal'][:60]}...")
            print(f"     Correct: Solution {example['correct_label'] + 1}, Predicted: {example['predicted_label'] + 1 if example['predicted_label'] >= 0 else 'Unknown'}")
        
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on PIQA dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = PIQAEvaluator(args.model, args.device)
    dataset = evaluator.load_piqa_dataset()
    results = evaluator.evaluate_dataset(dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
