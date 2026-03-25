"""
HellaSwag Evaluation Script for LLM Twin
Tests commonsense reasoning and situational completion
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

class HellaSwagEvaluator:
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
    
    def load_hellaswag_dataset(self, split: str = "validation") -> Dict:
        """Load HellaSwag dataset from HuggingFace"""
        print("Loading HellaSwag dataset...")
        dataset = load_dataset("Rowan/hellaswag", split=split)
        return dataset
    
    def format_question(self, context: str, endings: List[str]) -> str:
        """Format HellaSwag question with multiple choice endings"""
        prompt = f"""Context: {context}

Choose the most plausible ending:

A) {endings[0]}
B) {endings[1]}
C) {endings[2]}
D) {endings[3]}

The most logical ending is:"""
        return prompt
    
    def extract_answer(self, response: str) -> str:
        """Extract the answer letter from model response"""
        # Look for patterns like "A)", "B)", etc. or just the letter
        patterns = [
            r"([ABCD])\)",
            r"Answer:\s*([ABCD])",
            r"([ABCD])\.",
            r"The most logical ending is ([ABCD])",
            r"Correct answer: ([ABCD])"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).upper()
        
        # If no pattern matches, try to find any letter A-D
        letters = re.findall(r"[ABCD]", response)
        if letters:
            return letters[0].upper()
        
        return "UNKNOWN"
    
    def evaluate_single_question(self, context: str, endings: List[str], correct_label: int) -> Dict:
        """Evaluate model on a single HellaSwag question"""
        prompt = self.format_question(context, endings)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Create attention mask
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=50,  # Short responses for MCQ
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Extract answer
        predicted_answer = self.extract_answer(response)
        
        # Convert letter to index (A=0, B=1, C=2, D=3)
        answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        predicted_index = answer_map.get(predicted_answer, -1)
        
        # Check if correct
        is_correct = predicted_index == correct_label
        
        return {
            "context": context,
            "endings": endings,
            "correct_label": correct_label,
            "predicted_answer": predicted_answer,
            "predicted_index": predicted_index,
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
        domain_scores = {}
        
        print(f"Evaluating on {len(dataset)} HellaSwag questions...")
        
        for i, example in enumerate(tqdm(dataset)):
            try:
                result = self.evaluate_single_question(
                    example["ctx"],
                    example["endings"],
                    example["label"]
                )
                
                result["domain"] = example.get("activity_label", "Unknown")
                result["question_id"] = i
                results.append(result)
                
                if result["is_correct"]:
                    correct_count += 1
                
                # Update domain scores
                domain = result["domain"]
                if domain not in domain_scores:
                    domain_scores[domain] = {"correct": 0, "total": 0}
                domain_scores[domain]["total"] += 1
                if result["is_correct"]:
                    domain_scores[domain]["correct"] += 1
                
                # Save intermediate results every 50 questions
                if (i + 1) % 50 == 0:
                    self.save_results(results[:i+1], output_path / f"intermediate_hellaswag_{i+1}.json")
                    
            except Exception as e:
                print(f"Error on question {i}: {e}")
                continue
        
        # Calculate final metrics
        total_accuracy = correct_count / len(results) if results else 0
        
        # Calculate domain-wise accuracies
        domain_accuracies = {}
        for domain, scores in domain_scores.items():
            domain_accuracies[domain] = scores["correct"] / scores["total"]
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_questions": len(results),
            "correct_answers": correct_count,
            "overall_accuracy": total_accuracy,
            "domain_accuracies": domain_accuracies,
            "domain_scores": domain_scores,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "hellaswag_results.json")
        
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
        print("HellaSwag Evaluation Results")
        print("="*50)
        print(f"Model: {results['model_path']}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
        print("\nDomain-wise Accuracies:")
        for domain, accuracy in sorted(results['domain_accuracies'].items(), key=lambda x: x[1], reverse=True):
            total = results['domain_scores'][domain]['total']
            correct = results['domain_scores'][domain]['correct']
            print(f"  {domain}: {accuracy:.2%} ({correct}/{total})")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on HellaSwag dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = HellaSwagEvaluator(args.model, args.device)
    dataset = evaluator.load_hellaswag_dataset()
    results = evaluator.evaluate_dataset(dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
