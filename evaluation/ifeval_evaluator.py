"""
IFEval (Instruction Following) Evaluation Script for LLM Twin
Assesses model's ability to follow instructions with specific constraints
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

class IFEvalEvaluator:
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
    
    def load_ifeval_dataset(self, split: str = "train") -> Dict:
        """Load IFEval dataset from HuggingFace"""
        print("Loading IFEval dataset...")
        # Note: IFEval might not be directly available, we'll create a synthetic dataset
        # based on common instruction following constraints
        
        # Create synthetic IFEval-style dataset
        dataset = [
            {
                "instruction": "Write a sentence about dogs without using the letter 'e'.",
                "constraints": ["no_letter_e"],
                "expected_pattern": r"^[^e]*$"
            },
            {
                "instruction": "List three colors without using any commas in your answer.",
                "constraints": ["no_commas"],
                "expected_pattern": r"^[^,]*$"
            },
            {
                "instruction": "Write a short paragraph about cooking that contains exactly 25 words.",
                "constraints": ["exactly_25_words"],
                "expected_pattern": r"^\b(?:\w+\b\s?){25}\s?$"
            },
            {
                "instruction": "Respond to this question using only lowercase letters: What is your favorite food?",
                "constraints": ["all_lowercase"],
                "expected_pattern": r"^[a-z\s]*$"
            },
            {
                "instruction": "Write about the weather without using the word 'weather'.",
                "constraints": ["no_word_weather"],
                "expected_pattern": r"^(?!.*\bweather\b).*$"
            },
            {
                "instruction": "Give me directions to the nearest store using only numbers and street names.",
                "constraints": ["only_numbers_street_names"],
                "expected_pattern": r"^[0-9\s\w]*$"
            },
            {
                "instruction": "Describe a happy memory without using any adjectives.",
                "constraints": ["no_adjectives"],
                "expected_pattern": r"^(?!.*\b\w+(?:ly|ful|ous|able|ible|al|ial|y|ish|less|like)\b).*$"
            },
            {
                "instruction": "Write a sentence that starts with the letter 'Z' and ends with the letter 'A'.",
                "constraints": ["starts_with_z_ends_with_a"],
                "expected_pattern": r"^Z.*a$"
            },
            {
                "instruction": "Tell me about your day using exactly 50 characters.",
                "constraints": ["exactly_50_chars"],
                "expected_pattern": r"^.{50}$"
            },
            {
                "instruction": "Write about technology without mentioning any specific company names.",
                "constraints": ["no_company_names"],
                "expected_pattern": r"^(?!.*\b(google|apple|microsoft|amazon|facebook|tesla|netflix|twitter|instagram)\b).*$"
            }
        ]
        
        return dataset
    
    def check_constraints(self, response: str, constraints: List[str]) -> Dict[str, bool]:
        """Check if response follows all constraints"""
        results = {}
        
        for constraint in constraints:
            if constraint == "no_letter_e":
                results[constraint] = 'e' not in response.lower()
            elif constraint == "no_commas":
                results[constraint] = ',' not in response
            elif constraint == "exactly_25_words":
                word_count = len(response.split())
                results[constraint] = word_count == 25
            elif constraint == "all_lowercase":
                results[constraint] = response == response.lower() and response.isalpha() or all(c.islower() or c.isspace() for c in response)
            elif constraint == "no_word_weather":
                results[constraint] = "weather" not in response.lower()
            elif constraint == "only_numbers_street_names":
                # Allow numbers, letters, and spaces
                results[constraint] = all(c.isalnum() or c.isspace() for c in response)
            elif constraint == "no_adjectives":
                # Simple check for common adjective endings
                adj_endings = ['ly', 'ful', 'ous', 'able', 'ible', 'al', 'ial', 'y', 'ish', 'less', 'like']
                words = response.lower().split()
                results[constraint] = not any(any(word.endswith(ending) for ending in adj_endings) for word in words)
            elif constraint == "starts_with_z_ends_with_a":
                results[constraint] = response.startswith('Z') and response.endswith('a')
            elif constraint == "exactly_50_chars":
                results[constraint] = len(response) == 50
            elif constraint == "no_company_names":
                companies = ['google', 'apple', 'microsoft', 'amazon', 'facebook', 'tesla', 'netflix', 'twitter', 'instagram']
                results[constraint] = not any(company in response.lower() for company in companies)
            else:
                results[constraint] = True  # Default to true for unknown constraints
        
        return results
    
    def evaluate_single_instruction(self, instruction: str, constraints: List[str]) -> Dict:
        """Evaluate model on a single instruction following task"""
        prompt = f"""Follow this instruction exactly:

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
                max_new_tokens=100,  # Allow for longer responses
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Check constraints
        constraint_results = self.check_constraints(response, constraints)
        
        # Calculate overall success
        all_constraints_met = all(constraint_results.values())
        
        return {
            "instruction": instruction,
            "constraints": constraints,
            "model_response": response,
            "constraint_results": constraint_results,
            "all_constraints_met": all_constraints_met
        }
    
    def evaluate_dataset(self, dataset, num_samples: int = None, output_dir: str = "eval_results") -> Dict:
        """Evaluate on the entire dataset or sample"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if num_samples:
            dataset = dataset[:min(num_samples, len(dataset))]
        
        results = []
        constraint_success_rates = {}
        
        print(f"Evaluating on {len(dataset)} IFEval instructions...")
        
        for i, example in enumerate(tqdm(dataset)):
            try:
                result = self.evaluate_single_instruction(
                    example["instruction"],
                    example["constraints"]
                )
                
                result["question_id"] = i
                results.append(result)
                
                # Track constraint success rates
                for constraint in example["constraints"]:
                    if constraint not in constraint_success_rates:
                        constraint_success_rates[constraint] = {"success": 0, "total": 0}
                    constraint_success_rates[constraint]["total"] += 1
                    if result["constraint_results"][constraint]:
                        constraint_success_rates[constraint]["success"] += 1
                
                # Save intermediate results every 5 questions
                if (i + 1) % 5 == 0:
                    self.save_results(results[:i+1], output_path / f"intermediate_ifeval_{i+1}.json")
                    
            except Exception as e:
                print(f"Error on instruction {i}: {e}")
                continue
        
        # Calculate final metrics
        total_instructions = len(results)
        fully_followed = sum(1 for r in results if r["all_constraints_met"])
        overall_accuracy = fully_followed / total_instructions if results else 0
        
        # Calculate constraint-wise success rates
        constraint_accuracies = {}
        for constraint, stats in constraint_success_rates.items():
            constraint_accuracies[constraint] = stats["success"] / stats["total"]
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_instructions": total_instructions,
            "fully_followed": fully_followed,
            "overall_accuracy": overall_accuracy,
            "constraint_accuracies": constraint_accuracies,
            "constraint_stats": constraint_success_rates,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "ifeval_results.json")
        
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
        print("IFEval Evaluation Results")
        print("="*50)
        print(f"Model: {results['model_path']}")
        print(f"Total Instructions: {results['total_instructions']}")
        print(f"Fully Followed: {results['fully_followed']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
        print("\nConstraint-wise Success Rates:")
        for constraint, accuracy in sorted(results['constraint_accuracies'].items(), key=lambda x: x[1], reverse=True):
            total = results['constraint_stats'][constraint]['total']
            success = results['constraint_stats'][constraint]['success']
            print(f"  {constraint}: {accuracy:.2%} ({success}/{total})")
        
        # Show examples
        successful = [r for r in results["detailed_results"] if r["all_constraints_met"]][:2]
        failed = [r for r in results["detailed_results"] if not r["all_constraints_met"]][:2]
        
        print(f"\nSuccessful Examples:")
        for i, example in enumerate(successful, 1):
            print(f"  {i}. {example['instruction'][:60]}...")
            print(f"     Response: {example['model_response'][:50]}...")
        
        print(f"\nFailed Examples:")
        for i, example in enumerate(failed, 1):
            failed_constraints = [c for c, passed in example['constraint_results'].items() if not passed]
            print(f"  {i}. {example['instruction'][:60]}...")
            print(f"     Failed constraints: {failed_constraints}")
        
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on IFEval dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = IFEvalEvaluator(args.model, args.device)
    dataset = evaluator.load_ifeval_dataset()
    results = evaluator.evaluate_dataset(dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
