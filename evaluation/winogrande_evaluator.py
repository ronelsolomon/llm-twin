"""
Winogrande Evaluation Script for LLM Twin
Assesses common sense reasoning through pronoun resolution
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

class WinograndeEvaluator:
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
    
    def load_winogrande_dataset(self, split: str = "validation") -> Dict:
        """Load Winogrande dataset from HuggingFace"""
        print("Loading Winogrande dataset...")
        dataset = load_dataset("winogrande", "winogrande_xl", split=split)
        return dataset
    
    def format_question(self, sentence: str, option1: str, option2: str) -> str:
        """Format Winogrande question with pronoun resolution options"""
        prompt = f"""Complete the sentence by choosing the correct option:

Sentence: {sentence}

Option 1: {option1}
Option 2: {option2}

Which option correctly resolves the pronoun?"""
        return prompt
    
    def extract_answer(self, response: str) -> str:
        """Extract the answer number from model response"""
        # Look for patterns like "Option 1", "Option 2", "1", "2"
        patterns = [
            r"Option (\d)",
            r"option (\d)",
            r"Answer: (\d)",
            r"The correct option is (\d)",
            r"Correct answer: (\d)",
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
    
    def evaluate_single_question(self, sentence: str, option1: str, option2: str, correct_answer: int) -> Dict:
        """Evaluate model on a single Winogrande question"""
        prompt = self.format_question(sentence, option1, option2)
        
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
        
        # Check if correct
        is_correct = predicted_answer == correct_answer
        
        return {
            "sentence": sentence,
            "option1": option1,
            "option2": option2,
            "correct_answer": correct_answer,
            "predicted_answer_str": predicted_answer_str,
            "predicted_answer": predicted_answer,
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
        
        print(f"Evaluating on {len(dataset)} Winogrande questions...")
        
        for i, example in enumerate(tqdm(dataset)):
            try:
                result = self.evaluate_single_question(
                    example["sentence"],
                    example["option1"],
                    example["option2"],
                    example["answer"]
                )
                
                result["question_id"] = i
                results.append(result)
                
                if result["is_correct"]:
                    correct_count += 1
                
                # Save intermediate results every 100 questions
                if (i + 1) % 100 == 0:
                    self.save_results(results[:i+1], output_path / f"intermediate_winogrande_{i+1}.json")
                    
            except Exception as e:
                print(f"Error on question {i}: {e}")
                continue
        
        # Calculate final metrics
        total_accuracy = correct_count / len(results) if results else 0
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_questions": len(results),
            "correct_answers": correct_count,
            "overall_accuracy": total_accuracy,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "winogrande_results.json")
        
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
        print("Winogrande Evaluation Results")
        print("="*50)
        print(f"Model: {results['model_path']}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
        
        # Show some examples of correct/incorrect answers
        correct_examples = [r for r in results["detailed_results"] if r["is_correct"]][:3]
        incorrect_examples = [r for r in results["detailed_results"] if not r["is_correct"]][:3]
        
        print(f"\nSample Correct Answers:")
        for i, example in enumerate(correct_examples, 1):
            print(f"  {i}. {example['sentence'][:80]}...")
            print(f"     Correct: {example['correct_answer']}, Predicted: {example['predicted_answer']}")
        
        print(f"\nSample Incorrect Answers:")
        for i, example in enumerate(incorrect_examples, 1):
            print(f"  {i}. {example['sentence'][:80]}...")
            print(f"     Correct: {example['correct_answer']}, Predicted: {example['predicted_answer']}")
        
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Winogrande dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = WinograndeEvaluator(args.model, args.device)
    dataset = evaluator.load_winogrande_dataset()
    results = evaluator.evaluate_dataset(dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
