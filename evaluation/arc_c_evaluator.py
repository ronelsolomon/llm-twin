"""
ARC-C (AI2 Reasoning Challenge) Evaluation Script for LLM Twin
Tests grade-school-level science questions requiring causal reasoning
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

class ARCCEvaluator:
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
    
    def load_arc_c_dataset(self, split: str = "test") -> Dict:
        """Load ARC-C dataset from HuggingFace"""
        print("Loading ARC-C dataset...")
        dataset = load_dataset("allenai/ai2_arc", "ARC-C", split=split)
        return dataset
    
    def format_question(self, question: str, choices: Dict[str, str]) -> str:
        """Format ARC-C question with multiple choice options"""
        prompt = f"""Question: {question}

Choices:
"""
        # Sort choices by label (A, B, C, D...)
        sorted_labels = sorted(choices.keys())
        for label in sorted_labels:
            prompt += f"{label}) {choices[label]}\n"
        
        prompt += "\nThe correct answer is:"
        return prompt
    
    def extract_answer(self, response: str) -> str:
        """Extract the answer letter from model response"""
        # Look for patterns like "A)", "B)", etc.
        patterns = [
            r"([ABCD])\)",
            r"Answer:\s*([ABCD])",
            r"([ABCD])\.",
            r"The correct answer is ([ABCD])",
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
    
    def evaluate_single_question(self, question: str, choices: Dict[str, str], answer_key: str) -> Dict:
        """Evaluate model on a single ARC-C question"""
        prompt = self.format_question(question, choices)
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Create attention mask
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=10,  # Short responses for MCQ
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Extract answer
        predicted_answer = self.extract_answer(response)
        
        # Check if correct
        is_correct = predicted_answer == answer_key
        
        return {
            "question": question,
            "choices": choices,
            "answer_key": answer_key,
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
        subject_scores = {}
        
        print(f"Evaluating on {len(dataset)} ARC-C questions...")
        
        for i, example in enumerate(tqdm(dataset)):
            try:
                result = self.evaluate_single_question(
                    example["question"],
                    example["choices"],
                    example["answerKey"]
                )
                
                result["subject"] = example.get("id", "Unknown").split("_")[0]  # Extract subject from ID
                result["question_id"] = i
                results.append(result)
                
                if result["is_correct"]:
                    correct_count += 1
                
                # Update subject scores
                subject = result["subject"]
                if subject not in subject_scores:
                    subject_scores[subject] = {"correct": 0, "total": 0}
                subject_scores[subject]["total"] += 1
                if result["is_correct"]:
                    subject_scores[subject]["correct"] += 1
                
                # Save intermediate results every 25 questions
                if (i + 1) % 25 == 0:
                    self.save_results(results[:i+1], output_path / f"intermediate_arc_c_{i+1}.json")
                    
            except Exception as e:
                print(f"Error on question {i}: {e}")
                continue
        
        # Calculate final metrics
        total_accuracy = correct_count / len(results) if results else 0
        
        # Calculate subject-wise accuracies
        subject_accuracies = {}
        for subject, scores in subject_scores.items():
            subject_accuracies[subject] = scores["correct"] / scores["total"]
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_questions": len(results),
            "correct_answers": correct_count,
            "overall_accuracy": total_accuracy,
            "subject_accuracies": subject_accuracies,
            "subject_scores": subject_scores,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "arc_c_results.json")
        
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
        print("ARC-C Evaluation Results")
        print("="*50)
        print(f"Model: {results['model_path']}")
        print(f"Total Questions: {results['total_questions']}")
        print(f"Correct Answers: {results['correct_answers']}")
        print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
        print("\nSubject-wise Accuracies:")
        for subject, accuracy in sorted(results['subject_accuracies'].items(), key=lambda x: x[1], reverse=True):
            total = results['subject_scores'][subject]['total']
            correct = results['subject_scores'][subject]['correct']
            print(f"  {subject}: {accuracy:.2%} ({correct}/{total})")
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on ARC-C dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = ARCCEvaluator(args.model, args.device)
    dataset = evaluator.load_arc_c_dataset()
    results = evaluator.evaluate_dataset(dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
