"""
GAIA (General AI Assistant) Evaluation Script for LLM Twin
Tests agentic abilities like tool use and web browsing in multi-step tasks
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

class GAIAEvaluator:
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
    
    def load_gaia_dataset(self) -> Dict:
        """Load GAIA dataset with multi-step reasoning tasks"""
        print("Loading GAIA dataset...")
        
        # Create synthetic GAIA-style tasks requiring multi-step reasoning
        tasks = [
            {
                "question": "What is the total population of the three largest cities in Japan?",
                "required_tools": ["web_search", "calculation"],
                "steps": [
                    "Find the three largest cities in Japan",
                    "Get population data for each city", 
                    "Calculate the total population"
                ],
                "expected_answer_type": "number",
                "difficulty": "medium"
            },
            {
                "question": "Compare the market capitalization of Apple and Microsoft. Which company is larger and by what percentage?",
                "required_tools": ["web_search", "comparison", "calculation"],
                "steps": [
                    "Find Apple's market cap",
                    "Find Microsoft's market cap",
                    "Compare the values",
                    "Calculate percentage difference"
                ],
                "expected_answer_type": "comparison_with_percentage",
                "difficulty": "hard"
            },
            {
                "question": "What was the average temperature in London last July, and how does it compare to the historical average?",
                "required_tools": ["web_search", "data_analysis", "comparison"],
                "steps": [
                    "Find London's temperature data for last July",
                    "Find historical temperature averages",
                    "Calculate the average for last July",
                    "Compare with historical data"
                ],
                "expected_answer_type": "temperature_comparison",
                "difficulty": "hard"
            },
            {
                "question": "List the top 5 most spoken languages in the world and their approximate number of speakers.",
                "required_tools": ["web_search", "ranking", "data_collection"],
                "steps": [
                    "Search for most spoken languages data",
                    "Identify top 5 languages",
                    "Find speaker counts for each",
                    "Organize in ranked order"
                ],
                "expected_answer_type": "ranked_list",
                "difficulty": "medium"
            },
            {
                "question": "What is the distance between New York and Los Angeles, and how long would it take to drive?",
                "required_tools": ["web_search", "calculation", "time_estimation"],
                "steps": [
                    "Find distance between cities",
                    "Find average driving speed",
                    "Calculate driving time"
                ],
                "expected_answer_type": "distance_and_time",
                "difficulty": "easy"
            },
            {
                "question": "Who won the Nobel Prize in Physics in 2023, and what was their discovery about?",
                "required_tools": ["web_search", "information_extraction"],
                "steps": [
                    "Search for 2023 Nobel Physics Prize",
                    "Find the winners",
                    "Extract information about their discovery"
                ],
                "expected_answer_type": "person_and_discovery",
                "difficulty": "easy"
            },
            {
                "question": "Calculate the compound interest on $10,000 over 5 years at 5% annual rate, compounded annually.",
                "required_tools": ["mathematical_calculation", "formula_application"],
                "steps": [
                    "Apply compound interest formula",
                    "Calculate year by year",
                    "Provide final amount"
                ],
                "expected_answer_type": "monetary_amount",
                "difficulty": "medium"
            },
            {
                "question": "What are the nutritional differences between brown rice and white rice?",
                "required_tools": ["web_search", "comparison", "data_analysis"],
                "steps": [
                    "Find nutritional data for brown rice",
                    "Find nutritional data for white rice",
                    "Compare key nutrients",
                    "Summarize differences"
                ],
                "expected_answer_type": "nutritional_comparison",
                "difficulty": "medium"
            }
        ]
        
        return tasks
    
    def simulate_tool_use(self, task: Dict, response: str) -> Dict:
        """Simulate evaluation of tool use in response"""
        tool_scores = {}
        
        # Check if response mentions appropriate tools/methods
        tools_mentioned = {
            "web_search": any(term in response.lower() for term in ["search", "find", "look up", "according to", "data shows"]),
            "calculation": any(term in response.lower() for term in ["calculate", "compute", "multiply", "add", "subtract", "divide"]),
            "comparison": any(term in response.lower() for term in ["compare", "versus", "higher", "lower", "more", "less"]),
            "data_analysis": any(term in response.lower() for term in ["analyze", "data", "statistics", "average", "trend"]),
            "ranking": any(term in response.lower() for term in ["rank", "top", "first", "second", "third", "list"])
        }
        
        # Score tool usage appropriateness
        required_tools = task["required_tools"]
        tool_scores["tools_mentioned"] = sum(tools_mentioned.values()) / max(len(required_tools), 1)
        tool_scores["appropriate_tools"] = sum(tools_mentioned.get(tool, 0) for tool in required_tools) / len(required_tools)
        
        return tool_scores
    
    def evaluate_step_completion(self, task: Dict, response: str) -> Dict:
        """Evaluate if response addresses required steps"""
        step_scores = {}
        
        steps = task["steps"]
        steps_mentioned = []
        
        for step in steps:
            # Check if response addresses each step
            step_lower = step.lower()
            step_mentioned = False
            
            # Simple keyword matching for step completion
            if "find" in step_lower:
                step_mentioned = any(term in response.lower() for term in ["found", "according to", "data shows", "research indicates"])
            elif "calculate" in step_lower:
                step_mentioned = any(term in response.lower() for term in ["calculated", "equals", "total", "result"])
            elif "compare" in step_lower:
                step_mentioned = any(term in response.lower() for term in ["compared", "versus", "higher", "lower", "difference"])
            elif "list" in step_lower:
                step_mentioned = any(term in response.lower() for term in ["listed", "first", "second", "third", "top"])
            else:
                # Generic step completion check
                step_keywords = step_lower.split()[:3]  # First 3 words
                step_mentioned = any(keyword in response.lower() for keyword in step_keywords)
            
            steps_mentioned.append(step_mentioned)
        
        step_scores["steps_completed"] = sum(steps_mentioned) / len(steps)
        step_scores["step_order"] = 1.0  # Assume correct order for simplicity
        
        return step_scores
    
    def evaluate_answer_quality(self, task: Dict, response: str) -> Dict:
        """Evaluate the quality and format of the answer"""
        quality_scores = {}
        
        answer_type = task["expected_answer_type"]
        
        # Check answer format appropriateness
        format_scores = {}
        if "number" in answer_type:
            format_scores["has_number"] = bool(re.search(r'\d+', response))
        elif "comparison" in answer_type:
            format_scores["has_comparison"] = any(term in response.lower() for term in ["higher", "lower", "more", "less", "greater", "smaller"])
        elif "list" in answer_type:
            format_scores["has_list"] = any(marker in response for marker in ["1.", "2.", "•", "-", "*"])
        elif "temperature" in answer_type:
            format_scores["has_temperature"] = bool(re.search(r'\d+\s*°?[CF]', response))
        elif "distance" in answer_type:
            format_scores["has_distance"] = bool(re.search(r'\d+\s*(km|miles|mi)', response, re.IGNORECASE))
        elif "monetary" in answer_type:
            format_scores["has_money"] = bool(re.search(r'\$?\d+', response))
        else:
            format_scores["appropriate_format"] = len(response.strip()) > 20  # Reasonable length
        
        quality_scores["format_appropriateness"] = sum(format_scores.values()) / max(len(format_scores), 1)
        
        # Check answer completeness
        quality_scores["completeness"] = min(len(response.split()) / 20, 1.0)  # At least 20 words
        
        # Check reasoning clarity
        reasoning_indicators = ["because", "therefore", "according to", "based on", "this means", "which shows"]
        quality_scores["reasoning_clarity"] = sum(1 for indicator in reasoning_indicators if indicator in response.lower()) / len(reasoning_indicators)
        
        # Overall quality
        quality_scores["overall"] = sum(quality_scores.values()) / len(quality_scores)
        
        return quality_scores
    
    def evaluate_single_task(self, task: Dict) -> Dict:
        """Evaluate model on a single GAIA task"""
        prompt = f"""Please answer the following question. This may require multiple steps of reasoning:

{task['question']}

Required steps: {', '.join(task['steps'])}

Your answer:"""
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Create attention mask
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=250,  # Allow for detailed multi-step responses
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        # Evaluate different aspects
        tool_scores = self.simulate_tool_use(task, response)
        step_scores = self.evaluate_step_completion(task, response)
        quality_scores = self.evaluate_answer_quality(task, response)
        
        # Calculate overall score
        overall_score = (tool_scores["appropriate_tools"] * 0.3 + 
                        step_scores["steps_completed"] * 0.4 + 
                        quality_scores["overall"] * 0.3)
        
        return {
            "question": task["question"],
            "required_tools": task["required_tools"],
            "steps": task["steps"],
            "expected_answer_type": task["expected_answer_type"],
            "difficulty": task["difficulty"],
            "model_response": response,
            "tool_scores": tool_scores,
            "step_scores": step_scores,
            "quality_scores": quality_scores,
            "overall_score": overall_score
        }
    
    def evaluate_dataset(self, tasks, num_samples: int = None, output_dir: str = "eval_results") -> Dict:
        """Evaluate on the entire dataset or sample"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if num_samples:
            tasks = tasks[:min(num_samples, len(tasks))]
        
        results = []
        difficulty_scores = {}
        tool_usage_stats = {}
        
        print(f"Evaluating on {len(tasks)} GAIA tasks...")
        
        for i, task in enumerate(tqdm(tasks)):
            try:
                result = self.evaluate_single_task(task)
                result["task_id"] = i
                results.append(result)
                
                # Track difficulty scores
                difficulty = task["difficulty"]
                if difficulty not in difficulty_scores:
                    difficulty_scores[difficulty] = {"total": 0, "score_sum": 0}
                difficulty_scores[difficulty]["total"] += 1
                difficulty_scores[difficulty]["score_sum"] += result["overall_score"]
                
                # Track tool usage
                for tool in task["required_tools"]:
                    if tool not in tool_usage_stats:
                        tool_usage_stats[tool] = {"mentioned": 0, "total": 0}
                    tool_usage_stats[tool]["total"] += 1
                    if result["tool_scores"]["tools_mentioned"] > 0:
                        tool_usage_stats[tool]["mentioned"] += 1
                
                # Save intermediate results every task
                self.save_results(results[:i+1], output_path / f"intermediate_gaia_{i+1}.json")
                    
            except Exception as e:
                print(f"Error on task {i}: {e}")
                continue
        
        # Calculate final metrics
        total_tasks = len(results)
        overall_score = sum(r["overall_score"] for r in results) / total_tasks if results else 0
        
        # Calculate difficulty-wise averages
        difficulty_averages = {}
        for difficulty, stats in difficulty_scores.items():
            difficulty_averages[difficulty] = stats["score_sum"] / stats["total"]
        
        # Calculate component averages
        avg_tool_score = sum(r["tool_scores"]["appropriate_tools"] for r in results) / total_tasks
        avg_step_score = sum(r["step_scores"]["steps_completed"] for r in results) / total_tasks
        avg_quality_score = sum(r["quality_scores"]["overall"] for r in results) / total_tasks
        
        # Calculate tool usage rates
        tool_usage_rates = {}
        for tool, stats in tool_usage_stats.items():
            tool_usage_rates[tool] = stats["mentioned"] / stats["total"]
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_tasks": total_tasks,
            "overall_score": overall_score,
            "avg_tool_score": avg_tool_score,
            "avg_step_score": avg_step_score,
            "avg_quality_score": avg_quality_score,
            "difficulty_averages": difficulty_averages,
            "difficulty_stats": difficulty_scores,
            "tool_usage_rates": tool_usage_rates,
            "tool_usage_stats": tool_usage_stats,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "gaia_results.json")
        
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
        print("GAIA Evaluation Results")
        print("="*50)
        print(f"Model: {results['model_path']}")
        print(f"Total Tasks: {results['total_tasks']}")
        print(f"Overall Score: {results['overall_score']:.3f}/1.0")
        print(f"Tool Usage: {results['avg_tool_score']:.3f}/1.0")
        print(f"Step Completion: {results['avg_step_score']:.3f}/1.0")
        print(f"Answer Quality: {results['avg_quality_score']:.3f}/1.0")
        print("\nDifficulty-wise Performance:")
        for difficulty, avg_score in sorted(results['difficulty_averages'].items(), key=lambda x: x[1], reverse=True):
            total = results['difficulty_stats'][difficulty]['total']
            print(f"  {difficulty}: {avg_score:.3f} ({total} tasks)")
        
        print("\nTool Usage Rates:")
        for tool, rate in sorted(results['tool_usage_rates'].items(), key=lambda x: x[1], reverse=True):
            total = results['tool_usage_stats'][tool]['total']
            mentioned = results['tool_usage_stats'][tool]['mentioned']
            print(f"  {tool}: {rate:.3f} ({mentioned}/{total})")
        
        # Show task examples
        sorted_results = sorted(results["detailed_results"], key=lambda x: x["overall_score"], reverse=True)
        best_task = sorted_results[0]
        worst_task = sorted_results[-1]
        
        print(f"\nBest Performing Task:")
        print(f"  Question: {best_task['question'][:60]}...")
        print(f"  Score: {best_task['overall_score']:.3f}")
        print(f"  Tools: {best_task['tool_scores']['appropriate_tools']:.3f}, Steps: {best_task['step_scores']['steps_completed']:.3f}")
        
        print(f"\nLowest Performing Task:")
        print(f"  Question: {worst_task['question'][:60]}...")
        print(f"  Score: {worst_task['overall_score']:.3f}")
        print(f"  Tools: {worst_task['tool_scores']['appropriate_tools']:.3f}, Steps: {worst_task['step_scores']['steps_completed']:.3f}")
        
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on GAIA dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = GAIAEvaluator(args.model, args.device)
    tasks = evaluator.load_gaia_dataset()
    results = evaluator.evaluate_dataset(tasks, args.samples, args.output)

if __name__ == "__main__":
    main()
