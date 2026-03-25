"""
Performance Evaluator for LLM Twin
Measures inference speed, resource usage, and scalability
"""

import torch
import time
import psutil
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import gc
from typing import Dict, List, Tuple
import argparse
from pathlib import Path

class PerformanceEvaluator:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load model for performance testing"""
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
    
    def measure_inference_speed(self, prompt: str, num_runs: int = 10) -> Dict:
        """Measure inference speed (tokens/second)"""
        times = []
        tokens_per_second = []
        
        for _ in range(num_runs):
            # Clear cache
            torch.cuda.empty_cache() if self.device == "cuda" else None
            
            # Tokenize
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Create attention mask
            attention_mask = (inputs != self.tokenizer.pad_token_id).long()
            
            # Measure time
            start_time = time.time()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    attention_mask=attention_mask,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            end_time = time.time()
            inference_time = end_time - start_time
            
            # Calculate tokens generated
            generated_tokens = outputs[0][inputs.shape[1]:].shape[0]
            
            times.append(inference_time)
            tokens_per_second.append(generated_tokens / inference_time if inference_time > 0 else 0)
        
        # Calculate statistics
        avg_time = np.mean(times)
        std_time = np.std(times)
        avg_tokens_per_second = np.mean(tokens_per_second)
        
        return {
            "avg_inference_time": avg_time,
            "std_inference_time": std_time,
            "avg_tokens_per_second": avg_tokens_per_second,
            "total_runs": num_runs
        }
    
    def measure_memory_usage(self) -> Dict:
        """Measure memory usage"""
        if self.device == "cuda":
            # GPU memory usage
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
                memory_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
                memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                
                return {
                    "memory_allocated_gb": memory_allocated,
                    "memory_reserved_gb": memory_reserved,
                    "memory_total_gb": memory_total,
                    "memory_utilization": memory_allocated / memory_total
                }
        else:
            # CPU and system memory
            memory = psutil.virtual_memory()
            process = psutil.Process()
            
            return {
                "system_memory_gb": memory.total / 1024**3,
                "available_memory_gb": memory.available / 1024**3,
                "memory_utilization": (memory.total - memory.available) / memory.total
            }
    
    def measure_scalability(self, base_prompt: str, size_multipliers: List[float]) -> Dict:
        """Measure performance across different input sizes"""
        results = {}
        
        for multiplier in size_multipliers:
            # Scale prompt length
            scaled_length = int(len(base_prompt) * multiplier)
            scaled_prompt = base_prompt[:scaled_length] if len(base_prompt) >= scaled_length else base_prompt
            
            # Measure inference
            perf = self.measure_inference_speed(scaled_prompt, num_runs=3)
            
            results[f"size_multiplier_{multiplier}x"] = {
                "prompt_length": len(scaled_prompt),
                "avg_inference_time": perf["avg_inference_time"],
                "avg_tokens_per_second": perf["avg_tokens_per_second"]
            }
        
        return results
    
    def evaluate_model_performance(self, output_dir: str = "eval_results") -> Dict:
        """Comprehensive performance evaluation"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        print("Running comprehensive performance evaluation...")
        
        # Test prompts of different lengths
        test_prompts = [
            "What is artificial intelligence?",
            "Explain the process of photosynthesis in detail, including the chemical reactions and environmental factors involved.",
            "Write a Python function that implements a bubble sort algorithm with proper error handling and documentation.",
            "Compare and contrast machine learning and deep learning, highlighting their key differences, applications, and limitations.",
            "Describe the impact of social media on modern society, including both positive and negative aspects across different demographics."
        ]
        
        results = {
            "model_path": self.model_path,
            "device": self.device,
            "inference_speed_tests": [],
            "memory_usage": self.measure_memory_usage(),
            "scalability_tests": []
        }
        
        # Test inference speed
        print("Testing inference speed...")
        for i, prompt in enumerate(test_prompts):
            speed_test = self.measure_inference_speed(prompt, num_runs=5)
            speed_test["prompt_length"] = len(prompt.split())
            speed_test["prompt_id"] = i
            results["inference_speed_tests"].append(speed_test)
        
        # Test scalability
        print("Testing scalability...")
        base_prompt = test_prompts[0]  # Use first prompt as base
        size_multipliers = [0.5, 1.0, 1.5, 2.0, 3.0]
        scalability_test = self.measure_scalability(base_prompt, size_multipliers)
        scalability_test["base_prompt_length"] = len(base_prompt.split())
        results["scalability_tests"] = scalability_test
        
        # Calculate summary statistics
        all_speed_tests = results["inference_speed_tests"]
        avg_inference_time = np.mean([t["avg_inference_time"] for t in all_speed_tests])
        avg_tokens_per_second = np.mean([t["avg_tokens_per_second"] for t in all_speed_tests])
        
        # Calculate scalability metrics
        scalability_results = results["scalability_tests"]
        base_performance = scalability_results[f"size_multiplier_1.0x"]["avg_inference_time"]
        
        scalability_scores = {}
        for key, test in scalability_results.items():
            if key.startswith("size_multiplier_"):
                multiplier = float(key.split("_")[-1].replace("x", ""))
                test_time = test["avg_inference_time"]
                scalability_score = base_performance / test_time if test_time > 0 else 1.0
                scalability_scores[f"multiplier_{multiplier}x"] = scalability_score
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "device": self.device,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "inference_performance": {
                "avg_inference_time": avg_inference_time,
                "avg_tokens_per_second": avg_tokens_per_second,
                "detailed_tests": all_speed_tests
            },
            "memory_usage": results["memory_usage"],
            "scalability": {
                "base_performance": base_performance,
                "scalability_scores": scalability_scores,
                "detailed_tests": scalability_results
            }
        }
        
        # Save results
        self.save_results(final_results, output_path / "performance_results.json")
        
        # Print summary
        self.print_summary(final_results)
        
        return final_results
    
    def save_results(self, results: Dict, filepath: Path):
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def print_summary(self, results: Dict):
        """Print performance summary"""
        print("\n" + "="*70)
        print("Performance Evaluation Results")
        print("="*70)
        print(f"Model: {results['model_path']}")
        print(f"Device: {results['device']}")
        print(f"Timestamp: {results['timestamp']}")
        
        print("\nInference Performance:")
        inference_perf = results["inference_performance"]
        print(f"  Average Inference Time: {inference_perf['avg_inference_time']:.3f}s")
        print(f"  Average Tokens/Second: {inference_perf['avg_tokens_per_second']:.1f}")
        
        print("\nMemory Usage:")
        memory = results["memory_usage"]
        if "memory_allocated_gb" in memory:
            print(f"  GPU Memory Allocated: {memory['memory_allocated_gb']:.2f} GB")
            print(f"  GPU Memory Reserved: {memory['memory_reserved_gb']:.2f} GB")
            print(f"  GPU Memory Total: {memory['memory_total_gb']:.2f} GB")
            print(f"  GPU Utilization: {memory['memory_utilization']:.1%}")
        else:
            print(f"  System Memory: {memory['system_memory_gb']:.1f} GB")
            print(f"  Available Memory: {memory['available_memory_gb']:.1f} GB")
            print(f"  Memory Utilization: {memory['memory_utilization']:.1%}")
        
        print("\nScalability:")
        scalability = results["scalability"]
        print(f"  Base Performance (1x): {scalability['base_performance']:.3f}s")
        
        for multiplier, score in scalability['scalability_scores'].items():
            print(f"  {multiplier}: {score:.3f}x performance")
        
        # Performance assessment
        avg_time = inference_perf['avg_inference_time']
        tokens_per_sec = inference_perf['avg_tokens_per_second']
        
        print(f"\nPerformance Assessment:")
        if avg_time < 1.0 and tokens_per_sec > 50:
            print("🏆 EXCELLENT: Fast inference with high throughput")
        elif avg_time < 2.0 and tokens_per_sec > 30:
            print("🎯 GOOD: Solid performance for production use")
        elif avg_time < 5.0 and tokens_per_sec > 20:
            print("📊 FAIR: Acceptable performance with room for improvement")
        else:
            print("⚠️  NEEDS OPTIMIZATION: Slow inference requiring optimization")
        
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Evaluate model performance")
    parser.add_argument("--model", type=str, required=True, help="Path to model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = PerformanceEvaluator(args.model, args.device)
    results = evaluator.evaluate_model_performance(args.output)

if __name__ == "__main__":
    main()
