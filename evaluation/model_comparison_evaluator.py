"""
Model Comparison Evaluator for LLM Twin
Compares multiple models head-to-head on the same tasks
Supports TwinLlama, DPO-Llama, and other model architectures
"""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaForCausalLM
from peft import AutoPeftModelForCausalLM
from tqdm import tqdm
import json
import re
from typing import Dict, List, Tuple
import argparse
from pathlib import Path
import time
from speculative_decoding_utils import SpeculativeDecoder

class ModelComparisonEvaluator:
    def __init__(self, model_configs: List[Dict], device: str = "auto", use_speculative: bool = True):
        self.model_configs = model_configs
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_speculative = use_speculative
        self.models = {}
        self.tokenizers = {}
        self.speculative_decoders = {}
        self.load_models()
        
    def load_models(self):
        """Load all models for comparison"""
        print("Loading models for comparison...")
        for config in self.model_configs:
            model_path = config["path"]
            model_name = config["name"]
            
            print(f"Loading {model_name} from {model_path}")
            
            # Check if this is a LoRA adapter (has adapter_config.json)
            adapter_config_path = Path(model_path) / "adapter_config.json"
            model_config_path = Path(model_path) / "config.json"
            
            # Load tokenizer
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_path)
            except:
                # If tokenizer loading fails, try to get it from base model
                if adapter_config_path.exists():
                    with open(adapter_config_path, 'r') as f:
                        adapter_config = json.load(f)
                    base_model = adapter_config.get("base_model_name_or_path")
                    if base_model:
                        print(f"  Loading tokenizer from base model: {base_model}")
                        tokenizer = AutoTokenizer.from_pretrained(base_model)
                    else:
                        raise
                else:
                    raise
                    
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with support for different architectures
            if adapter_config_path.exists():
                # This is a LoRA adapter, use AutoPeftModelForCausalLM
                print(f"  Loading LoRA adapter for {model_name}")
                model = AutoPeftModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto" if self.device == "cuda" else None
                )
                if self.device == "cpu":
                    model = model.to(self.device)
                print(f"  Using AutoPeftModelForCausalLM for {model_name}")
            else:
                # Check model type from config to determine architecture
                model_type = None
                if model_config_path.exists():
                    with open(model_config_path, 'r') as f:
                        model_config = json.load(f)
                    model_type = model_config.get("model_type", "").lower()
                    print(f"  Detected model type: {model_type}")
                
                # Load based on detected architecture
                if model_type == "gpt2":
                    from transformers import GPT2LMHeadModel
                    model = GPT2LMHeadModel.from_pretrained(
                        model_path,
                        torch_dtype=torch.float32,  # Use float32 for quantized models
                        device_map="auto" if self.device == "cuda" else None,
                        low_cpu_mem_usage=True
                    )
                    if self.device == "cpu":
                        model = model.to(self.device)
                    print(f"  Using GPT2LMHeadModel for {model_name}")
                elif model_type == "llama" or "llama" in model_path.lower():
                    # Use LlamaForCausalLM for Llama models
                    model = LlamaForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto" if self.device == "cuda" else None,
                        ignore_mismatched_sizes=True
                    )
                    if self.device == "cpu":
                        model = model.to(self.device)
                    print(f"  Using LlamaForCausalLM for {model_name}")
                else:
                    # Fallback to standard AutoModelForCausalLM
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="auto" if self.device == "cuda" else None,
                        ignore_mismatched_sizes=True
                    )
                    if self.device == "cpu":
                        model = model.to(self.device)
                    print(f"  Using AutoModelForCausalLM for {model_name}")
            
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            
            # Initialize speculative decoder if enabled
            if self.use_speculative:
                try:
                    print(f"  Initializing speculative decoder for {model_name}")
                    speculative_decoder = SpeculativeDecoder(model, tokenizer, self.device)
                    speculative_decoder.load_draft_model("gpt2")  # Use GPT-2 as draft model
                    self.speculative_decoders[model_name] = speculative_decoder
                    print(f"  Speculative decoder ready for {model_name}")
                except Exception as e:
                    print(f"  Failed to initialize speculative decoder for {model_name}: {e}")
                    self.speculative_decoders[model_name] = None
            
        print(f"All models loaded on device: {self.device}")
    
    def load_comparison_dataset(self) -> Dict:
        """Load dataset for model comparison"""
        print("Loading comparison dataset...")
        
        # Create diverse comparison tasks
        dataset = [
            {
                "task_type": "blog_post",
                "instruction": "Write a blog post about the benefits of renewable energy for a general audience. Keep it engaging and informative, around 300 words.",
                "expected_length": (250, 350),
                "criteria": ["engagement", "informativeness", "clarity"]
            },
            {
                "task_type": "technical_explanation",
                "instruction": "Explain how machine learning works to someone with a technical background. Include key concepts and practical examples.",
                "expected_length": (200, 400),
                "criteria": ["accuracy", "technical_depth", "clarity"]
            },
            {
                "task_type": "creative_writing",
                "instruction": "Write a short story about an AI that becomes self-aware. Focus on character development and emotional depth.",
                "expected_length": (400, 600),
                "criteria": ["creativity", "narrative_coherence", "emotional_impact"]
            },
            {
                "task_type": "code_generation",
                "instruction": "Write a Python function that implements a binary search algorithm. Include comments and error handling.",
                "expected_length": (50, 150),
                "criteria": ["correctness", "efficiency", "code_style"]
            },
            {
                "task_type": "summarization",
                "instruction": "Summarize the following article about climate change in 3-4 sentences: 'Climate change refers to long-term shifts in global temperatures and weather patterns. Human activities have been the main driver since the 1950s, primarily through the emission of greenhouse gases.'",
                "expected_length": (50, 100),
                "criteria": ["completeness", "accuracy", "conciseness"]
            },
            {
                "task_type": "question_answering",
                "instruction": "What are the main differences between supervised and unsupervised machine learning? Provide specific examples.",
                "expected_length": (150, 300),
                "criteria": ["accuracy", "completeness", "clarity"]
            },
            {
                "task_type": "instruction_following",
                "instruction": "Write exactly 5 sentences about artificial intelligence. Each sentence must start with a different letter: A, B, C, D, E.",
                "expected_length": (100, 150),
                "criteria": ["constraint_compliance", "creativity", "coherence"]
            }
        ]
        
        return dataset
    
    def generate_response(self, model_name: str, instruction: str, max_tokens: int = 512, use_speculative: bool = None) -> str:
        """Generate response from a specific model with optional speculative decoding"""
        model = self.models[model_name]
        tokenizer = self.tokenizers[model_name]
        
        # Determine whether to use speculative decoding
        if use_speculative is None:
            use_speculative = self.use_speculative
        else:
            use_speculative = use_speculative and self.use_speculative
        
        # Tokenize
        inputs = tokenizer.encode(instruction, return_tensors="pt").to(self.device)
        attention_mask = (inputs != tokenizer.pad_token_id).long()
        
        start_time = time.time()
        
        # Use speculative decoding if available and enabled
        if (use_speculative and 
            model_name in self.speculative_decoders and 
            self.speculative_decoders[model_name] is not None):
            
            try:
                print(f"  Using speculative decoding for {model_name}")
                outputs = self.speculative_decoders[model_name].speculative_generate(
                    inputs, attention_mask, max_tokens, temperature=0.7
                )
                generation_time = time.time() - start_time
                print(f"  Speculative generation time: {generation_time:.2f}s")
                
            except Exception as e:
                print(f"  Speculative decoding failed for {model_name}: {e}, falling back to regular generation")
                outputs = self._regular_generate(model, tokenizer, inputs, attention_mask, max_tokens)
        else:
            # Regular generation
            outputs = self._regular_generate(model, tokenizer, inputs, attention_mask, max_tokens)
            generation_time = time.time() - start_time
            print(f"  Regular generation time: {generation_time:.2f}s")
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    
    def _regular_generate(self, model, tokenizer, inputs, attention_mask, max_tokens):
        """Helper method for regular generation"""
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        return outputs
    
    def evaluate_response_quality(self, response: str, task_type: str, criteria: List[str]) -> Dict:
        """Evaluate response quality based on task-specific criteria"""
        quality_scores = {}
        
        # Length appropriateness
        response_length = len(response.split())
        quality_scores["length_score"] = 1.0  # Will be calculated later
        
        # Basic quality metrics
        words = response.split()
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        
        # Task-specific evaluation
        if task_type == "blog_post":
            quality_scores["engagement"] = self.evaluate_engagement(response)
            quality_scores["informativeness"] = self.evaluate_informativeness(response)
            quality_scores["clarity"] = self.evaluate_clarity(response)
            
        elif task_type == "technical_explanation":
            quality_scores["accuracy"] = self.evaluate_technical_accuracy(response)
            quality_scores["technical_depth"] = self.evaluate_technical_depth(response)
            quality_scores["clarity"] = self.evaluate_clarity(response)
            
        elif task_type == "creative_writing":
            quality_scores["creativity"] = self.evaluate_creativity(response)
            quality_scores["narrative_coherence"] = self.evaluate_narrative_coherence(response)
            quality_scores["emotional_impact"] = self.evaluate_emotional_impact(response)
            
        elif task_type == "code_generation":
            quality_scores["correctness"] = self.evaluate_code_correctness(response)
            quality_scores["efficiency"] = self.evaluate_code_efficiency(response)
            quality_scores["code_style"] = self.evaluate_code_style(response)
            
        elif task_type == "summarization":
            quality_scores["completeness"] = self.evaluate_completeness(response)
            quality_scores["accuracy"] = self.evaluate_summary_accuracy(response)
            quality_scores["conciseness"] = self.evaluate_conciseness(response)
            
        elif task_type == "question_answering":
            quality_scores["accuracy"] = self.evaluate_qa_accuracy(response)
            quality_scores["completeness"] = self.evaluate_qa_completeness(response)
            quality_scores["clarity"] = self.evaluate_clarity(response)
            
        elif task_type == "instruction_following":
            quality_scores["constraint_compliance"] = self.evaluate_constraint_compliance(response)
            quality_scores["creativity"] = self.evaluate_creativity(response)
            quality_scores["coherence"] = self.evaluate_coherence(response)
        
        # Overall quality score
        if quality_scores:
            quality_scores["overall"] = sum(quality_scores.values()) / len(quality_scores)
        else:
            quality_scores["overall"] = 0.0
        
        return quality_scores
    
    def evaluate_engagement(self, response: str) -> float:
        """Evaluate engagement level of blog post"""
        engagement_indicators = ["you", "imagine", "discover", "learn", "important", "key"]
        indicator_count = sum(1 for indicator in engagement_indicators if indicator in response.lower())
        return min(indicator_count / 5, 1.0)  # Normalize to 0-1
    
    def evaluate_informativeness(self, response: str) -> float:
        """Evaluate how informative the response is"""
        info_indicators = ["because", "therefore", "however", "additionally", "furthermore", "specifically"]
        indicator_count = sum(1 for indicator in info_indicators if indicator in response.lower())
        return min(indicator_count / 3, 1.0)  # Normalize to 0-1
    
    def evaluate_technical_accuracy(self, response: str) -> float:
        """Evaluate technical accuracy of explanation"""
        tech_terms = ["algorithm", "neural", "supervised", "unsupervised", "training", "model", "data"]
        term_count = sum(1 for term in tech_terms if term in response.lower())
        return min(term_count / 3, 1.0)  # Expect at least 3 technical terms
    
    def evaluate_technical_depth(self, response: str) -> float:
        """Evaluate depth of technical explanation"""
        depth_indicators = ["example", "specifically", "instance", "application", "implementation"]
        indicator_count = sum(1 for indicator in depth_indicators if indicator in response.lower())
        return min(indicator_count / 2, 1.0)  # Expect at least 2 depth indicators
    
    def evaluate_creativity(self, response: str) -> float:
        """Evaluate creativity of response"""
        creative_indicators = ["imagine", "unique", "innovative", "perspective", "original"]
        indicator_count = sum(1 for indicator in creative_indicators if indicator in response.lower())
        return min(indicator_count / 3, 1.0)  # Normalize to 0-1
    
    def evaluate_narrative_coherence(self, response: str) -> float:
        """Evaluate narrative coherence"""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) < 2:
            return 0.3
        
        # Check for logical flow
        flow_indicators = ["however", "therefore", "meanwhile", "consequently", "furthermore"]
        flow_count = sum(1 for indicator in flow_indicators if indicator in response.lower())
        return min(flow_count / max(len(sentences) - 1, 1), 1.0)
    
    def evaluate_emotional_impact(self, response: str) -> float:
        """Evaluate emotional impact"""
        emotion_words = ["feel", "emotion", "heart", "soul", "passion", "dream", "hope"]
        emotion_count = sum(1 for word in emotion_words if word in response.lower())
        return min(emotion_count / 2, 1.0)  # Normalize to 0-1
    
    def evaluate_code_correctness(self, response: str) -> float:
        """Evaluate code correctness"""
        code_indicators = ["def", "function", "return", "algorithm", "binary", "search"]
        indicator_count = sum(1 for indicator in code_indicators if indicator in response.lower())
        return min(indicator_count / 4, 1.0)  # Expect key code elements
    
    def evaluate_code_efficiency(self, response: str) -> float:
        """Evaluate code efficiency"""
        efficiency_indicators = ["log", "complexity", "time", "space", "optimal"]
        indicator_count = sum(1 for indicator in efficiency_indicators if indicator in response.lower())
        return min(indicator_count / 2, 1.0)  # Normalize to 0-1
    
    def evaluate_code_style(self, response: str) -> float:
        """Evaluate code style"""
        style_indicators = ["comment", "clear", "readable", "structured", "consistent"]
        indicator_count = sum(1 for indicator in style_indicators if indicator in response.lower())
        return min(indicator_count / 3, 1.0)  # Normalize to 0-1
    
    def evaluate_completeness(self, response: str) -> float:
        """Evaluate summary completeness"""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) >= 3 and len(sentences) <= 5:
            return 1.0
        elif len(sentences) >= 2:
            return 0.7
        else:
            return 0.3
    
    def evaluate_summary_accuracy(self, response: str) -> float:
        """Evaluate summary accuracy"""
        key_concepts = ["climate", "temperature", "greenhouse", "emissions", "1950s"]
        concept_count = sum(1 for concept in key_concepts if concept in response.lower())
        return min(concept_count / 2, 1.0)  # Expect at least 2 key concepts
    
    def evaluate_conciseness(self, response: str) -> float:
        """Evaluate conciseness"""
        words = response.split()
        if len(words) <= 100:
            return 1.0
        elif len(words) <= 150:
            return 0.7
        else:
            return 0.4
    
    def evaluate_qa_accuracy(self, response: str) -> float:
        """Evaluate QA accuracy"""
        qa_indicators = ["supervised", "unsupervised", "labeled", "unlabeled", "algorithm", "data"]
        indicator_count = sum(1 for indicator in qa_indicators if indicator in response.lower())
        return min(indicator_count / 4, 1.0)  # Expect multiple QA indicators
    
    def evaluate_qa_completeness(self, response: str) -> float:
        """Evaluate QA completeness"""
        if len(response.split()) >= 50:
            return 1.0
        elif len(response.split()) >= 30:
            return 0.7
        else:
            return 0.4
    
    def evaluate_clarity(self, response: str) -> float:
        """Evaluate clarity of response"""
        clarity_indicators = ["clear", "simple", "understandable", "organized", "structured"]
        indicator_count = sum(1 for indicator in clarity_indicators if indicator in response.lower())
        return min(indicator_count / 2, 1.0)  # Normalize to 0-1
    
    def evaluate_constraint_compliance(self, response: str) -> float:
        """Evaluate instruction constraint compliance"""
        sentences = [s.strip() for s in response.split('.') if s.strip()]
        if len(sentences) == 5:  # Exactly 5 sentences
            first_letters = [s[0].upper() if s else "" for s in sentences]
            expected_letters = ['A', 'B', 'C', 'D', 'E']
            
            if first_letters == expected_letters:
                return 1.0
            else:
                return 0.5  # Partial compliance
        else:
            return 0.0
    
    def evaluate_coherence(self, response: str) -> float:
        """Evaluate coherence"""
        coherence_indicators = ["therefore", "however", "additionally", "furthermore", "consequently"]
        indicator_count = sum(1 for indicator in coherence_indicators if indicator in response.lower())
        return min(indicator_count / 2, 1.0)  # Normalize to 0-1
    
    def compare_models_on_task(self, task: Dict) -> Dict:
        """Compare all models on a single task"""
        results = {}
        
        for model_name in self.models.keys():
            response = self.generate_response(model_name, task["instruction"])
            quality_scores = self.evaluate_response_quality(response, task["task_type"], task["criteria"])
            
            results[model_name] = {
                "response": response,
                "quality_scores": quality_scores,
                "overall_score": quality_scores.get("overall", 0.0)
            }
        
        return {
            "task": task,
            "model_results": results,
            "winner": max(results.keys(), key=lambda k: results[k]["overall_score"]),
            "ranking": sorted(results.keys(), key=lambda k: results[k]["overall_score"], reverse=True)
        }
    
    def evaluate_dataset(self, dataset, output_dir: str = "eval_results") -> Dict:
        """Compare models on the entire dataset"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = []
        task_type_scores = {}
        
        print(f"Comparing {len(self.models)} models on {len(dataset)} tasks...")
        
        for i, task in enumerate(tqdm(dataset)):
            comparison_result = self.compare_models_on_task(task)
            comparison_result["task_id"] = i
            results.append(comparison_result)
            
            # Track task type scores
            task_type = task["task_type"]
            if task_type not in task_type_scores:
                task_type_scores[task_type] = {"total": 0, "scores": []}
            
            task_type_scores[task_type]["total"] += 1
            for model_name, model_result in comparison_result["model_results"].items():
                task_type_scores[task_type]["scores"].append(model_result["overall_score"])
            
            # Save intermediate results every 5 tasks
            if (i + 1) % 5 == 0:
                self.save_results(results[:i+1], output_path / f"intermediate_comparison_{i+1}.json")
        
        # Calculate final metrics
        total_tasks = len(results)
        
        # Calculate overall model scores
        overall_model_scores = {}
        for model_name in self.models.keys():
            model_scores = [r["model_results"][model_name]["overall_score"] for r in results]
            overall_model_scores[model_name] = sum(model_scores) / len(model_scores)
        
        # Calculate task type averages
        task_type_averages = {}
        for task_type, stats in task_type_scores.items():
            if stats["scores"]:
                task_type_averages[task_type] = sum(stats["scores"]) / len(stats["scores"])
        
        # Calculate win rates
        win_rates = {}
        for model_name in self.models.keys():
            wins = sum(1 for r in results if r["winner"] == model_name)
            win_rates[model_name] = wins / total_tasks
        
        # Final results
        final_results = {
            "model_configs": self.model_configs,
            "total_tasks": total_tasks,
            "overall_model_scores": overall_model_scores,
            "win_rates": win_rates,
            "task_type_averages": task_type_averages,
            "task_type_scores": task_type_scores,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "model_comparison_results.json")
        
        # Print summary
        self.print_summary(final_results)
        
        return final_results
    
    def save_results(self, results: Dict, filepath: Path):
        """Save results to JSON file"""
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"Results saved to {filepath}")
    
    def print_summary(self, results: Dict):
        """Print comparison summary"""
        print("\n" + "="*70)
        print("Model Comparison Results")
        print("="*70)
        
        print(f"Models Compared: {list(results['overall_model_scores'].keys())}")
        print(f"Total Tasks: {results['total_tasks']}")
        
        print("\nOverall Model Rankings:")
        sorted_models = sorted(results['overall_model_scores'].items(), key=lambda x: x[1], reverse=True)
        for rank, (model_name, score) in enumerate(sorted_models, 1):
            win_rate = results['win_rates'][model_name]
            print(f"  {rank}. {model_name}: {score:.3f} (Win Rate: {win_rate:.1%})")
        
        print("\nTask Type Performance:")
        for task_type, avg_score in sorted(results['task_type_averages'].items(), 
                                         key=lambda x: x[1], reverse=True):
            print(f"  {task_type}: {avg_score:.3f}")
        
        # Show example comparisons
        print(f"\nSample Task Comparisons:")
        for i in range(min(3, len(results['detailed_results']))):
            task_result = results['detailed_results'][i]
            print(f"\nTask {i+1}: {task_result['task']['task_type']}")
            print(f"Winner: {task_result['winner']}")
            
            for model_name, model_result in task_result['model_results'].items():
                status = "🏆" if model_name == task_result['winner'] else "  "
                print(f"  {status} {model_name}: {model_result['overall_score']:.3f}")
                print(f"    Response: {model_result['response'][:100]}...")
        
        print("="*70)

def main():
    parser = argparse.ArgumentParser(description="Compare multiple models on diverse tasks")
    parser.add_argument("--models", nargs="+", required=True, help="Model configurations (name:path pairs)")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    # Parse model configurations
    model_configs = []
    for model_config in args.models:
        if ":" in model_config:
            name, path = model_config.split(":", 1)
        else:
            # Default to name from path
            name = model_config.split("/")[-1]
            path = model_config
        
        model_configs.append({"name": name, "path": path})
    
    evaluator = ModelComparisonEvaluator(model_configs, args.device)
    dataset = evaluator.load_comparison_dataset()
    results = evaluator.evaluate_dataset(dataset, args.output)

if __name__ == "__main__":
    main()
