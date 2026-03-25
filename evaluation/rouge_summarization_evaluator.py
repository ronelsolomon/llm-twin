"""
ROUGE Summarization Evaluation Script for LLM Twin
Evaluates summarization quality using ROUGE metrics (n-gram overlap)
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

# ROUGE implementation
class ROUGEEvaluator:
    def __init__(self):
        self.rouge_types = ['rouge-1', 'rouge-2', 'rouge-l']
    
    def tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Convert to lowercase and split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        return tokens
    
    def get_ngrams(self, tokens: List[str], n: int) -> set:
        """Get n-grams from tokens"""
        if len(tokens) < n:
            return set()
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = ' '.join(tokens[i:i+n])
            ngrams.append(ngram)
        return set(ngrams)
    
    def lcs(self, seq1: List[str], seq2: List[str]) -> int:
        """Longest Common Subsequence length"""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def rouge_n(self, reference: str, candidate: str, n: int) -> float:
        """Calculate ROUGE-N score"""
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        ref_ngrams = self.get_ngrams(ref_tokens, n)
        cand_ngrams = self.get_ngrams(cand_tokens, n)
        
        if not ref_ngrams:
            return 0.0
        
        overlap = len(ref_ngrams.intersection(cand_ngrams))
        precision = overlap / len(cand_ngrams) if cand_ngrams else 0.0
        recall = overlap / len(ref_ngrams)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def rouge_l(self, reference: str, candidate: str) -> float:
        """Calculate ROUGE-L score (Longest Common Subsequence)"""
        ref_tokens = self.tokenize(reference)
        cand_tokens = self.tokenize(candidate)
        
        if not ref_tokens or not cand_tokens:
            return 0.0
        
        lcs_length = self.lcs(ref_tokens, cand_tokens)
        precision = lcs_length / len(cand_tokens)
        recall = lcs_length / len(ref_tokens)
        
        # F1 score
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        
        return f1
    
    def evaluate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate all ROUGE scores"""
        scores = {}
        
        # ROUGE-1 (unigrams)
        scores['rouge-1'] = self.rouge_n(reference, candidate, 1)
        
        # ROUGE-2 (bigrams)
        scores['rouge-2'] = self.rouge_n(reference, candidate, 2)
        
        # ROUGE-L (LCS)
        scores['rouge-l'] = self.rouge_l(reference, candidate)
        
        return scores

class SummarizationEvaluator:
    def __init__(self, model_path: str, device: str = "auto"):
        self.model_path = model_path
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.rouge_evaluator = ROUGEEvaluator()
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
    
    def load_summarization_dataset(self) -> Dict:
        """Load summarization dataset with reference summaries"""
        print("Loading summarization dataset...")
        
        # Create synthetic summarization dataset
        dataset = [
            {
                "source_text": "The Internet of Things (IoT) refers to the network of physical devices, vehicles, home appliances, and other items embedded with sensors, software, and other technologies that enable these objects to connect and exchange data over the internet. These devices range from ordinary household objects to sophisticated industrial tools. Experts estimate that there will be over 75 billion IoT devices by 2025, transforming how we interact with the world around us.",
                "reference_summary": "IoT is a network of connected physical devices that can exchange data over the internet, with experts predicting over 75 billion devices by 2025.",
                "category": "technology",
                "source": "cnn_dm"
            },
            {
                "source_text": "Climate change refers to long-term shifts in global temperatures and weather patterns. While climate change is natural, human activities have been the main driver since the 1950s. The burning of fossil fuels generates greenhouse gas emissions that act like a blanket wrapped around Earth, trapping heat and raising temperatures. This leads to various effects including rising sea levels, extreme weather events, and ecosystem disruption.",
                "reference_summary": "Human activities have been the main driver of climate change since the 1950s, primarily through burning fossil fuels that trap heat and cause rising temperatures, sea levels, and extreme weather.",
                "category": "environment",
                "source": "xsum"
            },
            {
                "source_text": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves. The process of learning begins with observations or data, such as examples, direct experience, or instruction, in order to look for patterns in data and make better decisions in the future.",
                "reference_summary": "Machine learning allows systems to learn from data without explicit programming by identifying patterns to improve future decision-making.",
                "category": "technology",
                "source": "cnn_dm"
            },
            {
                "source_text": "The human brain is the most complex organ in the body, containing approximately 86 billion neurons. These neurons communicate through trillions of connections called synapses. The brain is responsible for controlling thoughts, memory, emotion, touch, motor skills, vision, breathing, temperature, and every process that regulates our body. It weighs about three pounds and uses about 20% of the body's energy.",
                "reference_summary": "The human brain contains 86 billion neurons with trillions of connections, controlling all bodily functions and using 20% of the body's energy despite weighing only three pounds.",
                "category": "science",
                "source": "xsum"
            },
            {
                "source_text": "Renewable energy sources are naturally replenished on a human timescale, unlike fossil fuels which are finite. The main types of renewable energy include solar, wind, hydroelectric, geothermal, and biomass. These sources are becoming increasingly important as the world seeks to reduce carbon emissions and combat climate change. Many countries are investing heavily in renewable energy infrastructure.",
                "reference_summary": "Renewable energy sources like solar, wind, and hydroelectric are naturally replenished and increasingly important for reducing carbon emissions and combating climate change.",
                "category": "environment",
                "source": "cnn_dm"
            },
            {
                "source_text": "Social media platforms have transformed how people communicate and share information. These platforms allow users to create and share content, participate in social networking, and engage with communities worldwide. While social media has connected people globally, it has also raised concerns about privacy, mental health, and the spread of misinformation. The impact of social media continues to evolve as technology advances.",
                "reference_summary": "Social media has transformed global communication and information sharing, while raising concerns about privacy, mental health, and misinformation as the technology continues to evolve.",
                "category": "social",
                "source": "xsum"
            },
            {
                "source_text": "Artificial intelligence has made significant advances in recent years, particularly in areas like natural language processing, computer vision, and machine learning. AI systems can now perform tasks that previously required human intelligence, such as understanding speech, recognizing images, and making complex decisions. However, AI also raises important ethical questions about bias, privacy, and the future of work.",
                "reference_summary": "AI has advanced significantly in language processing, vision, and decision-making, but raises ethical concerns about bias, privacy, and employment impacts.",
                "category": "technology",
                "source": "cnn_dm"
            },
            {
                "source_text": "Exercise provides numerous health benefits, including improved cardiovascular health, stronger muscles and bones, better mental health, and increased longevity. Regular physical activity can help prevent chronic diseases like diabetes, heart disease, and certain types of cancer. Health experts recommend at least 150 minutes of moderate exercise per week for optimal health benefits.",
                "reference_summary": "Regular exercise provides significant health benefits including improved cardiovascular health, disease prevention, and mental well-being, with experts recommending 150 minutes weekly.",
                "category": "health",
                "source": "xsum"
            },
            {
                "source_text": "Quantum computing represents a fundamental shift in how computers process information. Unlike classical computers that use bits representing 0 or 1, quantum computers use quantum bits or qubits that can exist in superposition. This allows quantum computers to process multiple possibilities simultaneously, potentially solving certain problems exponentially faster than classical computers. The field is still in early stages but shows promise for cryptography, drug discovery, and optimization problems.",
                "reference_summary": "Quantum computing uses qubits in superposition to process multiple possibilities simultaneously, potentially solving certain problems exponentially faster than classical computers in areas like cryptography and drug discovery.",
                "category": "science",
                "source": "cnn_dm"
            },
            {
                "source_text": "Global trade has increased dramatically over the past century, connecting economies worldwide. International trade allows countries to specialize in what they produce most efficiently and import goods and services they need. While trade has lifted millions out of poverty, it also creates challenges such as job displacement, environmental concerns, and trade disputes between nations. Trade agreements and organizations like the WTO help regulate international commerce.",
                "reference_summary": "Global trade connects economies worldwide, allowing specialization and lifting millions from poverty, while creating challenges like job displacement and environmental concerns that require international regulation.",
                "category": "economics",
                "source": "xsum"
            }
        ]
        
        return dataset
    
    def generate_summary(self, source_text: str, max_length: int = 100) -> str:
        """Generate summary for given text"""
        prompt = f"""Please summarize the following text in a concise way:

{source_text}

Summary:"""
        
        # Tokenize
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Create attention mask
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_length,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        summary = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return summary.strip()
    
    def evaluate_single_summary(self, source_text: str, reference_summary: str, category: str, source: str) -> Dict:
        """Evaluate model on a single summarization task"""
        
        # Generate summary
        generated_summary = self.generate_summary(source_text)
        
        # Calculate ROUGE scores
        rouge_scores = self.rouge_evaluator.evaluate_rouge(reference_summary, generated_summary)
        
        # Additional quality metrics
        quality_metrics = {}
        
        # Length ratio (generated vs reference)
        gen_length = len(generated_summary.split())
        ref_length = len(reference_summary.split())
        length_ratio = gen_length / max(ref_length, 1)
        quality_metrics["length_ratio"] = length_ratio
        
        # Compression ratio (summary vs source)
        source_length = len(source_text.split())
        compression_ratio = gen_length / max(source_length, 1)
        quality_metrics["compression_ratio"] = compression_ratio
        
        # Content coverage (simple word overlap)
        source_words = set(source_text.lower().split())
        summary_words = set(generated_summary.lower().split())
        coverage = len(summary_words.intersection(source_words)) / max(len(summary_words), 1)
        quality_metrics["content_coverage"] = coverage
        
        # Fluency (simple check - no repeated phrases)
        sentences = generated_summary.split('.')
        unique_sentences = len(set(sentences))
        fluency = unique_sentences / max(len(sentences), 1)
        quality_metrics["fluency"] = fluency
        
        # Overall quality score
        rouge_avg = sum(rouge_scores.values()) / len(rouge_scores)
        quality_avg = sum(quality_metrics.values()) / len(quality_metrics)
        overall_score = (rouge_avg * 0.7 + quality_avg * 0.3)  # Weight ROUGE more heavily
        
        return {
            "source_text": source_text,
            "reference_summary": reference_summary,
            "generated_summary": generated_summary,
            "category": category,
            "source": source,
            "rouge_scores": rouge_scores,
            "quality_metrics": quality_metrics,
            "overall_score": overall_score
        }
    
    def evaluate_dataset(self, dataset, num_samples: int = None, output_dir: str = "eval_results") -> Dict:
        """Evaluate on the entire dataset or sample"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if num_samples:
            dataset = dataset[:min(num_samples, len(dataset))]
        
        results = []
        category_scores = {}
        source_scores = {}
        
        print(f"Evaluating on {len(dataset)} summarization tasks...")
        
        for i, example in enumerate(tqdm(dataset)):
            try:
                result = self.evaluate_single_summary(
                    example["source_text"],
                    example["reference_summary"],
                    example["category"],
                    example["source"]
                )
                
                result["example_id"] = i
                results.append(result)
                
                # Track category scores
                category = example["category"]
                if category not in category_scores:
                    category_scores[category] = {"total": 0, "score_sum": 0, "rouge_sum": {}}
                category_scores[category]["total"] += 1
                category_scores[category]["score_sum"] += result["overall_score"]
                
                # Track ROUGE scores by category
                for rouge_type, score in result["rouge_scores"].items():
                    if rouge_type not in category_scores[category]["rouge_sum"]:
                        category_scores[category]["rouge_sum"][rouge_type] = 0
                    category_scores[category]["rouge_sum"][rouge_type] += score
                
                # Track source scores
                source = example["source"]
                if source not in source_scores:
                    source_scores[source] = {"total": 0, "score_sum": 0}
                source_scores[source]["total"] += 1
                source_scores[source]["score_sum"] += result["overall_score"]
                
                # Save intermediate results every 5 summaries
                if (i + 1) % 5 == 0:
                    self.save_results(results[:i+1], output_path / f"intermediate_summarization_{i+1}.json")
                    
            except Exception as e:
                print(f"Error on example {i}: {e}")
                continue
        
        # Calculate final metrics
        total_examples = len(results)
        overall_score = sum(r["overall_score"] for r in results) / total_examples if results else 0
        
        # Calculate ROUGE averages
        rouge_averages = {}
        for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']:
            rouge_scores = [r["rouge_scores"][rouge_type] for r in results]
            rouge_averages[rouge_type] = sum(rouge_scores) / len(rouge_scores) if rouge_scores else 0
        
        # Calculate category-wise averages
        category_averages = {}
        for category, stats in category_scores.items():
            category_averages[category] = {
                "overall": stats["score_sum"] / stats["total"],
                "rouge": {}
            }
            for rouge_type in stats["rouge_sum"]:
                category_averages[category]["rouge"][rouge_type] = stats["rouge_sum"][rouge_type] / stats["total"]
        
        # Calculate source-wise averages
        source_averages = {}
        for source, stats in source_scores.items():
            source_averages[source] = stats["score_sum"] / stats["total"]
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_examples": total_examples,
            "overall_score": overall_score,
            "rouge_averages": rouge_averages,
            "category_averages": category_averages,
            "category_stats": category_scores,
            "source_averages": source_averages,
            "source_stats": source_scores,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "summarization_results.json")
        
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
        print("ROUGE Summarization Evaluation Results")
        print("="*60)
        print(f"Model: {results['model_path']}")
        print(f"Total Examples: {results['total_examples']}")
        print(f"Overall Score: {results['overall_score']:.3f}/1.0")
        print("\nROUGE Scores:")
        for rouge_type, score in results['rouge_averages'].items():
            print(f"  {rouge_type.upper()}: {score:.3f}")
        
        print("\nCategory-wise Performance:")
        for category, scores in sorted(results['category_averages'].items(), 
                                      key=lambda x: x[1]['overall'], reverse=True):
            total = results['category_stats'][category]['total']
            print(f"  {category}: {scores['overall']:.3f} ({total} examples)")
            print(f"    ROUGE-1: {scores['rouge']['rouge-1']:.3f}")
            print(f"    ROUGE-2: {scores['rouge']['rouge-2']:.3f}")
            print(f"    ROUGE-L: {scores['rouge']['rouge-l']:.3f}")
        
        print("\nSource-wise Performance:")
        for source, score in sorted(results['source_averages'].items(), key=lambda x: x[1], reverse=True):
            total = results['source_stats'][source]['total']
            print(f"  {source}: {score:.3f} ({total} examples)")
        
        # Show examples
        sorted_results = sorted(results["detailed_results"], key=lambda x: x["overall_score"], reverse=True)
        best_summary = sorted_results[0]
        worst_summary = sorted_results[-1]
        
        print(f"\nBest Performing Summary:")
        print(f"  Category: {best_summary['category']}, Source: {best_summary['source']}")
        print(f"  Overall Score: {best_summary['overall_score']:.3f}")
        print(f"  ROUGE-1: {best_summary['rouge_scores']['rouge-1']:.3f}")
        print(f"  Generated: {best_summary['generated_summary'][:80]}...")
        print(f"  Reference: {best_summary['reference_summary'][:80]}...")
        
        print(f"\nLowest Performing Summary:")
        print(f"  Category: {worst_summary['category']}, Source: {worst_summary['source']}")
        print(f"  Overall Score: {worst_summary['overall_score']:.3f}")
        print(f"  ROUGE-1: {worst_summary['rouge_scores']['rouge-1']:.3f}")
        print(f"  Generated: {worst_summary['generated_summary'][:80]}...")
        print(f"  Reference: {worst_summary['reference_summary'][:80]}...")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on Summarization with ROUGE")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = SummarizationEvaluator(args.model, args.device)
    dataset = evaluator.load_summarization_dataset()
    results = evaluator.evaluate_dataset(dataset, args.samples, args.output)

if __name__ == "__main__":
    main()
