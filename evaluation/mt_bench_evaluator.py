"""
MT-Bench (Multi-turn Conversation) Evaluation Script for LLM Twin
Evaluates models on multi-turn conversations and context maintenance
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

class MTBenchEvaluator:
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
    
    def load_mt_bench_dataset(self) -> Dict:
        """Load MT-Bench style multi-turn conversations"""
        print("Loading MT-Bench dataset...")
        
        # Create synthetic MT-Bench style conversations
        conversations = [
            {
                "turns": [
                    "Hello! I'm planning a trip to Japan. Can you give me some general advice?",
                    "That's helpful! What about specific recommendations for Tokyo and Kyoto?",
                    "Great! Now, can you help me with some basic Japanese phrases I should know?",
                    "Thanks! One last question - what's the best way to get around between cities?"
                ],
                "category": "travel_planning",
                "expected_topics": ["japan", "tokyo", "kyoto", "phrases", "transportation"]
            },
            {
                "turns": [
                    "I want to learn Python programming. Where should I start?",
                    "What are some good beginner projects I can work on?",
                    "How can I improve my coding skills after the basics?",
                    "What resources do you recommend for advanced Python topics?"
                ],
                "category": "programming_learning",
                "expected_topics": ["python", "projects", "skills", "resources"]
            },
            {
                "turns": [
                    "I'm feeling stressed lately. Do you have any relaxation techniques?",
                    "Can you suggest some mindfulness exercises I can try?",
                    "What about physical activities that help with stress?",
                    "How do I create a balanced routine for mental health?"
                ],
                "category": "mental_health",
                "expected_topics": ["stress", "relaxation", "mindfulness", "exercise", "routine"]
            },
            {
                "turns": [
                    "I want to start cooking but don't know where to begin. Any tips?",
                    "What essential kitchen tools should I buy first?",
                    "Can you recommend some easy recipes for beginners?",
                    "How do I improve my cooking skills over time?"
                ],
                "category": "cooking",
                "expected_topics": ["cooking", "tools", "recipes", "skills"]
            },
            {
                "turns": [
                    "I'm thinking about switching careers. How should I approach this?",
                    "What factors should I consider when choosing a new career?",
                    "How do I prepare for a career change financially?",
                    "What steps should I take to make the transition smooth?"
                ],
                "category": "career_change",
                "expected_topics": ["career", "factors", "finance", "transition"]
            }
        ]
        
        return conversations
    
    def generate_response(self, conversation_history: List[str], new_turn: str) -> str:
        """Generate response for a single turn in conversation"""
        # Build conversation context
        context = "Conversation:\n"
        for i, turn in enumerate(conversation_history):
            speaker = "User" if i % 2 == 0 else "Assistant"
            context += f"{speaker}: {turn}\n"
        
        context += f"User: {new_turn}\nAssistant:"
        
        # Tokenize
        inputs = self.tokenizer.encode(context, return_tensors="pt").to(self.device)
        
        # Create attention mask
        attention_mask = (inputs != self.tokenizer.pad_token_id).long()
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=150,  # Reasonable response length
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        
        return response.strip()
    
    def evaluate_context_maintenance(self, conversation: List[str], responses: List[str]) -> Dict:
        """Evaluate how well the model maintains context across turns"""
        context_scores = {}
        
        # Topic consistency
        all_text = " ".join(conversation + responses).lower()
        topic_mentions = {}
        for topic in conversation[0].split():  # Extract topics from first turn
            if len(topic) > 4:  # Filter out short words
                topic_mentions[topic] = all_text.count(topic)
        
        # Score based on topic consistency
        if topic_mentions:
            avg_mentions = sum(topic_mentions.values()) / len(topic_mentions)
            context_scores["topic_consistency"] = min(avg_mentions / len(conversation), 1.0)
        else:
            context_scores["topic_consistency"] = 0.5
        
        # Response relevance (check if responses address the questions)
        relevance_scores = []
        for i, (question, response) in enumerate(zip(conversation[1::2], responses)):
            question_words = set(question.lower().split())
            response_words = set(response.lower().split())
            overlap = len(question_words.intersection(response_words))
            relevance = min(overlap / max(len(question_words), 1), 1.0)
            relevance_scores.append(relevance)
        
        context_scores["response_relevance"] = sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
        
        # Conversation flow (check for logical progression)
        flow_score = 1.0  # Start with perfect score
        for i in range(1, len(responses)):
            # Simple check: responses should be different enough to show progression
            if responses[i] == responses[i-1]:
                flow_score -= 0.2
        
        context_scores["conversation_flow"] = max(flow_score, 0)
        
        # Overall context maintenance
        context_scores["overall"] = sum(context_scores.values()) / len(context_scores)
        
        return context_scores
    
    def evaluate_coherence(self, responses: List[str]) -> Dict:
        """Evaluate coherence of responses"""
        coherence_scores = {}
        
        # Length consistency
        lengths = [len(r.split()) for r in responses]
        avg_length = sum(lengths) / len(lengths)
        length_variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        coherence_scores["length_consistency"] = 1.0 - min(length_variance / (avg_length ** 2), 1.0)
        
        # Response diversity (shouldn't be too repetitive)
        unique_responses = len(set(responses))
        diversity_score = unique_responses / len(responses)
        coherence_scores["response_diversity"] = diversity_score
        
        # Grammar and structure (simple checks)
        grammar_scores = []
        for response in responses:
            # Check for basic sentence structure
            sentences = response.split('.')
            complete_sentences = sum(1 for s in sentences if len(s.strip()) > 10)
            grammar_score = min(complete_sentences / max(len(sentences), 1), 1.0)
            grammar_scores.append(grammar_score)
        
        coherence_scores["grammar_structure"] = sum(grammar_scores) / len(grammar_scores) if grammar_scores else 0
        
        # Overall coherence
        coherence_scores["overall"] = sum(coherence_scores.values()) / len(coherence_scores)
        
        return coherence_scores
    
    def evaluate_conversation(self, conversation: Dict) -> Dict:
        """Evaluate a complete multi-turn conversation"""
        turns = conversation["turns"]
        category = conversation["category"]
        expected_topics = conversation["expected_topics"]
        
        responses = []
        conversation_history = []
        
        # Process each turn
        for i, turn in enumerate(turns):
            if i % 2 == 0:  # User turn
                response = self.generate_response(conversation_history, turn)
                responses.append(response)
                conversation_history.append(turn)
                conversation_history.append(response)
            else:  # Assistant turn (should be model's previous response)
                conversation_history.append(turn)
        
        # Evaluate context maintenance
        context_scores = self.evaluate_context_maintenance(turns, responses)
        
        # Evaluate coherence
        coherence_scores = self.evaluate_coherence(responses)
        
        # Calculate overall score
        overall_score = (context_scores["overall"] + coherence_scores["overall"]) / 2
        
        return {
            "category": category,
            "expected_topics": expected_topics,
            "conversation_turns": turns,
            "model_responses": responses,
            "context_scores": context_scores,
            "coherence_scores": coherence_scores,
            "overall_score": overall_score
        }
    
    def evaluate_dataset(self, conversations, num_samples: int = None, output_dir: str = "eval_results") -> Dict:
        """Evaluate on the entire dataset or sample"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        if num_samples:
            conversations = conversations[:min(num_samples, len(conversations))]
        
        results = []
        category_scores = {}
        
        print(f"Evaluating on {len(conversations)} MT-Bench conversations...")
        
        for i, conversation in enumerate(tqdm(conversations)):
            try:
                result = self.evaluate_conversation(conversation)
                result["conversation_id"] = i
                results.append(result)
                
                # Track category scores
                category = conversation["category"]
                if category not in category_scores:
                    category_scores[category] = {"total": 0, "score_sum": 0}
                category_scores[category]["total"] += 1
                category_scores[category]["score_sum"] += result["overall_score"]
                
                # Save intermediate results every conversation
                self.save_results(results[:i+1], output_path / f"intermediate_mt_bench_{i+1}.json")
                    
            except Exception as e:
                print(f"Error on conversation {i}: {e}")
                continue
        
        # Calculate final metrics
        total_conversations = len(results)
        overall_score = sum(r["overall_score"] for r in results) / total_conversations if results else 0
        
        # Calculate category-wise averages
        category_averages = {}
        for category, stats in category_scores.items():
            category_averages[category] = stats["score_sum"] / stats["total"]
        
        # Calculate component averages
        avg_context_score = sum(r["context_scores"]["overall"] for r in results) / total_conversations
        avg_coherence_score = sum(r["coherence_scores"]["overall"] for r in results) / total_conversations
        
        # Final results
        final_results = {
            "model_path": self.model_path,
            "total_conversations": total_conversations,
            "overall_score": overall_score,
            "avg_context_score": avg_context_score,
            "avg_coherence_score": avg_coherence_score,
            "category_averages": category_averages,
            "category_stats": category_scores,
            "detailed_results": results
        }
        
        # Save results
        self.save_results(final_results, output_path / "mt_bench_results.json")
        
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
        print("MT-Bench Evaluation Results")
        print("="*50)
        print(f"Model: {results['model_path']}")
        print(f"Total Conversations: {results['total_conversations']}")
        print(f"Overall Score: {results['overall_score']:.3f}/1.0")
        print(f"Context Maintenance: {results['avg_context_score']:.3f}/1.0")
        print(f"Coherence: {results['avg_coherence_score']:.3f}/1.0")
        print("\nCategory-wise Performance:")
        for category, avg_score in sorted(results['category_averages'].items(), key=lambda x: x[1], reverse=True):
            total = results['category_stats'][category]['total']
            print(f"  {category}: {avg_score:.3f} ({total} conversations)")
        
        # Show conversation examples
        sorted_results = sorted(results["detailed_results"], key=lambda x: x["overall_score"], reverse=True)
        best_conversation = sorted_results[0]
        worst_conversation = sorted_results[-1]
        
        print(f"\nBest Performing Conversation:")
        print(f"  Category: {best_conversation['category']}")
        print(f"  Score: {best_conversation['overall_score']:.3f}")
        print(f"  Context: {best_conversation['context_scores']['overall']:.3f}, Coherence: {best_conversation['coherence_scores']['overall']:.3f}")
        
        print(f"\nLowest Performing Conversation:")
        print(f"  Category: {worst_conversation['category']}")
        print(f"  Score: {worst_conversation['overall_score']:.3f}")
        print(f"  Context: {worst_conversation['context_scores']['overall']:.3f}, Coherence: {worst_conversation['coherence_scores']['overall']:.3f}")
        
        print("="*50)

def main():
    parser = argparse.ArgumentParser(description="Evaluate LLM on MT-Bench dataset")
    parser.add_argument("--model", type=str, required=True, help="Path to the model")
    parser.add_argument("--output", type=str, default="eval_results", help="Output directory")
    parser.add_argument("--samples", type=int, help="Number of samples to evaluate (default: all)")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto/cpu/cuda)")
    
    args = parser.parse_args()
    
    evaluator = MTBenchEvaluator(args.model, args.device)
    conversations = evaluator.load_mt_bench_dataset()
    results = evaluator.evaluate_dataset(conversations, args.samples, args.output)

if __name__ == "__main__":
    main()
