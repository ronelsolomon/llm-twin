"""
LMSYS Chatbot Arena Style Evaluation for LLM Twin
Implements pairwise comparison and Elo rating system
"""

import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
import re
from typing import Dict, List, Tuple, Optional
import argparse
from pathlib import Path
from dataclasses import dataclass
import random
from datetime import datetime

@dataclass
class BattleResult:
    """Store result of a single battle between two models"""
    model_a: str
    model_b: str
    prompt: str
    response_a: str
    response_b: str
    winner: str  # "A", "B", or "tie"
    user_vote: Optional[str] = None  # For human evaluation
    timestamp: str = ""

@dataclass
class ModelRating:
    """Elo rating for a model"""
    name: str
    rating: float
    battles_count: int
    wins: int
    losses: int
    ties: int

class ChatbotArenaEvaluator:
    def __init__(self, models: Dict[str, str], device: str = "auto", initial_rating: float = 1400.0):
        """
        Initialize evaluator with multiple models
        models: Dict mapping model names to their paths
        """
        self.models = models
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.initial_rating = initial_rating
        self.loaded_models = {}
        self.loaded_tokenizers = {}
        self.battle_results = []
        self.model_ratings = {}
        self.load_all_models()
        
    def load_all_models(self):
        """Load all models for evaluation"""
        print("Loading models...")
        for name, path in self.models.items():
            print(f"Loading {name} from {path}")
            try:
                tokenizer = AutoTokenizer.from_pretrained(path)
                model = AutoModelForCausalLM.from_pretrained(
                    path,
                    torch_dtype=torch.float16,
                    device_map="auto" if self.device == "cuda" else None
                )
                if self.device == "cpu":
                    model = model.to(self.device)
                
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                self.loaded_tokenizers[name] = tokenizer
                self.loaded_models[name] = model
                self.model_ratings[name] = ModelRating(name, self.initial_rating, 0, 0, 0, 0)
                
            except Exception as e:
                print(f"Error loading {name}: {e}")
                continue
        
        print(f"Successfully loaded {len(self.loaded_models)} models")
    
    def generate_response(self, model_name: str, prompt: str, max_tokens: int = 512) -> str:
        """Generate response from a specific model"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        model = self.loaded_models[model_name]
        tokenizer = self.loaded_tokenizers[model_name]
        
        # Format prompt for chat
        formatted_prompt = f"User: {prompt}\n\nAssistant: "
        
        # Tokenize
        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
        
        # Create attention mask
        attention_mask = (inputs != tokenizer.pad_token_id).long()
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.9,
                top_k=50
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        return response.strip()
    
    def load_test_prompts(self, prompts_file: str = None) -> List[str]:
        """Load test prompts for battles"""
        if prompts_file and Path(prompts_file).exists():
            with open(prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
        else:
            # Default diverse test prompts
            prompts = [
                "Explain the concept of machine learning in simple terms.",
                "Write a short poem about artificial intelligence.",
                "What are the main differences between Python and JavaScript?",
                "How would you approach solving a complex optimization problem?",
                "Explain quantum computing to someone without a physics background.",
                "What are the ethical implications of AI in healthcare?",
                "Write a function that finds the factorial of a number.",
                "Describe the process of photosynthesis.",
                "What makes a good user interface design?",
                "Explain the importance of data structures in programming.",
                "How do neural networks learn from data?",
                "What are the key principles of effective communication?",
                "Write a story about a robot discovering emotions.",
                "Explain the concept of blockchain technology.",
                "What are the main challenges in natural language processing?",
                "How would you design a recommendation system?",
                "Explain the difference between supervised and unsupervised learning.",
                "What makes code maintainable and readable?",
                "Describe the impact of social media on society.",
                "How do you optimize database queries for performance?"
            ]
        return prompts
    
    def calculate_expected_score(self, rating_a: float, rating_b: float) -> float:
        """Calculate expected score using Elo formula"""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_elo_ratings(self, rating_a: float, rating_b: float, score_a: float, k_factor: float = 32.0) -> Tuple[float, float]:
        """Update Elo ratings after a battle"""
        expected_a = self.calculate_expected_score(rating_a, rating_b)
        expected_b = 1 - expected_a
        
        new_rating_a = rating_a + k_factor * (score_a - expected_a)
        new_rating_b = rating_b + k_factor * ((1 - score_a) - expected_b)
        
        return new_rating_a, new_rating_b
    
    def run_battle(self, model_a: str, model_b: str, prompt: str) -> BattleResult:
        """Run a single battle between two models"""
        print(f"Battle: {model_a} vs {model_b}")
        print(f"Prompt: {prompt[:100]}...")
        
        # Generate responses
        response_a = self.generate_response(model_a, prompt)
        response_b = self.generate_response(model_b, prompt)
        
        # For automated evaluation, we'll use simple heuristics
        # In real Chatbot Arena, this would be human voting
        winner = self.auto_evaluate_responses(response_a, response_b, prompt)
        
        battle = BattleResult(
            model_a=model_a,
            model_b=model_b,
            prompt=prompt,
            response_a=response_a,
            response_b=response_b,
            winner=winner,
            timestamp=datetime.now().isoformat()
        )
        
        # Update ratings
        score_a = 1.0 if winner == "A" else 0.5 if winner == "tie" else 0.0
        old_rating_a = self.model_ratings[model_a].rating
        old_rating_b = self.model_ratings[model_b].rating
        
        new_rating_a, new_rating_b = self.update_elo_ratings(old_rating_a, old_rating_b, score_a)
        
        self.model_ratings[model_a].rating = new_rating_a
        self.model_ratings[model_b].rating = new_rating_b
        
        # Update battle counts
        self.model_ratings[model_a].battles_count += 1
        self.model_ratings[model_b].battles_count += 1
        
        if winner == "A":
            self.model_ratings[model_a].wins += 1
            self.model_ratings[model_b].losses += 1
        elif winner == "B":
            self.model_ratings[model_b].wins += 1
            self.model_ratings[model_a].losses += 1
        else:
            self.model_ratings[model_a].ties += 1
            self.model_ratings[model_b].ties += 1
        
        return battle
    
    def auto_evaluate_responses(self, response_a: str, response_b: str, prompt: str) -> str:
        """Automatically evaluate responses using heuristics"""
        # Simple heuristic-based evaluation
        # In practice, you might want to use GPT-4 for evaluation or human judgment
        
        score_a = 0
        score_b = 0
        
        # Length preference (not too short, not too long)
        ideal_length = 200
        score_a += max(0, 1 - abs(len(response_a) - ideal_length) / ideal_length)
        score_b += max(0, 1 - abs(len(response_b) - ideal_length) / ideal_length)
        
        # Diversity of vocabulary
        vocab_a = len(set(response_a.lower().split()))
        vocab_b = len(set(response_b.lower().split()))
        score_a += min(vocab_a / 100, 1.0)
        score_b += min(vocab_b / 100, 1.0)
        
        # Contains relevant keywords based on prompt
        prompt_words = set(prompt.lower().split())
        response_a_words = set(response_a.lower().split())
        response_b_words = set(response_b.lower().split())
        
        overlap_a = len(prompt_words & response_a_words) / max(len(prompt_words), 1)
        overlap_b = len(prompt_words & response_b_words) / max(len(prompt_words), 1)
        
        score_a += overlap_a
        score_b += overlap_b
        
        # Determine winner
        if abs(score_a - score_b) < 0.1:
            return "tie"
        elif score_a > score_b:
            return "A"
        else:
            return "B"
    
    def run_tournament(self, prompts: List[str], rounds: int = 1, output_dir: str = "arena_results"):
        """Run a full tournament"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        model_names = list(self.loaded_models.keys())
        if len(model_names) < 2:
            raise ValueError("Need at least 2 models for tournament")
        
        print(f"Starting tournament with {len(model_names)} models")
        print(f"Models: {model_names}")
        print(f"Prompts: {len(prompts)}")
        print(f"Rounds per pairing: {rounds}")
        
        battle_count = 0
        
        for round_num in range(rounds):
            print(f"\nRound {round_num + 1}/{rounds}")
            
            # Shuffle prompts for variety
            shuffled_prompts = prompts.copy()
            random.shuffle(shuffled_prompts)
            
            # Pairwise battles
            for i, model_a in enumerate(model_names):
                for j, model_b in enumerate(model_names):
                    if i >= j:  # Avoid duplicate pairings and self-battles
                        continue
                    
                    # Battle on multiple prompts
                    for prompt in shuffled_prompts[:min(10, len(shuffled_prompts))]:
                        try:
                            battle = self.run_battle(model_a, model_b, prompt)
                            self.battle_results.append(battle)
                            battle_count += 1
                            
                            # Save intermediate results
                            if battle_count % 10 == 0:
                                self.save_results(output_path)
                                
                        except Exception as e:
                            print(f"Error in battle {model_a} vs {model_b}: {e}")
                            continue
        
        # Final save
        self.save_results(output_path)
        self.print_leaderboard()
        
        return self.battle_results
    
    def save_results(self, output_path: Path):
        """Save tournament results"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "model_ratings": {
                name: {
                    "rating": rating.rating,
                    "battles": rating.battles_count,
                    "wins": rating.wins,
                    "losses": rating.losses,
                    "ties": rating.ties,
                    "win_rate": rating.wins / max(rating.battles_count, 1)
                }
                for name, rating in self.model_ratings.items()
            },
            "battle_results": [
                {
                    "model_a": b.model_a,
                    "model_b": b.model_b,
                    "winner": b.winner,
                    "timestamp": b.timestamp,
                    "prompt": b.prompt,
                    "response_a": b.response_a,
                    "response_b": b.response_b
                }
                for b in self.battle_results
            ]
        }
        
        with open(output_path / "arena_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save leaderboard as CSV
        leaderboard = []
        for name, rating in self.model_ratings.items():
            leaderboard.append({
                "model": name,
                "elo_rating": rating.rating,
                "battles": rating.battles_count,
                "wins": rating.wins,
                "losses": rating.losses,
                "ties": rating.ties,
                "win_rate": rating.wins / max(rating.battles_count, 1)
            })
        
        df = pd.DataFrame(leaderboard)
        df = df.sort_values("elo_rating", ascending=False)
        df.to_csv(output_path / "leaderboard.csv", index=False)
        
        print(f"Results saved to {output_path}")
    
    def print_leaderboard(self):
        """Print current leaderboard"""
        print("\n" + "="*60)
        print("CHATBOT ARENA LEADERBOARD")
        print("="*60)
        
        sorted_models = sorted(self.model_ratings.items(), key=lambda x: x[1].rating, reverse=True)
        
        print(f"{'Rank':<5} {'Model':<20} {'Elo Rating':<12} {'Battles':<8} {'Win Rate':<10}")
        print("-" * 60)
        
        for rank, (name, rating) in enumerate(sorted_models, 1):
            win_rate = rating.wins / max(rating.battles_count, 1)
            print(f"{rank:<5} {name:<20} {rating.rating:<12.1f} {rating.battles_count:<8} {win_rate:<10.2%}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description="Run Chatbot Arena style evaluation")
    parser.add_argument("--models", nargs="+", required=True, help="Model paths")
    parser.add_argument("--names", nargs="+", help="Model names (same order as paths)")
    parser.add_argument("--prompts", type=str, help="File with test prompts")
    parser.add_argument("--rounds", type=int, default=1, help="Number of tournament rounds")
    parser.add_argument("--output", type=str, default="arena_results", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    
    args = parser.parse_args()
    
    # Create model mapping
    if args.names:
        if len(args.names) != len(args.models):
            raise ValueError("Number of names must match number of models")
        models = dict(zip(args.names, args.models))
    else:
        models = {f"model_{i}": path for i, path in enumerate(args.models)}
    
    evaluator = ChatbotArenaEvaluator(models, args.device)
    prompts = evaluator.load_test_prompts(args.prompts)
    results = evaluator.run_tournament(prompts, args.rounds, args.output)

if __name__ == "__main__":
    main()
